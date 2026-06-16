// src/ops/arithmetic.rs
use crate::autograd::{
    Device, StoragePreference, Tensor, TensorData, TensorStorageOwned, TensorStorageView,
    assert_native_device_support, assert_same_device, is_no_grad, is_strict_device_execution,
};
use crate::ops::cuda;
use crate::ops::fp_kernels::{
    add_bf16_bf16_arch, add_bf16_bf16_to_f32_arch, add_bf16_scalar_to_f32_arch, add_f16_f16_arch,
    add_f16_f16_to_f32_arch, add_f16_scalar_to_f32_arch, add_f32_bf16_to_f32_arch,
    add_f32_f16_to_f32_arch, mul_bf16_bf16_arch, mul_bf16_bf16_to_f32_arch,
    mul_bf16_scalar_to_f32_arch, mul_f16_f16_arch, mul_f16_f16_to_f32_arch,
    mul_f16_scalar_to_f32_arch, mul_f32_bf16_to_f32_arch, mul_f32_f16_to_f32_arch,
    sub_bf16_bf16_arch, sub_bf16_bf16_to_f32_arch, sub_bf16_scalar_to_f32_arch, sub_f16_f16_arch,
    sub_f16_f16_to_f32_arch, sub_f16_scalar_to_f32_arch, sub_f32_bf16_to_f32_arch,
    sub_f32_f16_to_f32_arch, sum_bf16_arch, sum_f16_arch,
};
use crate::ops::int8_kernels::{
    I8RowBroadcast, I8ScaledRow, add_f32_i8_to_f32_arch, add_i8_i8_arch,
    add_i8_i8_row_broadcast_arch, add_i8_i8_row_broadcast_to_f32_arch, add_i8_i8_to_f32_arch,
    add_i8_scalar_to_f32_arch, dynamic_i8_scale, mul_f32_i8_to_f32_arch, mul_i8_i8_arch,
    mul_i8_i8_row_broadcast_arch, mul_i8_i8_row_broadcast_to_f32_arch, mul_i8_i8_to_f32_arch,
    mul_i8_scalar_to_f32_arch, sub_f32_i8_to_f32_arch, sub_i8_i8_arch,
    sub_i8_i8_row_broadcast_arch, sub_i8_i8_row_broadcast_to_f32_arch, sub_i8_i8_to_f32_arch,
    sub_i8_scalar_to_f32_arch, sum_i8_arch,
};
use crate::precision::DType;
use half::{bf16, f16};
use ndarray::{ArrayD, ArrayViewD, IxDyn, Zip};
use std::cell::RefCell;
use std::ops::{Add, Mul, Sub};
use std::rc::Rc;

#[derive(Clone, Copy, Debug)]
enum BinaryOp {
    Add,
    Sub,
    Mul,
}

type NativeMulGradResult = (ArrayD<f32>, ArrayD<f32>, Vec<usize>, Vec<usize>);

fn apply_binary_views(
    lhs: ArrayViewD<'_, f32>,
    rhs: ArrayViewD<'_, f32>,
    op: BinaryOp,
) -> ArrayD<f32> {
    if lhs.shape() == rhs.shape()
        && let (Some(lhs_slice), Some(rhs_slice)) =
            (lhs.as_slice_memory_order(), rhs.as_slice_memory_order())
    {
        let mut out = Vec::with_capacity(lhs_slice.len());
        match op {
            BinaryOp::Add => out.extend(lhs_slice.iter().zip(rhs_slice).map(|(&a, &b)| a + b)),
            BinaryOp::Sub => out.extend(lhs_slice.iter().zip(rhs_slice).map(|(&a, &b)| a - b)),
            BinaryOp::Mul => out.extend(lhs_slice.iter().zip(rhs_slice).map(|(&a, &b)| a * b)),
        }
        return ArrayD::from_shape_vec(IxDyn(lhs.shape()), out)
            .expect("contiguous binary output shape build failed");
    }

    if rhs.ndim() == 1
        && lhs.ndim() >= 2
        && let (Some(&last_dim), Some(lhs_slice), Some(rhs_slice)) = (
            lhs.shape().last(),
            lhs.as_slice_memory_order(),
            rhs.as_slice_memory_order(),
        )
        && rhs_slice.len() == last_dim
        && last_dim > 0
    {
        let mut out = Vec::with_capacity(lhs_slice.len());
        match op {
            BinaryOp::Add => out.extend(
                lhs_slice
                    .iter()
                    .enumerate()
                    .map(|(idx, &a)| a + rhs_slice[idx % last_dim]),
            ),
            BinaryOp::Sub => out.extend(
                lhs_slice
                    .iter()
                    .enumerate()
                    .map(|(idx, &a)| a - rhs_slice[idx % last_dim]),
            ),
            BinaryOp::Mul => out.extend(
                lhs_slice
                    .iter()
                    .enumerate()
                    .map(|(idx, &a)| a * rhs_slice[idx % last_dim]),
            ),
        }
        return ArrayD::from_shape_vec(IxDyn(lhs.shape()), out)
            .expect("last-dim binary output shape build failed");
    }

    match op {
        BinaryOp::Add => (&lhs + &rhs).into_dyn(),
        BinaryOp::Sub => (&lhs - &rhs).into_dyn(),
        BinaryOp::Mul => (&lhs * &rhs).into_dyn(),
    }
}

fn try_binary_no_grad_native_same_shape(
    lhs: &Tensor,
    rhs: &Tensor,
    op: BinaryOp,
) -> Option<Tensor> {
    if lhs.device() != Device::Cpu || rhs.device() != Device::Cpu {
        return None;
    }
    if lhs.dtype() != rhs.dtype() || lhs.shape_vec() != rhs.shape_vec() {
        return None;
    }

    let shape = lhs.shape_vec();
    match lhs.dtype() {
        DType::F16 => lhs.with_storage_view_preferring(StoragePreference::Native, |lhs_view| {
            rhs.with_storage_view_preferring(StoragePreference::Native, |rhs_view| {
                let (TensorStorageView::F16(lhs_view), TensorStorageView::F16(rhs_view)) =
                    (lhs_view, rhs_view)
                else {
                    return None;
                };
                let (Some(lhs_slice), Some(rhs_slice)) = (
                    lhs_view.as_slice_memory_order(),
                    rhs_view.as_slice_memory_order(),
                ) else {
                    return None;
                };

                let mut out = vec![f16::from_f32(0.0); lhs_slice.len()];
                apply_f16_slices_to_f16(lhs_slice, rhs_slice, &mut out, op);
                let data = ArrayD::from_shape_vec(IxDyn(&shape), out)
                    .expect("native f16 binary output shape build failed")
                    .into_shared();
                Some(Tensor::from_shared_f16_no_grad_with_device(
                    data,
                    Device::Cpu,
                ))
            })
        }),
        DType::BF16 => lhs.with_storage_view_preferring(StoragePreference::Native, |lhs_view| {
            rhs.with_storage_view_preferring(StoragePreference::Native, |rhs_view| {
                let (TensorStorageView::BF16(lhs_view), TensorStorageView::BF16(rhs_view)) =
                    (lhs_view, rhs_view)
                else {
                    return None;
                };
                let (Some(lhs_slice), Some(rhs_slice)) = (
                    lhs_view.as_slice_memory_order(),
                    rhs_view.as_slice_memory_order(),
                ) else {
                    return None;
                };

                let mut out = vec![bf16::from_f32(0.0); lhs_slice.len()];
                apply_bf16_slices_to_bf16(lhs_slice, rhs_slice, &mut out, op);
                let data = ArrayD::from_shape_vec(IxDyn(&shape), out)
                    .expect("native bf16 binary output shape build failed")
                    .into_shared();
                Some(Tensor::from_shared_bf16_no_grad_with_device(
                    data,
                    Device::Cpu,
                ))
            })
        }),
        DType::I8 => {
            let (
                TensorStorageOwned::I8(lhs_data, lhs_scale),
                TensorStorageOwned::I8(rhs_data, rhs_scale),
            ) = (lhs.native_storage_owned(), rhs.native_storage_owned())
            else {
                return None;
            };
            let (Some(lhs_slice), Some(rhs_slice)) = (
                lhs_data.as_slice_memory_order(),
                rhs_data.as_slice_memory_order(),
            ) else {
                return None;
            };

            let mut out = vec![0i8; lhs_slice.len()];
            let out_scale = match op {
                BinaryOp::Add => {
                    add_i8_i8_arch(lhs_slice, lhs_scale, rhs_slice, rhs_scale, &mut out)
                }
                BinaryOp::Sub => {
                    sub_i8_i8_arch(lhs_slice, lhs_scale, rhs_slice, rhs_scale, &mut out)
                }
                BinaryOp::Mul => {
                    mul_i8_i8_arch(lhs_slice, lhs_scale, rhs_slice, rhs_scale, &mut out)
                }
            }
            .unwrap_or_else(|| {
                let mut max_abs = 0.0f32;
                for (&a, &b) in lhs_slice.iter().zip(rhs_slice) {
                    let lhs_v = (a as f32) * lhs_scale;
                    let rhs_v = (b as f32) * rhs_scale;
                    let value = match op {
                        BinaryOp::Add => lhs_v + rhs_v,
                        BinaryOp::Sub => lhs_v - rhs_v,
                        BinaryOp::Mul => lhs_v * rhs_v,
                    };
                    max_abs = max_abs.max(value.abs());
                }
                let out_scale = dynamic_i8_scale(max_abs);
                let inv_scale = 1.0 / out_scale;
                for ((dst, &a), &b) in out.iter_mut().zip(lhs_slice).zip(rhs_slice) {
                    let lhs_v = (a as f32) * lhs_scale;
                    let rhs_v = (b as f32) * rhs_scale;
                    let value = match op {
                        BinaryOp::Add => lhs_v + rhs_v,
                        BinaryOp::Sub => lhs_v - rhs_v,
                        BinaryOp::Mul => lhs_v * rhs_v,
                    };
                    *dst = (value * inv_scale).round().clamp(-127.0, 127.0) as i8;
                }
                out_scale
            });
            let data = ArrayD::from_shape_vec(IxDyn(&shape), out)
                .expect("native i8 binary output shape build failed")
                .into_shared();
            Some(Tensor::from_shared_i8_no_grad_with_device(
                data,
                out_scale,
                Device::Cpu,
            ))
        }
        _ => None,
    }
}

fn try_binary_training_native_same_shape_f32(
    lhs: &Tensor,
    rhs: &Tensor,
    op: BinaryOp,
) -> Option<ArrayD<f32>> {
    if lhs.device() != Device::Cpu || rhs.device() != Device::Cpu {
        return None;
    }
    if lhs.dtype() != rhs.dtype() || lhs.shape_vec() != rhs.shape_vec() {
        return None;
    }

    let shape = lhs.shape_vec();
    match lhs.dtype() {
        DType::F16 => lhs.with_storage_view_preferring(StoragePreference::Native, |lhs_view| {
            rhs.with_storage_view_preferring(StoragePreference::Native, |rhs_view| {
                let (TensorStorageView::F16(lhs_view), TensorStorageView::F16(rhs_view)) =
                    (lhs_view, rhs_view)
                else {
                    return None;
                };
                let (Some(lhs_slice), Some(rhs_slice)) = (
                    lhs_view.as_slice_memory_order(),
                    rhs_view.as_slice_memory_order(),
                ) else {
                    return None;
                };

                let mut out = vec![0.0f32; lhs_slice.len()];
                apply_f16_slices_to_f32(lhs_slice, rhs_slice, &mut out, op);
                array_from_shape_vec(&shape, out)
            })
        }),
        DType::BF16 => lhs.with_storage_view_preferring(StoragePreference::Native, |lhs_view| {
            rhs.with_storage_view_preferring(StoragePreference::Native, |rhs_view| {
                let (TensorStorageView::BF16(lhs_view), TensorStorageView::BF16(rhs_view)) =
                    (lhs_view, rhs_view)
                else {
                    return None;
                };
                let (Some(lhs_slice), Some(rhs_slice)) = (
                    lhs_view.as_slice_memory_order(),
                    rhs_view.as_slice_memory_order(),
                ) else {
                    return None;
                };

                let mut out = vec![0.0f32; lhs_slice.len()];
                apply_bf16_slices_to_f32(lhs_slice, rhs_slice, &mut out, op);
                array_from_shape_vec(&shape, out)
            })
        }),
        DType::I8 => {
            let (
                TensorStorageOwned::I8(lhs_data, lhs_scale),
                TensorStorageOwned::I8(rhs_data, rhs_scale),
            ) = (lhs.native_storage_owned(), rhs.native_storage_owned())
            else {
                return None;
            };
            let (Some(lhs_slice), Some(rhs_slice)) = (
                lhs_data.as_slice_memory_order(),
                rhs_data.as_slice_memory_order(),
            ) else {
                return None;
            };

            let mut out = vec![0.0f32; lhs_slice.len()];
            apply_i8_slices_to_f32(lhs_slice, lhs_scale, rhs_slice, rhs_scale, &mut out, op);
            array_from_shape_vec(&shape, out)
        }
        _ => None,
    }
}

#[derive(Clone, Copy)]
enum RowBroadcastSide {
    RhsVector,
    LhsVector,
}

impl RowBroadcastSide {
    fn reverse(self) -> Self {
        match self {
            RowBroadcastSide::RhsVector => RowBroadcastSide::LhsVector,
            RowBroadcastSide::LhsVector => RowBroadcastSide::RhsVector,
        }
    }
}

fn i8_row_broadcast_args<'a>(
    lhs: &'a [i8],
    lhs_scale: f32,
    rhs: &'a [i8],
    rhs_scale: f32,
    row_len: usize,
    side: RowBroadcastSide,
) -> I8RowBroadcast<'a> {
    I8RowBroadcast {
        lhs: I8ScaledRow {
            values: lhs,
            scale: lhs_scale,
        },
        rhs: I8ScaledRow {
            values: rhs,
            scale: rhs_scale,
        },
        last_dim: row_len,
        vector_on_rhs: matches!(side, RowBroadcastSide::RhsVector),
    }
}

fn f16_binary_value(a: f16, b: f16, op: BinaryOp) -> f16 {
    match op {
        BinaryOp::Add => f16::from_f32(a.to_f32() + b.to_f32()),
        BinaryOp::Sub => f16::from_f32(a.to_f32() - b.to_f32()),
        BinaryOp::Mul => f16::from_f32(a.to_f32() * b.to_f32()),
    }
}

fn bf16_binary_value(a: bf16, b: bf16, op: BinaryOp) -> bf16 {
    match op {
        BinaryOp::Add => bf16::from_f32(a.to_f32() + b.to_f32()),
        BinaryOp::Sub => bf16::from_f32(a.to_f32() - b.to_f32()),
        BinaryOp::Mul => bf16::from_f32(a.to_f32() * b.to_f32()),
    }
}

fn apply_f16_slices_to_f16(lhs: &[f16], rhs: &[f16], out: &mut [f16], op: BinaryOp) {
    let handled = match op {
        BinaryOp::Add => add_f16_f16_arch(lhs, rhs, out),
        BinaryOp::Sub => sub_f16_f16_arch(lhs, rhs, out),
        BinaryOp::Mul => mul_f16_f16_arch(lhs, rhs, out),
    };
    if !handled {
        for ((dst, &a), &b) in out.iter_mut().zip(lhs).zip(rhs) {
            *dst = f16_binary_value(a, b, op);
        }
    }
}

fn apply_bf16_slices_to_bf16(lhs: &[bf16], rhs: &[bf16], out: &mut [bf16], op: BinaryOp) {
    let handled = match op {
        BinaryOp::Add => add_bf16_bf16_arch(lhs, rhs, out),
        BinaryOp::Sub => sub_bf16_bf16_arch(lhs, rhs, out),
        BinaryOp::Mul => mul_bf16_bf16_arch(lhs, rhs, out),
    };
    if !handled {
        for ((dst, &a), &b) in out.iter_mut().zip(lhs).zip(rhs) {
            *dst = bf16_binary_value(a, b, op);
        }
    }
}

fn row_broadcast_side(lhs_shape: &[usize], rhs_shape: &[usize]) -> Option<RowBroadcastSide> {
    if lhs_shape.len() >= 2
        && rhs_shape.len() == 1
        && lhs_shape.last().copied() == rhs_shape.first().copied()
    {
        return Some(RowBroadcastSide::RhsVector);
    }
    if rhs_shape.len() >= 2
        && lhs_shape.len() == 1
        && rhs_shape.last().copied() == lhs_shape.first().copied()
    {
        return Some(RowBroadcastSide::LhsVector);
    }
    None
}

fn row_scalar_broadcast_side(
    lhs_shape: &[usize],
    rhs_shape: &[usize],
) -> Option<(usize, usize, bool)> {
    if lhs_shape.len() < 2 || lhs_shape.len() != rhs_shape.len() {
        return None;
    }
    let rank = lhs_shape.len();
    if lhs_shape[..rank - 1] != rhs_shape[..rank - 1] {
        return None;
    }
    let lhs_last = lhs_shape[rank - 1];
    let rhs_last = rhs_shape[rank - 1];
    if lhs_last > 1 && rhs_last == 1 {
        let rows = lhs_shape[..rank - 1].iter().product::<usize>();
        return (rows > 0).then_some((rows, lhs_last, true));
    }
    if rhs_last > 1 && lhs_last == 1 {
        let rows = rhs_shape[..rank - 1].iter().product::<usize>();
        return (rows > 0).then_some((rows, rhs_last, false));
    }
    None
}

fn b1d_1h1_broadcast_side(
    lhs_shape: &[usize],
    rhs_shape: &[usize],
) -> Option<(usize, usize, usize, bool)> {
    if lhs_shape.len() == 3
        && rhs_shape.len() == 3
        && lhs_shape[1] == 1
        && rhs_shape[0] == 1
        && rhs_shape[2] == 1
    {
        let batch = lhs_shape[0];
        let heads = rhs_shape[1];
        let dim = lhs_shape[2];
        if batch > 0 && heads > 0 && dim > 0 {
            return Some((batch, heads, dim, true));
        }
    }
    if lhs_shape.len() == 3
        && rhs_shape.len() == 3
        && lhs_shape[0] == 1
        && lhs_shape[2] == 1
        && rhs_shape[1] == 1
    {
        let batch = rhs_shape[0];
        let heads = lhs_shape[1];
        let dim = rhs_shape[2];
        if batch > 0 && heads > 0 && dim > 0 {
            return Some((batch, heads, dim, false));
        }
    }
    None
}

fn b1d_1hd_broadcast_side(
    lhs_shape: &[usize],
    rhs_shape: &[usize],
) -> Option<(usize, usize, usize, bool)> {
    if lhs_shape.len() == 3
        && rhs_shape.len() == 3
        && lhs_shape[1] == 1
        && rhs_shape[0] == 1
        && lhs_shape[2] == rhs_shape[2]
        && rhs_shape[2] > 0
    {
        let batch = lhs_shape[0];
        let heads = rhs_shape[1];
        let dim = lhs_shape[2];
        if batch > 0 && heads > 0 {
            return Some((batch, heads, dim, true));
        }
    }
    if lhs_shape.len() == 3
        && rhs_shape.len() == 3
        && lhs_shape[0] == 1
        && rhs_shape[1] == 1
        && lhs_shape[2] == rhs_shape[2]
        && lhs_shape[2] > 0
    {
        let batch = rhs_shape[0];
        let heads = lhs_shape[1];
        let dim = lhs_shape[2];
        if batch > 0 && heads > 0 {
            return Some((batch, heads, dim, false));
        }
    }
    None
}

fn try_binary_no_grad_native_row_broadcast(
    lhs: &Tensor,
    rhs: &Tensor,
    op: BinaryOp,
) -> Option<Tensor> {
    if lhs.device() != Device::Cpu || rhs.device() != Device::Cpu || lhs.dtype() != rhs.dtype() {
        return None;
    }
    let lhs_shape = lhs.shape_vec();
    let rhs_shape = rhs.shape_vec();
    let side = row_broadcast_side(&lhs_shape, &rhs_shape)?;
    let out_shape = match side {
        RowBroadcastSide::RhsVector => lhs_shape,
        RowBroadcastSide::LhsVector => rhs_shape,
    };
    let row_len = *out_shape.last()?;
    if row_len == 0 {
        return None;
    }
    let rows = out_shape.iter().product::<usize>() / row_len;

    match lhs.dtype() {
        DType::F16 => lhs.with_storage_view_preferring(StoragePreference::Native, |lhs_view| {
            rhs.with_storage_view_preferring(StoragePreference::Native, |rhs_view| {
                let (TensorStorageView::F16(lhs_view), TensorStorageView::F16(rhs_view)) =
                    (lhs_view, rhs_view)
                else {
                    return None;
                };
                let (Some(lhs_slice), Some(rhs_slice)) = (
                    lhs_view.as_slice_memory_order(),
                    rhs_view.as_slice_memory_order(),
                ) else {
                    return None;
                };
                let mut out = vec![f16::from_f32(0.0); rows * row_len];
                match side {
                    RowBroadcastSide::RhsVector => {
                        for row in 0..rows {
                            let start = row * row_len;
                            apply_f16_slices_to_f16(
                                &lhs_slice[start..start + row_len],
                                rhs_slice,
                                &mut out[start..start + row_len],
                                op,
                            );
                        }
                    }
                    RowBroadcastSide::LhsVector => {
                        for row in 0..rows {
                            let start = row * row_len;
                            apply_f16_slices_to_f16(
                                lhs_slice,
                                &rhs_slice[start..start + row_len],
                                &mut out[start..start + row_len],
                                op,
                            );
                        }
                    }
                }
                let data = ArrayD::from_shape_vec(IxDyn(&out_shape), out)
                    .expect("native row-broadcast f16 binary output shape build failed")
                    .into_shared();
                Some(Tensor::from_shared_f16_no_grad_with_device(
                    data,
                    Device::Cpu,
                ))
            })
        }),
        DType::BF16 => lhs.with_storage_view_preferring(StoragePreference::Native, |lhs_view| {
            rhs.with_storage_view_preferring(StoragePreference::Native, |rhs_view| {
                let (TensorStorageView::BF16(lhs_view), TensorStorageView::BF16(rhs_view)) =
                    (lhs_view, rhs_view)
                else {
                    return None;
                };
                let (Some(lhs_slice), Some(rhs_slice)) = (
                    lhs_view.as_slice_memory_order(),
                    rhs_view.as_slice_memory_order(),
                ) else {
                    return None;
                };
                let mut out = vec![bf16::from_f32(0.0); rows * row_len];
                match side {
                    RowBroadcastSide::RhsVector => {
                        for row in 0..rows {
                            let start = row * row_len;
                            apply_bf16_slices_to_bf16(
                                &lhs_slice[start..start + row_len],
                                rhs_slice,
                                &mut out[start..start + row_len],
                                op,
                            );
                        }
                    }
                    RowBroadcastSide::LhsVector => {
                        for row in 0..rows {
                            let start = row * row_len;
                            apply_bf16_slices_to_bf16(
                                lhs_slice,
                                &rhs_slice[start..start + row_len],
                                &mut out[start..start + row_len],
                                op,
                            );
                        }
                    }
                }
                let data = ArrayD::from_shape_vec(IxDyn(&out_shape), out)
                    .expect("native row-broadcast bf16 binary output shape build failed")
                    .into_shared();
                Some(Tensor::from_shared_bf16_no_grad_with_device(
                    data,
                    Device::Cpu,
                ))
            })
        }),
        DType::I8 => {
            let (
                TensorStorageOwned::I8(lhs_data, lhs_scale),
                TensorStorageOwned::I8(rhs_data, rhs_scale),
            ) = (lhs.native_storage_owned(), rhs.native_storage_owned())
            else {
                return None;
            };
            let (Some(lhs_slice), Some(rhs_slice)) = (
                lhs_data.as_slice_memory_order(),
                rhs_data.as_slice_memory_order(),
            ) else {
                return None;
            };
            let args =
                i8_row_broadcast_args(lhs_slice, lhs_scale, rhs_slice, rhs_scale, row_len, side);
            let mut out = vec![0i8; rows * row_len];
            let arch_scale = match op {
                BinaryOp::Add => add_i8_i8_row_broadcast_arch(args, &mut out),
                BinaryOp::Sub => sub_i8_i8_row_broadcast_arch(args, &mut out),
                BinaryOp::Mul => mul_i8_i8_row_broadcast_arch(args, &mut out),
            };
            if let Some(out_scale) = arch_scale {
                let data = ArrayD::from_shape_vec(IxDyn(&out_shape), out)
                    .expect("native row-broadcast i8 binary output shape build failed")
                    .into_shared();
                return Some(Tensor::from_shared_i8_no_grad_with_device(
                    data,
                    out_scale,
                    Device::Cpu,
                ));
            }

            let value_at = |idx: usize| {
                let (a, b) = match side {
                    RowBroadcastSide::RhsVector => (lhs_slice[idx], rhs_slice[idx % row_len]),
                    RowBroadcastSide::LhsVector => (lhs_slice[idx % row_len], rhs_slice[idx]),
                };
                let lhs_v = (a as f32) * lhs_scale;
                let rhs_v = (b as f32) * rhs_scale;
                match op {
                    BinaryOp::Add => lhs_v + rhs_v,
                    BinaryOp::Sub => lhs_v - rhs_v,
                    BinaryOp::Mul => lhs_v * rhs_v,
                }
            };
            let mut max_abs = 0.0f32;
            for idx in 0..rows * row_len {
                max_abs = max_abs.max(value_at(idx).abs());
            }
            let out_scale = dynamic_i8_scale(max_abs);
            let inv_scale = 1.0 / out_scale;
            for (idx, dst) in out.iter_mut().enumerate() {
                *dst = (value_at(idx) * inv_scale).round().clamp(-127.0, 127.0) as i8;
            }
            let data = ArrayD::from_shape_vec(IxDyn(&out_shape), out)
                .expect("native row-broadcast i8 binary output shape build failed")
                .into_shared();
            Some(Tensor::from_shared_i8_no_grad_with_device(
                data,
                out_scale,
                Device::Cpu,
            ))
        }
        _ => None,
    }
}

fn try_binary_training_native_row_broadcast_f32(
    lhs: &Tensor,
    rhs: &Tensor,
    op: BinaryOp,
) -> Option<ArrayD<f32>> {
    if lhs.device() != Device::Cpu || rhs.device() != Device::Cpu || lhs.dtype() != rhs.dtype() {
        return None;
    }
    let lhs_shape = lhs.shape_vec();
    let rhs_shape = rhs.shape_vec();
    let side = row_broadcast_side(&lhs_shape, &rhs_shape)?;
    let out_shape = match side {
        RowBroadcastSide::RhsVector => lhs_shape,
        RowBroadcastSide::LhsVector => rhs_shape,
    };
    let row_len = *out_shape.last()?;
    if row_len == 0 {
        return array_from_shape_vec(&out_shape, Vec::new());
    }
    let rows = out_shape.iter().product::<usize>() / row_len;
    let mut out = vec![0.0f32; rows * row_len];

    match lhs.dtype() {
        DType::F16 => lhs.with_storage_view_preferring(StoragePreference::Native, |lhs_view| {
            rhs.with_storage_view_preferring(StoragePreference::Native, |rhs_view| {
                let (TensorStorageView::F16(lhs_view), TensorStorageView::F16(rhs_view)) =
                    (lhs_view, rhs_view)
                else {
                    return None;
                };
                let (Some(lhs_slice), Some(rhs_slice)) = (
                    lhs_view.as_slice_memory_order(),
                    rhs_view.as_slice_memory_order(),
                ) else {
                    return None;
                };
                match side {
                    RowBroadcastSide::RhsVector => {
                        for row in 0..rows {
                            let start = row * row_len;
                            apply_f16_slices_to_f32(
                                &lhs_slice[start..start + row_len],
                                rhs_slice,
                                &mut out[start..start + row_len],
                                op,
                            );
                        }
                    }
                    RowBroadcastSide::LhsVector => {
                        for row in 0..rows {
                            let start = row * row_len;
                            apply_f16_slices_to_f32(
                                lhs_slice,
                                &rhs_slice[start..start + row_len],
                                &mut out[start..start + row_len],
                                op,
                            );
                        }
                    }
                }
                array_from_shape_vec(&out_shape, out)
            })
        }),
        DType::BF16 => lhs.with_storage_view_preferring(StoragePreference::Native, |lhs_view| {
            rhs.with_storage_view_preferring(StoragePreference::Native, |rhs_view| {
                let (TensorStorageView::BF16(lhs_view), TensorStorageView::BF16(rhs_view)) =
                    (lhs_view, rhs_view)
                else {
                    return None;
                };
                let (Some(lhs_slice), Some(rhs_slice)) = (
                    lhs_view.as_slice_memory_order(),
                    rhs_view.as_slice_memory_order(),
                ) else {
                    return None;
                };
                match side {
                    RowBroadcastSide::RhsVector => {
                        for row in 0..rows {
                            let start = row * row_len;
                            apply_bf16_slices_to_f32(
                                &lhs_slice[start..start + row_len],
                                rhs_slice,
                                &mut out[start..start + row_len],
                                op,
                            );
                        }
                    }
                    RowBroadcastSide::LhsVector => {
                        for row in 0..rows {
                            let start = row * row_len;
                            apply_bf16_slices_to_f32(
                                lhs_slice,
                                &rhs_slice[start..start + row_len],
                                &mut out[start..start + row_len],
                                op,
                            );
                        }
                    }
                }
                array_from_shape_vec(&out_shape, out)
            })
        }),
        DType::I8 => {
            let (
                TensorStorageOwned::I8(lhs_data, lhs_scale),
                TensorStorageOwned::I8(rhs_data, rhs_scale),
            ) = (lhs.native_storage_owned(), rhs.native_storage_owned())
            else {
                return None;
            };
            let (Some(lhs_slice), Some(rhs_slice)) = (
                lhs_data.as_slice_memory_order(),
                rhs_data.as_slice_memory_order(),
            ) else {
                return None;
            };
            let args =
                i8_row_broadcast_args(lhs_slice, lhs_scale, rhs_slice, rhs_scale, row_len, side);
            let handled = match op {
                BinaryOp::Add => add_i8_i8_row_broadcast_to_f32_arch(args, &mut out),
                BinaryOp::Sub => sub_i8_i8_row_broadcast_to_f32_arch(args, &mut out),
                BinaryOp::Mul => mul_i8_i8_row_broadcast_to_f32_arch(args, &mut out),
            };
            if handled {
                return array_from_shape_vec(&out_shape, out);
            }

            match side {
                RowBroadcastSide::RhsVector => {
                    for row in 0..rows {
                        let start = row * row_len;
                        apply_i8_slices_to_f32(
                            &lhs_slice[start..start + row_len],
                            lhs_scale,
                            rhs_slice,
                            rhs_scale,
                            &mut out[start..start + row_len],
                            op,
                        );
                    }
                }
                RowBroadcastSide::LhsVector => {
                    for row in 0..rows {
                        let start = row * row_len;
                        apply_i8_slices_to_f32(
                            lhs_slice,
                            lhs_scale,
                            &rhs_slice[start..start + row_len],
                            rhs_scale,
                            &mut out[start..start + row_len],
                            op,
                        );
                    }
                }
            }
            array_from_shape_vec(&out_shape, out)
        }
        _ => None,
    }
}

fn try_binary_training_native_f32(lhs: &Tensor, rhs: &Tensor, op: BinaryOp) -> Option<ArrayD<f32>> {
    try_binary_training_native_same_shape_f32(lhs, rhs, op)
        .or_else(|| try_binary_training_native_row_broadcast_f32(lhs, rhs, op))
        .or_else(|| try_mixed_add_sub_native_same_shape_f32(lhs, rhs, op))
        .or_else(|| try_mixed_add_sub_native_row_broadcast_f32(lhs, rhs, op))
        .or_else(|| try_mixed_add_sub_native_row_scalar_f32(lhs, rhs, op))
        .or_else(|| try_mixed_add_sub_native_scalar_broadcast_f32(lhs, rhs, op))
        .or_else(|| try_mixed_mul_native_same_shape_f32(lhs, rhs, op))
        .or_else(|| try_mixed_mul_native_row_broadcast_f32(lhs, rhs, op))
        .or_else(|| try_mixed_mul_native_row_scalar_f32(lhs, rhs, op))
        .or_else(|| try_mixed_mul_native_scalar_broadcast_f32(lhs, rhs, op))
}

fn mixed_add_sub_value(f32_value: f32, lowp_value: f32, lowp_on_lhs: bool, op: BinaryOp) -> f32 {
    match op {
        BinaryOp::Add => f32_value + lowp_value,
        BinaryOp::Sub if lowp_on_lhs => lowp_value - f32_value,
        BinaryOp::Sub => f32_value - lowp_value,
        BinaryOp::Mul => unreachable!("mixed_add_sub_value only handles Add/Sub"),
    }
}

fn apply_mixed_add_sub_f16(
    f32_slice: &[f32],
    lowp_slice: &[f16],
    lowp_on_lhs: bool,
    op: BinaryOp,
    out: &mut [f32],
) {
    let used_arch = match op {
        BinaryOp::Add => add_f32_f16_to_f32_arch(f32_slice, lowp_slice, out),
        BinaryOp::Sub => sub_f32_f16_to_f32_arch(f32_slice, lowp_slice, lowp_on_lhs, out),
        BinaryOp::Mul => false,
    };
    if used_arch {
        return;
    }
    for ((dst, &f32_value), &lowp_value) in out.iter_mut().zip(f32_slice).zip(lowp_slice) {
        *dst = mixed_add_sub_value(f32_value, lowp_value.to_f32(), lowp_on_lhs, op);
    }
}

fn apply_mixed_add_sub_bf16(
    f32_slice: &[f32],
    lowp_slice: &[bf16],
    lowp_on_lhs: bool,
    op: BinaryOp,
    out: &mut [f32],
) {
    let used_arch = match op {
        BinaryOp::Add => add_f32_bf16_to_f32_arch(f32_slice, lowp_slice, out),
        BinaryOp::Sub => sub_f32_bf16_to_f32_arch(f32_slice, lowp_slice, lowp_on_lhs, out),
        BinaryOp::Mul => false,
    };
    if used_arch {
        return;
    }
    for ((dst, &f32_value), &lowp_value) in out.iter_mut().zip(f32_slice).zip(lowp_slice) {
        *dst = mixed_add_sub_value(f32_value, lowp_value.to_f32(), lowp_on_lhs, op);
    }
}

fn apply_mixed_add_sub_i8(
    f32_slice: &[f32],
    lowp_slice: &[i8],
    scale: f32,
    lowp_on_lhs: bool,
    op: BinaryOp,
    out: &mut [f32],
) {
    let used_arch = match op {
        BinaryOp::Add => add_f32_i8_to_f32_arch(f32_slice, lowp_slice, scale, out),
        BinaryOp::Sub => sub_f32_i8_to_f32_arch(f32_slice, lowp_slice, scale, lowp_on_lhs, out),
        BinaryOp::Mul => false,
    };
    if used_arch {
        return;
    }
    for ((dst, &f32_value), &lowp_value) in out.iter_mut().zip(f32_slice).zip(lowp_slice) {
        *dst = mixed_add_sub_value(f32_value, (lowp_value as f32) * scale, lowp_on_lhs, op);
    }
}

fn try_mixed_add_sub_native_same_shape_f32(
    lhs: &Tensor,
    rhs: &Tensor,
    op: BinaryOp,
) -> Option<ArrayD<f32>> {
    if !matches!(op, BinaryOp::Add | BinaryOp::Sub)
        || lhs.device() != Device::Cpu
        || rhs.device() != Device::Cpu
        || lhs.dtype() == rhs.dtype()
        || lhs.shape_vec() != rhs.shape_vec()
    {
        return None;
    }
    let shape = lhs.shape_vec();
    let len = shape.iter().product::<usize>();
    let mut out = vec![0.0f32; len];
    match (lhs.native_storage_owned(), rhs.native_storage_owned()) {
        (TensorStorageOwned::F32(f32_data), TensorStorageOwned::F16(lowp_data)) => {
            apply_mixed_add_sub_f16(
                f32_data.as_slice_memory_order()?,
                lowp_data.as_slice_memory_order()?,
                false,
                op,
                &mut out,
            );
        }
        (TensorStorageOwned::F16(lowp_data), TensorStorageOwned::F32(f32_data)) => {
            apply_mixed_add_sub_f16(
                f32_data.as_slice_memory_order()?,
                lowp_data.as_slice_memory_order()?,
                true,
                op,
                &mut out,
            );
        }
        (TensorStorageOwned::F32(f32_data), TensorStorageOwned::BF16(lowp_data)) => {
            apply_mixed_add_sub_bf16(
                f32_data.as_slice_memory_order()?,
                lowp_data.as_slice_memory_order()?,
                false,
                op,
                &mut out,
            );
        }
        (TensorStorageOwned::BF16(lowp_data), TensorStorageOwned::F32(f32_data)) => {
            apply_mixed_add_sub_bf16(
                f32_data.as_slice_memory_order()?,
                lowp_data.as_slice_memory_order()?,
                true,
                op,
                &mut out,
            );
        }
        (TensorStorageOwned::F32(f32_data), TensorStorageOwned::I8(lowp_data, scale)) => {
            apply_mixed_add_sub_i8(
                f32_data.as_slice_memory_order()?,
                lowp_data.as_slice_memory_order()?,
                scale,
                false,
                op,
                &mut out,
            );
        }
        (TensorStorageOwned::I8(lowp_data, scale), TensorStorageOwned::F32(f32_data)) => {
            apply_mixed_add_sub_i8(
                f32_data.as_slice_memory_order()?,
                lowp_data.as_slice_memory_order()?,
                scale,
                true,
                op,
                &mut out,
            );
        }
        _ => return None,
    }
    array_from_shape_vec(&shape, out)
}

fn try_mixed_mul_native_same_shape_f32(
    lhs: &Tensor,
    rhs: &Tensor,
    op: BinaryOp,
) -> Option<ArrayD<f32>> {
    if !matches!(op, BinaryOp::Mul)
        || lhs.device() != Device::Cpu
        || rhs.device() != Device::Cpu
        || lhs.dtype() == rhs.dtype()
        || lhs.shape_vec() != rhs.shape_vec()
    {
        return None;
    }
    let shape = lhs.shape_vec();
    let len = shape.iter().product::<usize>();
    let mut out = vec![0.0f32; len];
    match (lhs.native_storage_owned(), rhs.native_storage_owned()) {
        (TensorStorageOwned::F32(f32_data), TensorStorageOwned::F16(lowp_data))
        | (TensorStorageOwned::F16(lowp_data), TensorStorageOwned::F32(f32_data)) => {
            let f32_slice = f32_data.as_slice_memory_order()?;
            let lowp_slice = lowp_data.as_slice_memory_order()?;
            mul_f32_f16_slice_to_f32(f32_slice, lowp_slice, &mut out);
        }
        (TensorStorageOwned::F32(f32_data), TensorStorageOwned::BF16(lowp_data))
        | (TensorStorageOwned::BF16(lowp_data), TensorStorageOwned::F32(f32_data)) => {
            let f32_slice = f32_data.as_slice_memory_order()?;
            let lowp_slice = lowp_data.as_slice_memory_order()?;
            mul_f32_bf16_slice_to_f32(f32_slice, lowp_slice, &mut out);
        }
        (TensorStorageOwned::F32(f32_data), TensorStorageOwned::I8(lowp_data, scale))
        | (TensorStorageOwned::I8(lowp_data, scale), TensorStorageOwned::F32(f32_data)) => {
            let f32_slice = f32_data.as_slice_memory_order()?;
            let lowp_slice = lowp_data.as_slice_memory_order()?;
            mul_f32_i8_slice_to_f32(f32_slice, lowp_slice, scale, &mut out);
        }
        _ => return None,
    }
    array_from_shape_vec(&shape, out)
}

fn try_mixed_mul_native_row_broadcast_f32(
    lhs: &Tensor,
    rhs: &Tensor,
    op: BinaryOp,
) -> Option<ArrayD<f32>> {
    if !matches!(op, BinaryOp::Mul)
        || lhs.device() != Device::Cpu
        || rhs.device() != Device::Cpu
        || lhs.dtype() == rhs.dtype()
    {
        return None;
    }
    let lhs_shape = lhs.shape_vec();
    let rhs_shape = rhs.shape_vec();
    let side = row_broadcast_side(&lhs_shape, &rhs_shape)?;
    let out_shape = match side {
        RowBroadcastSide::RhsVector => lhs_shape,
        RowBroadcastSide::LhsVector => rhs_shape,
    };
    let row_len = *out_shape.last()?;
    if row_len == 0 {
        return array_from_shape_vec(&out_shape, Vec::new());
    }
    let rows = out_shape.iter().product::<usize>() / row_len;
    let mut out = vec![0.0f32; rows * row_len];

    match (lhs.native_storage_owned(), rhs.native_storage_owned()) {
        (TensorStorageOwned::F32(f32_data), TensorStorageOwned::F16(lowp_data)) => {
            let f32_slice = f32_data.as_slice_memory_order()?;
            let lowp_slice = lowp_data.as_slice_memory_order()?;
            apply_mixed_row_broadcast_f16(f32_slice, lowp_slice, side, rows, row_len, &mut out);
        }
        (TensorStorageOwned::F16(lowp_data), TensorStorageOwned::F32(f32_data)) => {
            let f32_slice = f32_data.as_slice_memory_order()?;
            let lowp_slice = lowp_data.as_slice_memory_order()?;
            apply_mixed_row_broadcast_f16(
                f32_slice,
                lowp_slice,
                side.reverse(),
                rows,
                row_len,
                &mut out,
            );
        }
        (TensorStorageOwned::F32(f32_data), TensorStorageOwned::BF16(lowp_data)) => {
            let f32_slice = f32_data.as_slice_memory_order()?;
            let lowp_slice = lowp_data.as_slice_memory_order()?;
            apply_mixed_row_broadcast_bf16(f32_slice, lowp_slice, side, rows, row_len, &mut out);
        }
        (TensorStorageOwned::BF16(lowp_data), TensorStorageOwned::F32(f32_data)) => {
            let f32_slice = f32_data.as_slice_memory_order()?;
            let lowp_slice = lowp_data.as_slice_memory_order()?;
            apply_mixed_row_broadcast_bf16(
                f32_slice,
                lowp_slice,
                side.reverse(),
                rows,
                row_len,
                &mut out,
            );
        }
        (TensorStorageOwned::F32(f32_data), TensorStorageOwned::I8(lowp_data, scale)) => {
            let f32_slice = f32_data.as_slice_memory_order()?;
            let lowp_slice = lowp_data.as_slice_memory_order()?;
            apply_mixed_row_broadcast_i8(
                f32_slice, lowp_slice, scale, side, rows, row_len, &mut out,
            );
        }
        (TensorStorageOwned::I8(lowp_data, scale), TensorStorageOwned::F32(f32_data)) => {
            let f32_slice = f32_data.as_slice_memory_order()?;
            let lowp_slice = lowp_data.as_slice_memory_order()?;
            apply_mixed_row_broadcast_i8(
                f32_slice,
                lowp_slice,
                scale,
                side.reverse(),
                rows,
                row_len,
                &mut out,
            );
        }
        _ => return None,
    }

    array_from_shape_vec(&out_shape, out)
}

struct MixedRowBroadcastArgs<'a, T> {
    f32_slice: &'a [f32],
    lowp_slice: &'a [T],
    lowp_on_lhs: bool,
    side: RowBroadcastSide,
    rows: usize,
    row_len: usize,
}

fn apply_mixed_add_sub_row_broadcast_f16(
    args: MixedRowBroadcastArgs<'_, f16>,
    out: &mut [f32],
    op: BinaryOp,
) {
    for row in 0..args.rows {
        let start = row * args.row_len;
        match (args.side, args.lowp_on_lhs) {
            (RowBroadcastSide::RhsVector, true) => apply_mixed_add_sub_f16(
                args.f32_slice,
                &args.lowp_slice[start..start + args.row_len],
                true,
                op,
                &mut out[start..start + args.row_len],
            ),
            (RowBroadcastSide::RhsVector, false) => apply_mixed_add_sub_f16(
                &args.f32_slice[start..start + args.row_len],
                args.lowp_slice,
                false,
                op,
                &mut out[start..start + args.row_len],
            ),
            (RowBroadcastSide::LhsVector, true) => apply_mixed_add_sub_f16(
                &args.f32_slice[start..start + args.row_len],
                args.lowp_slice,
                true,
                op,
                &mut out[start..start + args.row_len],
            ),
            (RowBroadcastSide::LhsVector, false) => apply_mixed_add_sub_f16(
                args.f32_slice,
                &args.lowp_slice[start..start + args.row_len],
                false,
                op,
                &mut out[start..start + args.row_len],
            ),
        }
    }
}

fn apply_mixed_add_sub_row_broadcast_bf16(
    args: MixedRowBroadcastArgs<'_, bf16>,
    out: &mut [f32],
    op: BinaryOp,
) {
    for row in 0..args.rows {
        let start = row * args.row_len;
        match (args.side, args.lowp_on_lhs) {
            (RowBroadcastSide::RhsVector, true) => apply_mixed_add_sub_bf16(
                args.f32_slice,
                &args.lowp_slice[start..start + args.row_len],
                true,
                op,
                &mut out[start..start + args.row_len],
            ),
            (RowBroadcastSide::RhsVector, false) => apply_mixed_add_sub_bf16(
                &args.f32_slice[start..start + args.row_len],
                args.lowp_slice,
                false,
                op,
                &mut out[start..start + args.row_len],
            ),
            (RowBroadcastSide::LhsVector, true) => apply_mixed_add_sub_bf16(
                &args.f32_slice[start..start + args.row_len],
                args.lowp_slice,
                true,
                op,
                &mut out[start..start + args.row_len],
            ),
            (RowBroadcastSide::LhsVector, false) => apply_mixed_add_sub_bf16(
                args.f32_slice,
                &args.lowp_slice[start..start + args.row_len],
                false,
                op,
                &mut out[start..start + args.row_len],
            ),
        }
    }
}

fn apply_mixed_add_sub_row_broadcast_i8(
    args: MixedRowBroadcastArgs<'_, i8>,
    scale: f32,
    out: &mut [f32],
    op: BinaryOp,
) {
    for row in 0..args.rows {
        let start = row * args.row_len;
        match (args.side, args.lowp_on_lhs) {
            (RowBroadcastSide::RhsVector, true) => apply_mixed_add_sub_i8(
                args.f32_slice,
                &args.lowp_slice[start..start + args.row_len],
                scale,
                true,
                op,
                &mut out[start..start + args.row_len],
            ),
            (RowBroadcastSide::RhsVector, false) => apply_mixed_add_sub_i8(
                &args.f32_slice[start..start + args.row_len],
                args.lowp_slice,
                scale,
                false,
                op,
                &mut out[start..start + args.row_len],
            ),
            (RowBroadcastSide::LhsVector, true) => apply_mixed_add_sub_i8(
                &args.f32_slice[start..start + args.row_len],
                args.lowp_slice,
                scale,
                true,
                op,
                &mut out[start..start + args.row_len],
            ),
            (RowBroadcastSide::LhsVector, false) => apply_mixed_add_sub_i8(
                args.f32_slice,
                &args.lowp_slice[start..start + args.row_len],
                scale,
                false,
                op,
                &mut out[start..start + args.row_len],
            ),
        }
    }
}

fn try_mixed_add_sub_native_row_broadcast_f32(
    lhs: &Tensor,
    rhs: &Tensor,
    op: BinaryOp,
) -> Option<ArrayD<f32>> {
    if !matches!(op, BinaryOp::Add | BinaryOp::Sub)
        || lhs.device() != Device::Cpu
        || rhs.device() != Device::Cpu
        || lhs.dtype() == rhs.dtype()
    {
        return None;
    }
    let lhs_shape = lhs.shape_vec();
    let rhs_shape = rhs.shape_vec();
    let side = row_broadcast_side(&lhs_shape, &rhs_shape)?;
    let out_shape = match side {
        RowBroadcastSide::RhsVector => lhs_shape,
        RowBroadcastSide::LhsVector => rhs_shape,
    };
    let row_len = *out_shape.last()?;
    if row_len == 0 {
        return array_from_shape_vec(&out_shape, Vec::new());
    }
    let rows = out_shape.iter().product::<usize>() / row_len;
    let mut out = vec![0.0f32; rows * row_len];

    match (lhs.native_storage_owned(), rhs.native_storage_owned()) {
        (TensorStorageOwned::F32(f32_data), TensorStorageOwned::F16(lowp_data)) => {
            apply_mixed_add_sub_row_broadcast_f16(
                MixedRowBroadcastArgs {
                    f32_slice: f32_data.as_slice_memory_order()?,
                    lowp_slice: lowp_data.as_slice_memory_order()?,
                    lowp_on_lhs: false,
                    side,
                    rows,
                    row_len,
                },
                &mut out,
                op,
            );
        }
        (TensorStorageOwned::F16(lowp_data), TensorStorageOwned::F32(f32_data)) => {
            apply_mixed_add_sub_row_broadcast_f16(
                MixedRowBroadcastArgs {
                    f32_slice: f32_data.as_slice_memory_order()?,
                    lowp_slice: lowp_data.as_slice_memory_order()?,
                    lowp_on_lhs: true,
                    side,
                    rows,
                    row_len,
                },
                &mut out,
                op,
            );
        }
        (TensorStorageOwned::F32(f32_data), TensorStorageOwned::BF16(lowp_data)) => {
            apply_mixed_add_sub_row_broadcast_bf16(
                MixedRowBroadcastArgs {
                    f32_slice: f32_data.as_slice_memory_order()?,
                    lowp_slice: lowp_data.as_slice_memory_order()?,
                    lowp_on_lhs: false,
                    side,
                    rows,
                    row_len,
                },
                &mut out,
                op,
            );
        }
        (TensorStorageOwned::BF16(lowp_data), TensorStorageOwned::F32(f32_data)) => {
            apply_mixed_add_sub_row_broadcast_bf16(
                MixedRowBroadcastArgs {
                    f32_slice: f32_data.as_slice_memory_order()?,
                    lowp_slice: lowp_data.as_slice_memory_order()?,
                    lowp_on_lhs: true,
                    side,
                    rows,
                    row_len,
                },
                &mut out,
                op,
            );
        }
        (TensorStorageOwned::F32(f32_data), TensorStorageOwned::I8(lowp_data, scale)) => {
            apply_mixed_add_sub_row_broadcast_i8(
                MixedRowBroadcastArgs {
                    f32_slice: f32_data.as_slice_memory_order()?,
                    lowp_slice: lowp_data.as_slice_memory_order()?,
                    lowp_on_lhs: false,
                    side,
                    rows,
                    row_len,
                },
                scale,
                &mut out,
                op,
            );
        }
        (TensorStorageOwned::I8(lowp_data, scale), TensorStorageOwned::F32(f32_data)) => {
            apply_mixed_add_sub_row_broadcast_i8(
                MixedRowBroadcastArgs {
                    f32_slice: f32_data.as_slice_memory_order()?,
                    lowp_slice: lowp_data.as_slice_memory_order()?,
                    lowp_on_lhs: true,
                    side,
                    rows,
                    row_len,
                },
                scale,
                &mut out,
                op,
            );
        }
        _ => return None,
    }

    array_from_shape_vec(&out_shape, out)
}

fn mul_f32_slice_by_scalar_to_f32(input: &[f32], scalar: f32, out: &mut [f32]) {
    for (dst, &x) in out.iter_mut().zip(input) {
        *dst = x * scalar;
    }
}

fn mul_f16_slice_by_scalar_to_f32(input: &[f16], scalar: f32, out: &mut [f32]) {
    if !mul_f16_scalar_to_f32_arch(input, scalar, out) {
        for (dst, &x) in out.iter_mut().zip(input) {
            *dst = x.to_f32() * scalar;
        }
    }
}

fn mul_bf16_slice_by_scalar_to_f32(input: &[bf16], scalar: f32, out: &mut [f32]) {
    if !mul_bf16_scalar_to_f32_arch(input, scalar, out) {
        for (dst, &x) in out.iter_mut().zip(input) {
            *dst = x.to_f32() * scalar;
        }
    }
}

fn mul_i8_slice_by_scalar_to_f32(input: &[i8], scale: f32, scalar: f32, out: &mut [f32]) {
    if !mul_i8_scalar_to_f32_arch(input, scale, scalar, out) {
        for (dst, &x) in out.iter_mut().zip(input) {
            *dst = (x as f32) * scale * scalar;
        }
    }
}

fn apply_mixed_add_sub_f32_with_lowp_scalar(
    f32_slice: &[f32],
    lowp_scalar: f32,
    lowp_on_lhs: bool,
    op: BinaryOp,
    out: &mut [f32],
) {
    for (dst, &f32_value) in out.iter_mut().zip(f32_slice) {
        *dst = mixed_add_sub_value(f32_value, lowp_scalar, lowp_on_lhs, op);
    }
}

fn apply_mixed_add_sub_f16_with_f32_scalar(
    lowp_slice: &[f16],
    f32_scalar: f32,
    lowp_on_lhs: bool,
    op: BinaryOp,
    out: &mut [f32],
) {
    let used_arch = match op {
        BinaryOp::Add => add_f16_scalar_to_f32_arch(lowp_slice, f32_scalar, out),
        BinaryOp::Sub => sub_f16_scalar_to_f32_arch(lowp_slice, f32_scalar, lowp_on_lhs, out),
        BinaryOp::Mul => false,
    };
    if used_arch {
        return;
    }
    for (dst, &lowp_value) in out.iter_mut().zip(lowp_slice) {
        *dst = mixed_add_sub_value(f32_scalar, lowp_value.to_f32(), lowp_on_lhs, op);
    }
}

fn apply_mixed_add_sub_bf16_with_f32_scalar(
    lowp_slice: &[bf16],
    f32_scalar: f32,
    lowp_on_lhs: bool,
    op: BinaryOp,
    out: &mut [f32],
) {
    let used_arch = match op {
        BinaryOp::Add => add_bf16_scalar_to_f32_arch(lowp_slice, f32_scalar, out),
        BinaryOp::Sub => sub_bf16_scalar_to_f32_arch(lowp_slice, f32_scalar, lowp_on_lhs, out),
        BinaryOp::Mul => false,
    };
    if used_arch {
        return;
    }
    for (dst, &lowp_value) in out.iter_mut().zip(lowp_slice) {
        *dst = mixed_add_sub_value(f32_scalar, lowp_value.to_f32(), lowp_on_lhs, op);
    }
}

fn apply_mixed_add_sub_i8_with_f32_scalar(
    lowp_slice: &[i8],
    scale: f32,
    f32_scalar: f32,
    lowp_on_lhs: bool,
    op: BinaryOp,
    out: &mut [f32],
) {
    let used_arch = match op {
        BinaryOp::Add => add_i8_scalar_to_f32_arch(lowp_slice, scale, f32_scalar, out),
        BinaryOp::Sub => sub_i8_scalar_to_f32_arch(lowp_slice, scale, f32_scalar, lowp_on_lhs, out),
        BinaryOp::Mul => false,
    };
    if used_arch {
        return;
    }
    for (dst, &lowp_value) in out.iter_mut().zip(lowp_slice) {
        *dst = mixed_add_sub_value(f32_scalar, (lowp_value as f32) * scale, lowp_on_lhs, op);
    }
}

fn try_mixed_add_sub_native_row_scalar_f32(
    lhs: &Tensor,
    rhs: &Tensor,
    op: BinaryOp,
) -> Option<ArrayD<f32>> {
    if !matches!(op, BinaryOp::Add | BinaryOp::Sub)
        || lhs.device() != Device::Cpu
        || rhs.device() != Device::Cpu
        || lhs.dtype() == rhs.dtype()
    {
        return None;
    }
    let lhs_shape = lhs.shape_vec();
    let rhs_shape = rhs.shape_vec();
    let (rows, last_dim, scalar_on_rhs) = row_scalar_broadcast_side(&lhs_shape, &rhs_shape)?;
    let out_shape = if scalar_on_rhs { lhs_shape } else { rhs_shape };
    let mut out = vec![0.0f32; rows * last_dim];

    match (lhs.native_storage_owned(), rhs.native_storage_owned()) {
        (TensorStorageOwned::F32(f32_data), TensorStorageOwned::F16(lowp_data)) => {
            let f32_slice = f32_data.as_slice_memory_order()?;
            let lowp_slice = lowp_data.as_slice_memory_order()?;
            for row in 0..rows {
                let start = row * last_dim;
                if scalar_on_rhs {
                    apply_mixed_add_sub_f32_with_lowp_scalar(
                        &f32_slice[start..start + last_dim],
                        lowp_slice[row].to_f32(),
                        false,
                        op,
                        &mut out[start..start + last_dim],
                    );
                } else {
                    apply_mixed_add_sub_f16_with_f32_scalar(
                        &lowp_slice[start..start + last_dim],
                        f32_slice[row],
                        false,
                        op,
                        &mut out[start..start + last_dim],
                    );
                }
            }
        }
        (TensorStorageOwned::F16(lowp_data), TensorStorageOwned::F32(f32_data)) => {
            let f32_slice = f32_data.as_slice_memory_order()?;
            let lowp_slice = lowp_data.as_slice_memory_order()?;
            for row in 0..rows {
                let start = row * last_dim;
                if scalar_on_rhs {
                    apply_mixed_add_sub_f16_with_f32_scalar(
                        &lowp_slice[start..start + last_dim],
                        f32_slice[row],
                        true,
                        op,
                        &mut out[start..start + last_dim],
                    );
                } else {
                    apply_mixed_add_sub_f32_with_lowp_scalar(
                        &f32_slice[start..start + last_dim],
                        lowp_slice[row].to_f32(),
                        true,
                        op,
                        &mut out[start..start + last_dim],
                    );
                }
            }
        }
        (TensorStorageOwned::F32(f32_data), TensorStorageOwned::BF16(lowp_data)) => {
            let f32_slice = f32_data.as_slice_memory_order()?;
            let lowp_slice = lowp_data.as_slice_memory_order()?;
            for row in 0..rows {
                let start = row * last_dim;
                if scalar_on_rhs {
                    apply_mixed_add_sub_f32_with_lowp_scalar(
                        &f32_slice[start..start + last_dim],
                        lowp_slice[row].to_f32(),
                        false,
                        op,
                        &mut out[start..start + last_dim],
                    );
                } else {
                    apply_mixed_add_sub_bf16_with_f32_scalar(
                        &lowp_slice[start..start + last_dim],
                        f32_slice[row],
                        false,
                        op,
                        &mut out[start..start + last_dim],
                    );
                }
            }
        }
        (TensorStorageOwned::BF16(lowp_data), TensorStorageOwned::F32(f32_data)) => {
            let f32_slice = f32_data.as_slice_memory_order()?;
            let lowp_slice = lowp_data.as_slice_memory_order()?;
            for row in 0..rows {
                let start = row * last_dim;
                if scalar_on_rhs {
                    apply_mixed_add_sub_bf16_with_f32_scalar(
                        &lowp_slice[start..start + last_dim],
                        f32_slice[row],
                        true,
                        op,
                        &mut out[start..start + last_dim],
                    );
                } else {
                    apply_mixed_add_sub_f32_with_lowp_scalar(
                        &f32_slice[start..start + last_dim],
                        lowp_slice[row].to_f32(),
                        true,
                        op,
                        &mut out[start..start + last_dim],
                    );
                }
            }
        }
        (TensorStorageOwned::F32(f32_data), TensorStorageOwned::I8(lowp_data, scale)) => {
            let f32_slice = f32_data.as_slice_memory_order()?;
            let lowp_slice = lowp_data.as_slice_memory_order()?;
            for row in 0..rows {
                let start = row * last_dim;
                if scalar_on_rhs {
                    apply_mixed_add_sub_f32_with_lowp_scalar(
                        &f32_slice[start..start + last_dim],
                        (lowp_slice[row] as f32) * scale,
                        false,
                        op,
                        &mut out[start..start + last_dim],
                    );
                } else {
                    apply_mixed_add_sub_i8_with_f32_scalar(
                        &lowp_slice[start..start + last_dim],
                        scale,
                        f32_slice[row],
                        false,
                        op,
                        &mut out[start..start + last_dim],
                    );
                }
            }
        }
        (TensorStorageOwned::I8(lowp_data, scale), TensorStorageOwned::F32(f32_data)) => {
            let f32_slice = f32_data.as_slice_memory_order()?;
            let lowp_slice = lowp_data.as_slice_memory_order()?;
            for row in 0..rows {
                let start = row * last_dim;
                if scalar_on_rhs {
                    apply_mixed_add_sub_i8_with_f32_scalar(
                        &lowp_slice[start..start + last_dim],
                        scale,
                        f32_slice[row],
                        true,
                        op,
                        &mut out[start..start + last_dim],
                    );
                } else {
                    apply_mixed_add_sub_f32_with_lowp_scalar(
                        &f32_slice[start..start + last_dim],
                        (lowp_slice[row] as f32) * scale,
                        true,
                        op,
                        &mut out[start..start + last_dim],
                    );
                }
            }
        }
        _ => return None,
    }

    array_from_shape_vec(&out_shape, out)
}

fn try_mixed_add_sub_native_scalar_broadcast_f32(
    lhs: &Tensor,
    rhs: &Tensor,
    op: BinaryOp,
) -> Option<ArrayD<f32>> {
    if !matches!(op, BinaryOp::Add | BinaryOp::Sub)
        || lhs.device() != Device::Cpu
        || rhs.device() != Device::Cpu
        || lhs.dtype() == rhs.dtype()
    {
        return None;
    }
    let lhs_shape = lhs.shape_vec();
    let rhs_shape = rhs.shape_vec();
    let out_shape = broadcast_shape(&lhs_shape, &rhs_shape)?;
    let out_len = out_shape.iter().product::<usize>();
    if out_len == 0
        || !((lhs.len() == 1 && rhs.len() == out_len) || (rhs.len() == 1 && lhs.len() == out_len))
    {
        return None;
    }
    let mut out = vec![0.0f32; out_len];

    match (lhs.native_storage_owned(), rhs.native_storage_owned()) {
        (TensorStorageOwned::F32(f32_data), TensorStorageOwned::F16(lowp_data)) => {
            let f32_slice = f32_data.as_slice_memory_order()?;
            let lowp_slice = lowp_data.as_slice_memory_order()?;
            if f32_slice.len() == 1 {
                apply_mixed_add_sub_f16_with_f32_scalar(
                    lowp_slice,
                    f32_slice[0],
                    false,
                    op,
                    &mut out,
                );
            } else {
                apply_mixed_add_sub_f32_with_lowp_scalar(
                    f32_slice,
                    lowp_slice[0].to_f32(),
                    false,
                    op,
                    &mut out,
                );
            }
        }
        (TensorStorageOwned::F16(lowp_data), TensorStorageOwned::F32(f32_data)) => {
            let f32_slice = f32_data.as_slice_memory_order()?;
            let lowp_slice = lowp_data.as_slice_memory_order()?;
            if lowp_slice.len() == 1 {
                apply_mixed_add_sub_f32_with_lowp_scalar(
                    f32_slice,
                    lowp_slice[0].to_f32(),
                    true,
                    op,
                    &mut out,
                );
            } else {
                apply_mixed_add_sub_f16_with_f32_scalar(
                    lowp_slice,
                    f32_slice[0],
                    true,
                    op,
                    &mut out,
                );
            }
        }
        (TensorStorageOwned::F32(f32_data), TensorStorageOwned::BF16(lowp_data)) => {
            let f32_slice = f32_data.as_slice_memory_order()?;
            let lowp_slice = lowp_data.as_slice_memory_order()?;
            if f32_slice.len() == 1 {
                apply_mixed_add_sub_bf16_with_f32_scalar(
                    lowp_slice,
                    f32_slice[0],
                    false,
                    op,
                    &mut out,
                );
            } else {
                apply_mixed_add_sub_f32_with_lowp_scalar(
                    f32_slice,
                    lowp_slice[0].to_f32(),
                    false,
                    op,
                    &mut out,
                );
            }
        }
        (TensorStorageOwned::BF16(lowp_data), TensorStorageOwned::F32(f32_data)) => {
            let f32_slice = f32_data.as_slice_memory_order()?;
            let lowp_slice = lowp_data.as_slice_memory_order()?;
            if lowp_slice.len() == 1 {
                apply_mixed_add_sub_f32_with_lowp_scalar(
                    f32_slice,
                    lowp_slice[0].to_f32(),
                    true,
                    op,
                    &mut out,
                );
            } else {
                apply_mixed_add_sub_bf16_with_f32_scalar(
                    lowp_slice,
                    f32_slice[0],
                    true,
                    op,
                    &mut out,
                );
            }
        }
        (TensorStorageOwned::F32(f32_data), TensorStorageOwned::I8(lowp_data, scale)) => {
            let f32_slice = f32_data.as_slice_memory_order()?;
            let lowp_slice = lowp_data.as_slice_memory_order()?;
            if f32_slice.len() == 1 {
                apply_mixed_add_sub_i8_with_f32_scalar(
                    lowp_slice,
                    scale,
                    f32_slice[0],
                    false,
                    op,
                    &mut out,
                );
            } else {
                apply_mixed_add_sub_f32_with_lowp_scalar(
                    f32_slice,
                    (lowp_slice[0] as f32) * scale,
                    false,
                    op,
                    &mut out,
                );
            }
        }
        (TensorStorageOwned::I8(lowp_data, scale), TensorStorageOwned::F32(f32_data)) => {
            let f32_slice = f32_data.as_slice_memory_order()?;
            let lowp_slice = lowp_data.as_slice_memory_order()?;
            if lowp_slice.len() == 1 {
                apply_mixed_add_sub_f32_with_lowp_scalar(
                    f32_slice,
                    (lowp_slice[0] as f32) * scale,
                    true,
                    op,
                    &mut out,
                );
            } else {
                apply_mixed_add_sub_i8_with_f32_scalar(
                    lowp_slice,
                    scale,
                    f32_slice[0],
                    true,
                    op,
                    &mut out,
                );
            }
        }
        _ => return None,
    }

    array_from_shape_vec(&out_shape, out)
}

fn try_mixed_mul_native_row_scalar_f32(
    lhs: &Tensor,
    rhs: &Tensor,
    op: BinaryOp,
) -> Option<ArrayD<f32>> {
    if !matches!(op, BinaryOp::Mul)
        || lhs.device() != Device::Cpu
        || rhs.device() != Device::Cpu
        || lhs.dtype() == rhs.dtype()
    {
        return None;
    }
    let lhs_shape = lhs.shape_vec();
    let rhs_shape = rhs.shape_vec();
    let (rows, last_dim, scalar_on_rhs) = row_scalar_broadcast_side(&lhs_shape, &rhs_shape)?;
    let out_shape = if scalar_on_rhs { lhs_shape } else { rhs_shape };
    let mut out = vec![0.0f32; rows * last_dim];

    match (lhs.native_storage_owned(), rhs.native_storage_owned()) {
        (TensorStorageOwned::F32(f32_data), TensorStorageOwned::F16(lowp_data)) => {
            let f32_slice = f32_data.as_slice_memory_order()?;
            let lowp_slice = lowp_data.as_slice_memory_order()?;
            for row in 0..rows {
                let start = row * last_dim;
                if scalar_on_rhs {
                    mul_f32_slice_by_scalar_to_f32(
                        &f32_slice[start..start + last_dim],
                        lowp_slice[row].to_f32(),
                        &mut out[start..start + last_dim],
                    );
                } else {
                    mul_f16_slice_by_scalar_to_f32(
                        &lowp_slice[start..start + last_dim],
                        f32_slice[row],
                        &mut out[start..start + last_dim],
                    );
                }
            }
        }
        (TensorStorageOwned::F16(lowp_data), TensorStorageOwned::F32(f32_data)) => {
            let f32_slice = f32_data.as_slice_memory_order()?;
            let lowp_slice = lowp_data.as_slice_memory_order()?;
            for row in 0..rows {
                let start = row * last_dim;
                if scalar_on_rhs {
                    mul_f16_slice_by_scalar_to_f32(
                        &lowp_slice[start..start + last_dim],
                        f32_slice[row],
                        &mut out[start..start + last_dim],
                    );
                } else {
                    mul_f32_slice_by_scalar_to_f32(
                        &f32_slice[start..start + last_dim],
                        lowp_slice[row].to_f32(),
                        &mut out[start..start + last_dim],
                    );
                }
            }
        }
        (TensorStorageOwned::F32(f32_data), TensorStorageOwned::BF16(lowp_data)) => {
            let f32_slice = f32_data.as_slice_memory_order()?;
            let lowp_slice = lowp_data.as_slice_memory_order()?;
            for row in 0..rows {
                let start = row * last_dim;
                if scalar_on_rhs {
                    mul_f32_slice_by_scalar_to_f32(
                        &f32_slice[start..start + last_dim],
                        lowp_slice[row].to_f32(),
                        &mut out[start..start + last_dim],
                    );
                } else {
                    mul_bf16_slice_by_scalar_to_f32(
                        &lowp_slice[start..start + last_dim],
                        f32_slice[row],
                        &mut out[start..start + last_dim],
                    );
                }
            }
        }
        (TensorStorageOwned::BF16(lowp_data), TensorStorageOwned::F32(f32_data)) => {
            let f32_slice = f32_data.as_slice_memory_order()?;
            let lowp_slice = lowp_data.as_slice_memory_order()?;
            for row in 0..rows {
                let start = row * last_dim;
                if scalar_on_rhs {
                    mul_bf16_slice_by_scalar_to_f32(
                        &lowp_slice[start..start + last_dim],
                        f32_slice[row],
                        &mut out[start..start + last_dim],
                    );
                } else {
                    mul_f32_slice_by_scalar_to_f32(
                        &f32_slice[start..start + last_dim],
                        lowp_slice[row].to_f32(),
                        &mut out[start..start + last_dim],
                    );
                }
            }
        }
        (TensorStorageOwned::F32(f32_data), TensorStorageOwned::I8(lowp_data, scale)) => {
            let f32_slice = f32_data.as_slice_memory_order()?;
            let lowp_slice = lowp_data.as_slice_memory_order()?;
            for row in 0..rows {
                let start = row * last_dim;
                if scalar_on_rhs {
                    mul_f32_slice_by_scalar_to_f32(
                        &f32_slice[start..start + last_dim],
                        (lowp_slice[row] as f32) * scale,
                        &mut out[start..start + last_dim],
                    );
                } else {
                    mul_i8_slice_by_scalar_to_f32(
                        &lowp_slice[start..start + last_dim],
                        scale,
                        f32_slice[row],
                        &mut out[start..start + last_dim],
                    );
                }
            }
        }
        (TensorStorageOwned::I8(lowp_data, scale), TensorStorageOwned::F32(f32_data)) => {
            let f32_slice = f32_data.as_slice_memory_order()?;
            let lowp_slice = lowp_data.as_slice_memory_order()?;
            for row in 0..rows {
                let start = row * last_dim;
                if scalar_on_rhs {
                    mul_i8_slice_by_scalar_to_f32(
                        &lowp_slice[start..start + last_dim],
                        scale,
                        f32_slice[row],
                        &mut out[start..start + last_dim],
                    );
                } else {
                    mul_f32_slice_by_scalar_to_f32(
                        &f32_slice[start..start + last_dim],
                        (lowp_slice[row] as f32) * scale,
                        &mut out[start..start + last_dim],
                    );
                }
            }
        }
        _ => return None,
    }

    array_from_shape_vec(&out_shape, out)
}

fn try_mixed_mul_native_scalar_broadcast_f32(
    lhs: &Tensor,
    rhs: &Tensor,
    op: BinaryOp,
) -> Option<ArrayD<f32>> {
    if !matches!(op, BinaryOp::Mul)
        || lhs.device() != Device::Cpu
        || rhs.device() != Device::Cpu
        || lhs.dtype() == rhs.dtype()
    {
        return None;
    }
    let lhs_shape = lhs.shape_vec();
    let rhs_shape = rhs.shape_vec();
    let out_shape = broadcast_shape(&lhs_shape, &rhs_shape)?;
    let out_len = out_shape.iter().product::<usize>();
    if out_len == 0
        || !((lhs.len() == 1 && rhs.len() == out_len) || (rhs.len() == 1 && lhs.len() == out_len))
    {
        return None;
    }
    let mut out = vec![0.0f32; out_len];

    match (lhs.native_storage_owned(), rhs.native_storage_owned()) {
        (TensorStorageOwned::F32(f32_data), TensorStorageOwned::F16(lowp_data)) => {
            let f32_slice = f32_data.as_slice_memory_order()?;
            let lowp_slice = lowp_data.as_slice_memory_order()?;
            if f32_slice.len() == 1 {
                mul_f16_slice_by_scalar_to_f32(lowp_slice, f32_slice[0], &mut out);
            } else {
                mul_f32_slice_by_scalar_to_f32(f32_slice, lowp_slice[0].to_f32(), &mut out);
            }
        }
        (TensorStorageOwned::F16(lowp_data), TensorStorageOwned::F32(f32_data)) => {
            let f32_slice = f32_data.as_slice_memory_order()?;
            let lowp_slice = lowp_data.as_slice_memory_order()?;
            if lowp_slice.len() == 1 {
                mul_f32_slice_by_scalar_to_f32(f32_slice, lowp_slice[0].to_f32(), &mut out);
            } else {
                mul_f16_slice_by_scalar_to_f32(lowp_slice, f32_slice[0], &mut out);
            }
        }
        (TensorStorageOwned::F32(f32_data), TensorStorageOwned::BF16(lowp_data)) => {
            let f32_slice = f32_data.as_slice_memory_order()?;
            let lowp_slice = lowp_data.as_slice_memory_order()?;
            if f32_slice.len() == 1 {
                mul_bf16_slice_by_scalar_to_f32(lowp_slice, f32_slice[0], &mut out);
            } else {
                mul_f32_slice_by_scalar_to_f32(f32_slice, lowp_slice[0].to_f32(), &mut out);
            }
        }
        (TensorStorageOwned::BF16(lowp_data), TensorStorageOwned::F32(f32_data)) => {
            let f32_slice = f32_data.as_slice_memory_order()?;
            let lowp_slice = lowp_data.as_slice_memory_order()?;
            if lowp_slice.len() == 1 {
                mul_f32_slice_by_scalar_to_f32(f32_slice, lowp_slice[0].to_f32(), &mut out);
            } else {
                mul_bf16_slice_by_scalar_to_f32(lowp_slice, f32_slice[0], &mut out);
            }
        }
        (TensorStorageOwned::F32(f32_data), TensorStorageOwned::I8(lowp_data, scale)) => {
            let f32_slice = f32_data.as_slice_memory_order()?;
            let lowp_slice = lowp_data.as_slice_memory_order()?;
            if f32_slice.len() == 1 {
                mul_i8_slice_by_scalar_to_f32(lowp_slice, scale, f32_slice[0], &mut out);
            } else {
                mul_f32_slice_by_scalar_to_f32(f32_slice, (lowp_slice[0] as f32) * scale, &mut out);
            }
        }
        (TensorStorageOwned::I8(lowp_data, scale), TensorStorageOwned::F32(f32_data)) => {
            let f32_slice = f32_data.as_slice_memory_order()?;
            let lowp_slice = lowp_data.as_slice_memory_order()?;
            if lowp_slice.len() == 1 {
                mul_f32_slice_by_scalar_to_f32(f32_slice, (lowp_slice[0] as f32) * scale, &mut out);
            } else {
                mul_i8_slice_by_scalar_to_f32(lowp_slice, scale, f32_slice[0], &mut out);
            }
        }
        _ => return None,
    }

    array_from_shape_vec(&out_shape, out)
}

fn try_cuda_binary_native_same_shape_f32_buffer(
    lhs: &Tensor,
    rhs: &Tensor,
    op: BinaryOp,
) -> Option<cuda::CudaBuffer> {
    if lhs.device() != Device::Cuda
        || rhs.device() != Device::Cuda
        || lhs.dtype() != rhs.dtype()
        || lhs.shape_vec() != rhs.shape_vec()
        || lhs.is_empty()
    {
        return None;
    }

    let cuda_op = cuda_binary_op(op);
    if let (Some((lhs_dtype, lhs_buffer, lhs_scale)), Some((rhs_dtype, rhs_buffer, rhs_scale))) = (
        lhs.cloned_cuda_native_lowp_buffer(),
        rhs.cloned_cuda_native_lowp_buffer(),
    ) && lhs_dtype == rhs_dtype
    {
        return match lhs_dtype {
            DType::F16 => cuda::binary_f16_buffer_no_host(&lhs_buffer, &rhs_buffer, cuda_op).ok(),
            DType::BF16 => cuda::binary_bf16_buffer_no_host(&lhs_buffer, &rhs_buffer, cuda_op).ok(),
            DType::I8 => cuda::binary_i8_buffer_no_host(
                &lhs_buffer,
                lhs_scale?,
                &rhs_buffer,
                rhs_scale?,
                cuda_op,
            )
            .ok(),
            DType::F32 => None,
        };
    }

    match (lhs.native_storage_owned(), rhs.native_storage_owned()) {
        (TensorStorageOwned::F16(lhs_data), TensorStorageOwned::F16(rhs_data)) => {
            let lhs_slice = lhs_data.as_slice_memory_order()?;
            let rhs_slice = rhs_data.as_slice_memory_order()?;
            let lhs_bits: Vec<u16> = lhs_slice.iter().map(|v| v.to_bits()).collect();
            let rhs_bits: Vec<u16> = rhs_slice.iter().map(|v| v.to_bits()).collect();
            cuda::binary_f16_host_no_host(&lhs_bits, &rhs_bits, cuda_op).ok()
        }
        (TensorStorageOwned::BF16(lhs_data), TensorStorageOwned::BF16(rhs_data)) => {
            let lhs_slice = lhs_data.as_slice_memory_order()?;
            let rhs_slice = rhs_data.as_slice_memory_order()?;
            let lhs_bits: Vec<u16> = lhs_slice.iter().map(|v| v.to_bits()).collect();
            let rhs_bits: Vec<u16> = rhs_slice.iter().map(|v| v.to_bits()).collect();
            cuda::binary_bf16_host_no_host(&lhs_bits, &rhs_bits, cuda_op).ok()
        }
        (
            TensorStorageOwned::I8(lhs_data, lhs_scale),
            TensorStorageOwned::I8(rhs_data, rhs_scale),
        ) => {
            let lhs_slice = lhs_data.as_slice_memory_order()?;
            let rhs_slice = rhs_data.as_slice_memory_order()?;
            cuda::binary_i8_host_no_host(lhs_slice, lhs_scale, rhs_slice, rhs_scale, cuda_op).ok()
        }
        _ => None,
    }
}

fn try_cuda_binary_same_dtype_lowp_typed_output(
    lhs: &Tensor,
    rhs: &Tensor,
    op: BinaryOp,
) -> Option<Tensor> {
    if lhs.device() != Device::Cuda
        || rhs.device() != Device::Cuda
        || lhs.dtype() != rhs.dtype()
        || lhs.shape_vec() != rhs.shape_vec()
        || lhs.is_empty()
        || !matches!(lhs.dtype(), DType::F16 | DType::BF16 | DType::I8)
    {
        return None;
    }
    let (lhs_dtype, lhs_buffer, lhs_scale) = lhs.cloned_cuda_native_lowp_buffer()?;
    let (rhs_dtype, rhs_buffer, rhs_scale) = rhs.cloned_cuda_native_lowp_buffer()?;
    if lhs_dtype != rhs_dtype || lhs_dtype != lhs.dtype() {
        return None;
    }
    if lhs_dtype == DType::I8 {
        let (buffer, out_scale) = cuda::binary_i8_typed_output_buffer_no_host(
            &lhs_buffer,
            lhs_scale?,
            &rhs_buffer,
            rhs_scale?,
            cuda_binary_op(op),
        )
        .ok()?;
        return Some(Tensor::from_cuda_native_lowp_buffer_no_host_with_dtype(
            &lhs.shape_vec(),
            buffer,
            Device::Cuda,
            DType::I8,
            Some(out_scale),
        ));
    }
    let buffer = cuda::binary_lowp_typed_output_buffer_no_host(
        &lhs_buffer,
        &rhs_buffer,
        lhs_dtype,
        cuda_binary_op(op),
    )
    .ok()?;
    Some(Tensor::from_cuda_native_lowp_buffer_no_host_with_dtype(
        &lhs.shape_vec(),
        buffer,
        Device::Cuda,
        lhs_dtype,
        None,
    ))
}

fn cuda_binary_storage(tensor: &Tensor) -> Option<(cuda::CudaBuffer, DType, Option<f32>)> {
    tensor
        .cloned_cuda_native_lowp_buffer()
        .map(|(dtype, buffer, scale)| (buffer, dtype, scale))
        .or_else(|| {
            if tensor.dtype() == DType::F32 && tensor.is_cuda() {
                tensor
                    .cloned_cuda_f32_buffer()
                    .map(|buffer| (buffer, DType::F32, None))
            } else {
                None
            }
        })
}

fn try_cuda_binary_typed_same_shape_f32_buffer(
    lhs: &Tensor,
    rhs: &Tensor,
    op: BinaryOp,
) -> Option<cuda::CudaBuffer> {
    if lhs.device() != Device::Cuda
        || rhs.device() != Device::Cuda
        || lhs.shape_vec() != rhs.shape_vec()
        || lhs.is_empty()
    {
        return None;
    }
    let (lhs_buffer, lhs_dtype, lhs_scale) = cuda_binary_storage(lhs)?;
    let (rhs_buffer, rhs_dtype, rhs_scale) = cuda_binary_storage(rhs)?;
    if lhs_dtype == rhs_dtype && lhs_dtype != DType::F32 {
        return None;
    }
    cuda::binary_typed_buffer_no_host(
        &lhs_buffer,
        lhs_dtype,
        lhs_scale,
        &rhs_buffer,
        rhs_dtype,
        rhs_scale,
        cuda_binary_op(op),
    )
    .ok()
}

fn try_cuda_binary_native_row_broadcast_f32_buffer(
    lhs: &Tensor,
    rhs: &Tensor,
    op: BinaryOp,
) -> Option<cuda::CudaBuffer> {
    if lhs.device() != Device::Cuda || rhs.device() != Device::Cuda || lhs.dtype() != rhs.dtype() {
        return None;
    }
    let lhs_shape = lhs.shape_vec();
    let rhs_shape = rhs.shape_vec();
    let side = row_broadcast_side(&lhs_shape, &rhs_shape)?;
    let out_shape = match side {
        RowBroadcastSide::RhsVector => lhs_shape,
        RowBroadcastSide::LhsVector => rhs_shape,
    };
    let last_dim = *out_shape.last()?;
    let out_len = out_shape.iter().product::<usize>();
    if last_dim == 0 || out_len == 0 {
        return None;
    }

    let cuda_op = cuda_binary_op(op);
    let vector_on_rhs = matches!(side, RowBroadcastSide::RhsVector);
    let (Some((lhs_dtype, lhs_buffer, lhs_scale)), Some((rhs_dtype, rhs_buffer, rhs_scale))) = (
        lhs.cloned_cuda_native_lowp_buffer(),
        rhs.cloned_cuda_native_lowp_buffer(),
    ) else {
        return None;
    };
    if lhs_dtype != rhs_dtype {
        return None;
    }

    match lhs_dtype {
        DType::F16 => cuda::binary_f16_lastdim_buffer_no_host(
            &lhs_buffer,
            &rhs_buffer,
            out_len,
            last_dim,
            vector_on_rhs,
            cuda_op,
        )
        .ok(),
        DType::BF16 => cuda::binary_bf16_lastdim_buffer_no_host(
            &lhs_buffer,
            &rhs_buffer,
            out_len,
            last_dim,
            vector_on_rhs,
            cuda_op,
        )
        .ok(),
        DType::I8 => cuda::binary_i8_lastdim_buffer_no_host(
            &lhs_buffer,
            lhs_scale?,
            &rhs_buffer,
            rhs_scale?,
            out_len,
            last_dim,
            vector_on_rhs,
            cuda_op,
        )
        .ok(),
        DType::F32 => None,
    }
}

fn try_cuda_binary_row_broadcast_lowp_typed_output(
    lhs: &Tensor,
    rhs: &Tensor,
    op: BinaryOp,
) -> Option<Tensor> {
    if lhs.device() != Device::Cuda || rhs.device() != Device::Cuda || lhs.dtype() != rhs.dtype() {
        return None;
    }
    let dtype = lhs.dtype();
    if !matches!(dtype, DType::F16 | DType::BF16 | DType::I8) {
        return None;
    }
    let lhs_shape = lhs.shape_vec();
    let rhs_shape = rhs.shape_vec();
    let side = row_broadcast_side(&lhs_shape, &rhs_shape)?;
    let out_shape = match side {
        RowBroadcastSide::RhsVector => lhs_shape,
        RowBroadcastSide::LhsVector => rhs_shape,
    };
    let last_dim = *out_shape.last()?;
    let out_len = out_shape.iter().product::<usize>();
    if last_dim == 0 || out_len == 0 {
        return None;
    }
    let vector_on_rhs = matches!(side, RowBroadcastSide::RhsVector);
    let (lhs_dtype, lhs_buffer, lhs_scale) = lhs.cloned_cuda_native_lowp_buffer()?;
    let (rhs_dtype, rhs_buffer, rhs_scale) = rhs.cloned_cuda_native_lowp_buffer()?;
    if lhs_dtype != dtype || rhs_dtype != dtype {
        return None;
    }
    if dtype == DType::I8 {
        let (buffer, out_scale) = cuda::binary_i8_typed_lastdim_output_buffer_no_host(
            &lhs_buffer,
            lhs_scale?,
            &rhs_buffer,
            rhs_scale?,
            out_len,
            last_dim,
            vector_on_rhs,
            cuda_binary_op(op),
        )
        .ok()?;
        return Some(Tensor::from_cuda_native_lowp_buffer_no_host_with_dtype(
            &out_shape,
            buffer,
            Device::Cuda,
            DType::I8,
            Some(out_scale),
        ));
    }
    let buffer = cuda::binary_lowp_typed_lastdim_output_buffer_no_host(
        &lhs_buffer,
        &rhs_buffer,
        out_len,
        last_dim,
        vector_on_rhs,
        dtype,
        cuda_binary_op(op),
    )
    .ok()?;
    Some(Tensor::from_cuda_native_lowp_buffer_no_host_with_dtype(
        &out_shape,
        buffer,
        Device::Cuda,
        dtype,
        None,
    ))
}

fn try_cuda_binary_row_scalar_lowp_typed_output(
    lhs: &Tensor,
    rhs: &Tensor,
    op: BinaryOp,
) -> Option<Tensor> {
    if lhs.device() != Device::Cuda || rhs.device() != Device::Cuda || lhs.dtype() != rhs.dtype() {
        return None;
    }
    let dtype = lhs.dtype();
    if !matches!(dtype, DType::F16 | DType::BF16 | DType::I8) {
        return None;
    }
    let lhs_shape = lhs.shape_vec();
    let rhs_shape = rhs.shape_vec();
    let (rows, last_dim, scalar_on_rhs) = row_scalar_broadcast_side(&lhs_shape, &rhs_shape)?;
    let out_shape = if scalar_on_rhs { lhs_shape } else { rhs_shape };
    let (lhs_dtype, lhs_buffer, lhs_scale) = lhs.cloned_cuda_native_lowp_buffer()?;
    let (rhs_dtype, rhs_buffer, rhs_scale) = rhs.cloned_cuda_native_lowp_buffer()?;
    if lhs_dtype != dtype || rhs_dtype != dtype {
        return None;
    }
    if dtype == DType::I8 {
        let (buffer, out_scale) = cuda::binary_i8_typed_row_scalar_output_buffer_no_host(
            &lhs_buffer,
            lhs_scale?,
            &rhs_buffer,
            rhs_scale?,
            rows,
            last_dim,
            scalar_on_rhs,
            cuda_binary_op(op),
        )
        .ok()?;
        return Some(Tensor::from_cuda_native_lowp_buffer_no_host_with_dtype(
            &out_shape,
            buffer,
            Device::Cuda,
            DType::I8,
            Some(out_scale),
        ));
    }
    let buffer = cuda::binary_lowp_typed_row_scalar_output_buffer_no_host(
        &lhs_buffer,
        &rhs_buffer,
        rows,
        last_dim,
        scalar_on_rhs,
        dtype,
        cuda_binary_op(op),
    )
    .ok()?;
    Some(Tensor::from_cuda_native_lowp_buffer_no_host_with_dtype(
        &out_shape,
        buffer,
        Device::Cuda,
        dtype,
        None,
    ))
}

fn try_cuda_binary_b1d_1h1_lowp_typed_output(
    lhs: &Tensor,
    rhs: &Tensor,
    op: BinaryOp,
) -> Option<Tensor> {
    if lhs.device() != Device::Cuda || rhs.device() != Device::Cuda || lhs.dtype() != rhs.dtype() {
        return None;
    }
    let dtype = lhs.dtype();
    if !matches!(dtype, DType::F16 | DType::BF16 | DType::I8) {
        return None;
    }
    let lhs_shape = lhs.shape_vec();
    let rhs_shape = rhs.shape_vec();
    let (batch, heads, dim, b1d_on_lhs) = b1d_1h1_broadcast_side(&lhs_shape, &rhs_shape)?;
    let (lhs_dtype, lhs_buffer, lhs_scale) = lhs.cloned_cuda_native_lowp_buffer()?;
    let (rhs_dtype, rhs_buffer, rhs_scale) = rhs.cloned_cuda_native_lowp_buffer()?;
    if lhs_dtype != dtype || rhs_dtype != dtype {
        return None;
    }
    if dtype == DType::I8 {
        let (buffer, out_scale) = cuda::binary_i8_typed_b1d_1h1_output_buffer_no_host(
            &lhs_buffer,
            lhs_scale?,
            &rhs_buffer,
            rhs_scale?,
            batch,
            heads,
            dim,
            b1d_on_lhs,
            cuda_binary_op(op),
        )
        .ok()?;
        return Some(Tensor::from_cuda_native_lowp_buffer_no_host_with_dtype(
            &[batch, heads, dim],
            buffer,
            Device::Cuda,
            DType::I8,
            Some(out_scale),
        ));
    }
    let buffer = cuda::binary_lowp_typed_b1d_1h1_output_buffer_no_host(
        &lhs_buffer,
        &rhs_buffer,
        batch,
        heads,
        dim,
        b1d_on_lhs,
        dtype,
        cuda_binary_op(op),
    )
    .ok()?;
    Some(Tensor::from_cuda_native_lowp_buffer_no_host_with_dtype(
        &[batch, heads, dim],
        buffer,
        Device::Cuda,
        dtype,
        None,
    ))
}

fn try_cuda_binary_b1d_1hd_lowp_typed_output(
    lhs: &Tensor,
    rhs: &Tensor,
    op: BinaryOp,
) -> Option<Tensor> {
    if lhs.device() != Device::Cuda || rhs.device() != Device::Cuda || lhs.dtype() != rhs.dtype() {
        return None;
    }
    let dtype = lhs.dtype();
    if !matches!(dtype, DType::F16 | DType::BF16 | DType::I8) {
        return None;
    }
    let lhs_shape = lhs.shape_vec();
    let rhs_shape = rhs.shape_vec();
    let (batch, heads, dim, b1d_on_lhs) = b1d_1hd_broadcast_side(&lhs_shape, &rhs_shape)?;
    let (lhs_dtype, lhs_buffer, lhs_scale) = lhs.cloned_cuda_native_lowp_buffer()?;
    let (rhs_dtype, rhs_buffer, rhs_scale) = rhs.cloned_cuda_native_lowp_buffer()?;
    if lhs_dtype != dtype || rhs_dtype != dtype {
        return None;
    }
    if dtype == DType::I8 {
        let (buffer, out_scale) = cuda::binary_i8_typed_b1d_1hd_output_buffer_no_host(
            &lhs_buffer,
            lhs_scale?,
            &rhs_buffer,
            rhs_scale?,
            batch,
            heads,
            dim,
            b1d_on_lhs,
            cuda_binary_op(op),
        )
        .ok()?;
        return Some(Tensor::from_cuda_native_lowp_buffer_no_host_with_dtype(
            &[batch, heads, dim],
            buffer,
            Device::Cuda,
            DType::I8,
            Some(out_scale),
        ));
    }
    let buffer = cuda::binary_lowp_typed_b1d_1hd_output_buffer_no_host(
        &lhs_buffer,
        &rhs_buffer,
        batch,
        heads,
        dim,
        b1d_on_lhs,
        dtype,
        cuda_binary_op(op),
    )
    .ok()?;
    Some(Tensor::from_cuda_native_lowp_buffer_no_host_with_dtype(
        &[batch, heads, dim],
        buffer,
        Device::Cuda,
        dtype,
        None,
    ))
}

fn try_cuda_binary_broadcast_lowp_typed_output(
    lhs: &Tensor,
    rhs: &Tensor,
    op: BinaryOp,
) -> Option<Tensor> {
    if lhs.device() != Device::Cuda || rhs.device() != Device::Cuda || lhs.dtype() != rhs.dtype() {
        return None;
    }
    let dtype = lhs.dtype();
    if !matches!(dtype, DType::F16 | DType::BF16 | DType::I8) {
        return None;
    }
    let lhs_shape = lhs.shape_vec();
    let rhs_shape = rhs.shape_vec();
    let out_shape = broadcast_shape(&lhs_shape, &rhs_shape)?;
    if out_shape.iter().product::<usize>() == 0 {
        return None;
    }
    let (lhs_dtype, lhs_buffer, lhs_scale) = lhs.cloned_cuda_native_lowp_buffer()?;
    let (rhs_dtype, rhs_buffer, rhs_scale) = rhs.cloned_cuda_native_lowp_buffer()?;
    if lhs_dtype != dtype || rhs_dtype != dtype {
        return None;
    }
    if dtype == DType::I8 {
        let (buffer, out_scale) = cuda::binary_i8_typed_broadcast_output_buffer_no_host(
            &lhs_buffer,
            lhs_scale?,
            &rhs_buffer,
            rhs_scale?,
            &lhs_shape,
            &rhs_shape,
            &out_shape,
            cuda_binary_op(op),
        )
        .ok()?;
        return Some(Tensor::from_cuda_native_lowp_buffer_no_host_with_dtype(
            &out_shape,
            buffer,
            Device::Cuda,
            DType::I8,
            Some(out_scale),
        ));
    }
    let buffer = cuda::binary_lowp_typed_broadcast_output_buffer_no_host(
        &lhs_buffer,
        &rhs_buffer,
        &lhs_shape,
        &rhs_shape,
        &out_shape,
        dtype,
        cuda_binary_op(op),
    )
    .ok()?;
    Some(Tensor::from_cuda_native_lowp_buffer_no_host_with_dtype(
        &out_shape,
        buffer,
        Device::Cuda,
        dtype,
        None,
    ))
}

fn try_cuda_binary_typed_row_broadcast_f32_buffer(
    lhs: &Tensor,
    rhs: &Tensor,
    op: BinaryOp,
) -> Option<cuda::CudaBuffer> {
    if lhs.device() != Device::Cuda || rhs.device() != Device::Cuda {
        return None;
    }
    let lhs_shape = lhs.shape_vec();
    let rhs_shape = rhs.shape_vec();
    let side = row_broadcast_side(&lhs_shape, &rhs_shape)?;
    let out_shape = match side {
        RowBroadcastSide::RhsVector => lhs_shape,
        RowBroadcastSide::LhsVector => rhs_shape,
    };
    let last_dim = *out_shape.last()?;
    let out_len = out_shape.iter().product::<usize>();
    if last_dim == 0 || out_len == 0 {
        return None;
    }
    let vector_on_rhs = matches!(side, RowBroadcastSide::RhsVector);
    let (lhs_buffer, lhs_dtype, lhs_scale) = cuda_binary_storage(lhs)?;
    let (rhs_buffer, rhs_dtype, rhs_scale) = cuda_binary_storage(rhs)?;
    if lhs_dtype == rhs_dtype && lhs_dtype != DType::F32 {
        return None;
    }
    cuda::binary_typed_lastdim_buffer_no_host(
        &lhs_buffer,
        lhs_dtype,
        lhs_scale,
        &rhs_buffer,
        rhs_dtype,
        rhs_scale,
        out_len,
        last_dim,
        vector_on_rhs,
        cuda_binary_op(op),
    )
    .ok()
}

fn try_cuda_binary_typed_row_scalar_f32_buffer(
    lhs: &Tensor,
    rhs: &Tensor,
    op: BinaryOp,
) -> Option<cuda::CudaBuffer> {
    if lhs.device() != Device::Cuda || rhs.device() != Device::Cuda {
        return None;
    }
    let lhs_shape = lhs.shape_vec();
    let rhs_shape = rhs.shape_vec();
    let (rows, last_dim, scalar_on_rhs) = row_scalar_broadcast_side(&lhs_shape, &rhs_shape)?;
    let (lhs_buffer, lhs_dtype, lhs_scale) = cuda_binary_storage(lhs)?;
    let (rhs_buffer, rhs_dtype, rhs_scale) = cuda_binary_storage(rhs)?;
    if lhs_dtype == DType::F32 && rhs_dtype == DType::F32 {
        return None;
    }
    cuda::binary_typed_row_scalar_buffer_no_host(
        &lhs_buffer,
        lhs_dtype,
        lhs_scale,
        &rhs_buffer,
        rhs_dtype,
        rhs_scale,
        rows,
        last_dim,
        scalar_on_rhs,
        cuda_binary_op(op),
    )
    .ok()
}

fn try_cuda_binary_typed_broadcast_f32_buffer(
    lhs: &Tensor,
    rhs: &Tensor,
    op: BinaryOp,
) -> Option<cuda::CudaBuffer> {
    if lhs.device() != Device::Cuda || rhs.device() != Device::Cuda {
        return None;
    }
    let lhs_shape = lhs.shape_vec();
    let rhs_shape = rhs.shape_vec();
    let out_shape = broadcast_shape(&lhs_shape, &rhs_shape)?;
    if out_shape.iter().product::<usize>() == 0 {
        return None;
    }
    let (lhs_buffer, lhs_dtype, lhs_scale) = cuda_binary_storage(lhs)?;
    let (rhs_buffer, rhs_dtype, rhs_scale) = cuda_binary_storage(rhs)?;
    if lhs_dtype == DType::F32 && rhs_dtype == DType::F32 {
        return None;
    }
    cuda::binary_typed_broadcast_buffer_no_host(
        &lhs_buffer,
        lhs_dtype,
        lhs_scale,
        &rhs_buffer,
        rhs_dtype,
        rhs_scale,
        &lhs_shape,
        &rhs_shape,
        &out_shape,
        cuda_binary_op(op),
    )
    .ok()
}

fn try_cuda_binary_typed_b1d_1h1_f32_buffer(
    lhs: &Tensor,
    rhs: &Tensor,
    op: BinaryOp,
) -> Option<cuda::CudaBuffer> {
    if lhs.device() != Device::Cuda || rhs.device() != Device::Cuda {
        return None;
    }
    let lhs_shape = lhs.shape_vec();
    let rhs_shape = rhs.shape_vec();
    let (batch, heads, dim, b1d_on_lhs) = b1d_1h1_broadcast_side(&lhs_shape, &rhs_shape)?;
    let (lhs_buffer, lhs_dtype, lhs_scale) = cuda_binary_storage(lhs)?;
    let (rhs_buffer, rhs_dtype, rhs_scale) = cuda_binary_storage(rhs)?;
    if lhs_dtype == DType::F32 && rhs_dtype == DType::F32 {
        return None;
    }
    cuda::binary_typed_b1d_1h1_buffer_no_host(
        &lhs_buffer,
        lhs_dtype,
        lhs_scale,
        &rhs_buffer,
        rhs_dtype,
        rhs_scale,
        batch,
        heads,
        dim,
        b1d_on_lhs,
        cuda_binary_op(op),
    )
    .ok()
}

fn try_cuda_binary_typed_b1d_1hd_f32_buffer(
    lhs: &Tensor,
    rhs: &Tensor,
    op: BinaryOp,
) -> Option<cuda::CudaBuffer> {
    if lhs.device() != Device::Cuda || rhs.device() != Device::Cuda {
        return None;
    }
    let lhs_shape = lhs.shape_vec();
    let rhs_shape = rhs.shape_vec();
    let (batch, heads, dim, b1d_on_lhs) = b1d_1hd_broadcast_side(&lhs_shape, &rhs_shape)?;
    let (lhs_buffer, lhs_dtype, lhs_scale) = cuda_binary_storage(lhs)?;
    let (rhs_buffer, rhs_dtype, rhs_scale) = cuda_binary_storage(rhs)?;
    if lhs_dtype == DType::F32 && rhs_dtype == DType::F32 {
        return None;
    }
    cuda::binary_typed_b1d_1hd_buffer_no_host(
        &lhs_buffer,
        lhs_dtype,
        lhs_scale,
        &rhs_buffer,
        rhs_dtype,
        rhs_scale,
        batch,
        heads,
        dim,
        b1d_on_lhs,
        cuda_binary_op(op),
    )
    .ok()
}

fn try_cuda_same_shape_mul_grads_native_lowp_buffers(
    lhs: &Tensor,
    rhs: &Tensor,
    grad: &cuda::CudaBuffer,
) -> Option<Result<(cuda::CudaBuffer, cuda::CudaBuffer), String>> {
    if lhs.device() != Device::Cuda
        || rhs.device() != Device::Cuda
        || lhs.dtype() != rhs.dtype()
        || lhs.shape_vec() != rhs.shape_vec()
        || grad.len() != lhs.len()
    {
        return None;
    }

    if let (Some((lhs_dtype, lhs_buffer, lhs_scale)), Some((rhs_dtype, rhs_buffer, rhs_scale))) = (
        lhs.cloned_cuda_native_lowp_buffer(),
        rhs.cloned_cuda_native_lowp_buffer(),
    ) && lhs_dtype == rhs_dtype
    {
        return Some(match lhs_dtype {
            DType::F16 => {
                cuda::mul_grad_f16_buffer_no_host(grad, &rhs_buffer).and_then(|lhs_grad| {
                    cuda::mul_grad_f16_buffer_no_host(grad, &lhs_buffer)
                        .map(|rhs_grad| (lhs_grad, rhs_grad))
                })
            }
            DType::BF16 => {
                cuda::mul_grad_bf16_buffer_no_host(grad, &rhs_buffer).and_then(|lhs_grad| {
                    cuda::mul_grad_bf16_buffer_no_host(grad, &lhs_buffer)
                        .map(|rhs_grad| (lhs_grad, rhs_grad))
                })
            }
            DType::I8 => {
                let lhs_scale = lhs_scale?;
                let rhs_scale = rhs_scale?;
                cuda::mul_grad_i8_buffer_no_host(grad, &rhs_buffer, rhs_scale).and_then(
                    |lhs_grad| {
                        cuda::mul_grad_i8_buffer_no_host(grad, &lhs_buffer, lhs_scale)
                            .map(|rhs_grad| (lhs_grad, rhs_grad))
                    },
                )
            }
            DType::F32 => return None,
        });
    }

    match (lhs.native_storage_owned(), rhs.native_storage_owned()) {
        (TensorStorageOwned::F16(lhs_data), TensorStorageOwned::F16(rhs_data)) => {
            let lhs_slice = lhs_data.as_slice_memory_order()?;
            let rhs_slice = rhs_data.as_slice_memory_order()?;
            let lhs_bits = lhs_slice.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            let rhs_bits = rhs_slice.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            Some(
                cuda::mul_grad_f16_host_no_host(grad, &rhs_bits).and_then(|lhs_grad| {
                    cuda::mul_grad_f16_host_no_host(grad, &lhs_bits)
                        .map(|rhs_grad| (lhs_grad, rhs_grad))
                }),
            )
        }
        (TensorStorageOwned::BF16(lhs_data), TensorStorageOwned::BF16(rhs_data)) => {
            let lhs_slice = lhs_data.as_slice_memory_order()?;
            let rhs_slice = rhs_data.as_slice_memory_order()?;
            let lhs_bits = lhs_slice.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            let rhs_bits = rhs_slice.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            Some(
                cuda::mul_grad_bf16_host_no_host(grad, &rhs_bits).and_then(|lhs_grad| {
                    cuda::mul_grad_bf16_host_no_host(grad, &lhs_bits)
                        .map(|rhs_grad| (lhs_grad, rhs_grad))
                }),
            )
        }
        (
            TensorStorageOwned::I8(lhs_data, lhs_scale),
            TensorStorageOwned::I8(rhs_data, rhs_scale),
        ) => {
            let lhs_slice = lhs_data.as_slice_memory_order()?;
            let rhs_slice = rhs_data.as_slice_memory_order()?;
            Some(
                cuda::mul_grad_i8_host_no_host(grad, rhs_slice, rhs_scale).and_then(|lhs_grad| {
                    cuda::mul_grad_i8_host_no_host(grad, lhs_slice, lhs_scale)
                        .map(|rhs_grad| (lhs_grad, rhs_grad))
                }),
            )
        }
        _ => None,
    }
}

fn try_cuda_same_shape_mul_grads_typed_buffers(
    lhs: &Tensor,
    rhs: &Tensor,
    grad: &cuda::CudaBuffer,
) -> Option<Result<(cuda::CudaBuffer, cuda::CudaBuffer), String>> {
    if lhs.device() != Device::Cuda
        || rhs.device() != Device::Cuda
        || lhs.shape_vec() != rhs.shape_vec()
        || lhs.is_empty()
        || grad.len() != lhs.len()
    {
        return None;
    }
    let (lhs_buffer, lhs_dtype, lhs_scale) = cuda_binary_storage(lhs)?;
    let (rhs_buffer, rhs_dtype, rhs_scale) = cuda_binary_storage(rhs)?;
    Some(cuda::mul_grad_typed_buffer_no_host(
        grad,
        &lhs_buffer,
        lhs_dtype,
        lhs_scale,
        &rhs_buffer,
        rhs_dtype,
        rhs_scale,
    ))
}

fn try_cuda_row_broadcast_mul_grads_native_lowp_buffers(
    lhs: &Tensor,
    rhs: &Tensor,
    grad: &cuda::CudaBuffer,
) -> Option<Result<(cuda::CudaBuffer, cuda::CudaBuffer), String>> {
    if lhs.device() != Device::Cuda || rhs.device() != Device::Cuda || lhs.dtype() != rhs.dtype() {
        return None;
    }
    let lhs_shape = lhs.shape_vec();
    let rhs_shape = rhs.shape_vec();
    let side = row_broadcast_side(&lhs_shape, &rhs_shape)?;
    let out_shape = match side {
        RowBroadcastSide::RhsVector => lhs_shape,
        RowBroadcastSide::LhsVector => rhs_shape,
    };
    let last_dim = *out_shape.last()?;
    let out_len = out_shape.iter().product::<usize>();
    if last_dim == 0 || out_len == 0 || grad.len() != out_len {
        return None;
    }
    let vector_on_rhs = matches!(side, RowBroadcastSide::RhsVector);

    let (Some((lhs_dtype, lhs_buffer, lhs_scale)), Some((rhs_dtype, rhs_buffer, rhs_scale))) = (
        lhs.cloned_cuda_native_lowp_buffer(),
        rhs.cloned_cuda_native_lowp_buffer(),
    ) else {
        return None;
    };
    if lhs_dtype != rhs_dtype {
        return None;
    }

    Some(match lhs_dtype {
        DType::F16 => cuda::mul_grad_f16_lastdim_buffer_no_host(
            grad,
            &lhs_buffer,
            &rhs_buffer,
            out_len,
            last_dim,
            vector_on_rhs,
        ),
        DType::BF16 => cuda::mul_grad_bf16_lastdim_buffer_no_host(
            grad,
            &lhs_buffer,
            &rhs_buffer,
            out_len,
            last_dim,
            vector_on_rhs,
        ),
        DType::I8 => cuda::mul_grad_i8_lastdim_buffer_no_host(
            grad,
            &lhs_buffer,
            lhs_scale?,
            &rhs_buffer,
            rhs_scale?,
            out_len,
            last_dim,
            vector_on_rhs,
        ),
        DType::F32 => return None,
    })
}

fn try_cuda_row_broadcast_mul_grads_typed_buffers(
    lhs: &Tensor,
    rhs: &Tensor,
    grad: &cuda::CudaBuffer,
) -> Option<Result<(cuda::CudaBuffer, cuda::CudaBuffer), String>> {
    if lhs.device() != Device::Cuda || rhs.device() != Device::Cuda {
        return None;
    }
    let lhs_shape = lhs.shape_vec();
    let rhs_shape = rhs.shape_vec();
    let side = row_broadcast_side(&lhs_shape, &rhs_shape)?;
    let out_shape = match side {
        RowBroadcastSide::RhsVector => lhs_shape,
        RowBroadcastSide::LhsVector => rhs_shape,
    };
    let last_dim = *out_shape.last()?;
    let out_len = out_shape.iter().product::<usize>();
    if last_dim == 0 || out_len == 0 || grad.len() != out_len {
        return None;
    }
    let vector_on_rhs = matches!(side, RowBroadcastSide::RhsVector);
    let (lhs_buffer, lhs_dtype, lhs_scale) = cuda_binary_storage(lhs)?;
    let (rhs_buffer, rhs_dtype, rhs_scale) = cuda_binary_storage(rhs)?;
    Some(cuda::mul_grad_typed_lastdim_buffer_no_host(
        grad,
        &lhs_buffer,
        lhs_dtype,
        lhs_scale,
        &rhs_buffer,
        rhs_dtype,
        rhs_scale,
        out_len,
        last_dim,
        vector_on_rhs,
    ))
}

fn try_cuda_broadcast_mul_grads_typed_buffers(
    lhs: &Tensor,
    rhs: &Tensor,
    grad: &cuda::CudaBuffer,
) -> Option<Result<(cuda::CudaBuffer, cuda::CudaBuffer), String>> {
    if lhs.device() != Device::Cuda || rhs.device() != Device::Cuda {
        return None;
    }
    let lhs_shape = lhs.shape_vec();
    let rhs_shape = rhs.shape_vec();
    if lhs_shape == rhs_shape || row_broadcast_side(&lhs_shape, &rhs_shape).is_some() {
        return None;
    }
    let out_shape = broadcast_shape(&lhs_shape, &rhs_shape)?;
    let out_len = out_shape.iter().product::<usize>();
    if out_len == 0 || grad.len() != out_len {
        return None;
    }
    let (lhs_buffer, lhs_dtype, lhs_scale) = cuda_binary_storage(lhs)?;
    let (rhs_buffer, rhs_dtype, rhs_scale) = cuda_binary_storage(rhs)?;
    Some(cuda::mul_grad_typed_broadcast_buffer_no_host(
        grad,
        &lhs_buffer,
        lhs_dtype,
        lhs_scale,
        &rhs_buffer,
        rhs_dtype,
        rhs_scale,
        &lhs_shape,
        &rhs_shape,
        &out_shape,
    ))
}

fn try_cuda_row_scalar_mul_grads_typed_buffers(
    lhs: &Tensor,
    rhs: &Tensor,
    grad: &cuda::CudaBuffer,
) -> Option<Result<(cuda::CudaBuffer, cuda::CudaBuffer), String>> {
    if lhs.device() != Device::Cuda || rhs.device() != Device::Cuda {
        return None;
    }
    let lhs_shape = lhs.shape_vec();
    let rhs_shape = rhs.shape_vec();
    let (rows, last_dim, scalar_on_rhs) = row_scalar_broadcast_side(&lhs_shape, &rhs_shape)?;
    let out_len = rows.checked_mul(last_dim)?;
    if out_len == 0 || grad.len() != out_len {
        return None;
    }
    let (lhs_buffer, lhs_dtype, lhs_scale) = cuda_binary_storage(lhs)?;
    let (rhs_buffer, rhs_dtype, rhs_scale) = cuda_binary_storage(rhs)?;
    Some(cuda::mul_grad_typed_row_scalar_buffer_no_host(
        grad,
        &lhs_buffer,
        lhs_dtype,
        lhs_scale,
        &rhs_buffer,
        rhs_dtype,
        rhs_scale,
        rows,
        last_dim,
        scalar_on_rhs,
    ))
}

fn try_cuda_b1d_1h1_mul_grads_typed_buffers(
    lhs: &Tensor,
    rhs: &Tensor,
    grad: &cuda::CudaBuffer,
) -> Option<Result<(cuda::CudaBuffer, cuda::CudaBuffer), String>> {
    if lhs.device() != Device::Cuda || rhs.device() != Device::Cuda {
        return None;
    }
    let lhs_shape = lhs.shape_vec();
    let rhs_shape = rhs.shape_vec();
    let (batch, heads, dim, b1d_on_lhs) = b1d_1h1_broadcast_side(&lhs_shape, &rhs_shape)?;
    let out_len = batch.checked_mul(heads)?.checked_mul(dim)?;
    if out_len == 0 || grad.len() != out_len {
        return None;
    }
    let (lhs_buffer, lhs_dtype, lhs_scale) = cuda_binary_storage(lhs)?;
    let (rhs_buffer, rhs_dtype, rhs_scale) = cuda_binary_storage(rhs)?;
    Some(cuda::mul_grad_typed_b1d_1h1_buffer_no_host(
        grad,
        &lhs_buffer,
        lhs_dtype,
        lhs_scale,
        &rhs_buffer,
        rhs_dtype,
        rhs_scale,
        batch,
        heads,
        dim,
        b1d_on_lhs,
    ))
}

fn try_cuda_b1d_1hd_mul_grads_typed_buffers(
    lhs: &Tensor,
    rhs: &Tensor,
    grad: &cuda::CudaBuffer,
) -> Option<Result<(cuda::CudaBuffer, cuda::CudaBuffer), String>> {
    if lhs.device() != Device::Cuda || rhs.device() != Device::Cuda {
        return None;
    }
    let lhs_shape = lhs.shape_vec();
    let rhs_shape = rhs.shape_vec();
    let (batch, heads, dim, b1d_on_lhs) = b1d_1hd_broadcast_side(&lhs_shape, &rhs_shape)?;
    let out_len = batch.checked_mul(heads)?.checked_mul(dim)?;
    if out_len == 0 || grad.len() != out_len {
        return None;
    }
    let (lhs_buffer, lhs_dtype, lhs_scale) = cuda_binary_storage(lhs)?;
    let (rhs_buffer, rhs_dtype, rhs_scale) = cuda_binary_storage(rhs)?;
    Some(cuda::mul_grad_typed_b1d_1hd_buffer_no_host(
        grad,
        &lhs_buffer,
        lhs_dtype,
        lhs_scale,
        &rhs_buffer,
        rhs_dtype,
        rhs_scale,
        batch,
        heads,
        dim,
        b1d_on_lhs,
    ))
}

fn try_cuda_scalar_mul_grads_typed_buffers(
    lhs: &Tensor,
    rhs: &Tensor,
    grad: &cuda::CudaBuffer,
) -> Option<Result<(cuda::CudaBuffer, cuda::CudaBuffer), String>> {
    if lhs.device() != Device::Cuda || rhs.device() != Device::Cuda {
        return None;
    }
    let lhs_shape = lhs.shape_vec();
    let rhs_shape = rhs.shape_vec();
    let out_shape = broadcast_shape(&lhs_shape, &rhs_shape)?;
    let out_len = out_shape.iter().product::<usize>();
    if out_len == 0 || grad.len() != out_len || lhs.len() == rhs.len() {
        return None;
    }
    let scalar_on_rhs = if lhs.len() == out_len && rhs.len() == 1 {
        true
    } else if lhs.len() == 1 && rhs.len() == out_len {
        false
    } else {
        return None;
    };
    let (lhs_buffer, lhs_dtype, lhs_scale) = cuda_binary_storage(lhs)?;
    let (rhs_buffer, rhs_dtype, rhs_scale) = cuda_binary_storage(rhs)?;
    Some(cuda::mul_grad_typed_scalar_buffer_no_host(
        grad,
        &lhs_buffer,
        lhs_dtype,
        lhs_scale,
        &rhs_buffer,
        rhs_dtype,
        rhs_scale,
        out_len,
        scalar_on_rhs,
    ))
}

fn try_cuda_add_sub_grads_shape_only_buffers(
    lhs: &Tensor,
    rhs: &Tensor,
    grad: &cuda::CudaBuffer,
    op: BinaryOp,
) -> Option<Result<(cuda::CudaBuffer, cuda::CudaBuffer), String>> {
    if !matches!(op, BinaryOp::Add | BinaryOp::Sub)
        || lhs.device() != Device::Cuda
        || rhs.device() != Device::Cuda
    {
        return None;
    }
    let lhs_shape = lhs.shape_vec();
    let rhs_shape = rhs.shape_vec();
    if lhs_shape == rhs_shape {
        if grad.len() != lhs.len() {
            return None;
        }
        return Some(cuda::add_sub_backward_f32_buffers(
            grad,
            lhs.len(),
            cuda_binary_op(op),
        ));
    }

    let out_shape = broadcast_shape(&lhs_shape, &rhs_shape)?;
    let out_len = out_shape.iter().product::<usize>();
    if out_len == 0 || grad.len() != out_len {
        return None;
    }
    if let Some((batch, heads, dim, b1d_on_lhs)) = b1d_1hd_broadcast_side(&lhs_shape, &rhs_shape) {
        return Some(cuda::add_sub_backward_b1d_1hd_f32_buffers(
            grad,
            batch,
            heads,
            dim,
            b1d_on_lhs,
            cuda_binary_op(op),
        ));
    }
    if let Some((batch, heads, dim, b1d_on_lhs)) = b1d_1h1_broadcast_side(&lhs_shape, &rhs_shape) {
        return Some(cuda::add_sub_backward_b1d_1h1_f32_buffers(
            grad,
            batch,
            heads,
            dim,
            b1d_on_lhs,
            cuda_binary_op(op),
        ));
    }
    if let Some((rows, last_dim, scalar_on_rhs)) = row_scalar_broadcast_side(&lhs_shape, &rhs_shape)
    {
        return Some(cuda::add_sub_backward_row_scalar_f32_buffers(
            grad,
            rows,
            last_dim,
            scalar_on_rhs,
            cuda_binary_op(op),
        ));
    }
    if lhs.len() == out_len && rhs.len() == 1 {
        return Some(cuda::add_sub_backward_scalar_f32_buffers(
            grad,
            out_len,
            true,
            cuda_binary_op(op),
        ));
    }
    if lhs.len() == 1 && rhs.len() == out_len {
        return Some(cuda::add_sub_backward_scalar_f32_buffers(
            grad,
            out_len,
            false,
            cuda_binary_op(op),
        ));
    }
    if let Some(side) = row_broadcast_side(&lhs_shape, &rhs_shape) {
        let last_dim = *out_shape.last()?;
        if last_dim == 0 {
            return None;
        }
        return Some(cuda::add_sub_backward_lastdim_f32_buffers(
            grad,
            out_len,
            last_dim,
            matches!(side, RowBroadcastSide::RhsVector),
            cuda_binary_op(op),
        ));
    }

    Some(cuda::add_sub_broadcast_backward_f32_buffers(
        grad,
        &lhs_shape,
        &rhs_shape,
        &out_shape,
        cuda_binary_op(op),
    ))
}

fn try_cuda_sum_native_lowp(
    input: &Tensor,
) -> Option<Result<(cuda::CudaBuffer, Vec<f32>), String>> {
    if input.device() != Device::Cuda || input.is_empty() {
        return None;
    }

    let (dtype, buffer, scale) = input.cloned_cuda_native_lowp_buffer()?;
    Some(match dtype {
        DType::F16 => cuda::sum_f16_buffer(&buffer),
        DType::BF16 => cuda::sum_bf16_buffer(&buffer),
        DType::I8 => cuda::sum_i8_buffer(&buffer, scale?),
        DType::F32 => return None,
    })
}

fn try_sum_no_grad_native(input: &Tensor) -> Option<f32> {
    if input.device() != Device::Cpu {
        return None;
    }

    if let TensorStorageOwned::I8(data, scale) = input.native_storage_owned() {
        return data
            .as_slice_memory_order()
            .and_then(|slice| sum_i8_arch(slice, scale))
            .or_else(|| Some(data.iter().map(|&v| (v as f32) * scale).sum()));
    }

    input.with_storage_view_preferring(StoragePreference::Native, |view| match view {
        TensorStorageView::F16(view) => view
            .as_slice_memory_order()
            .and_then(sum_f16_arch)
            .or_else(|| Some(view.iter().map(|v| v.to_f32()).sum())),
        TensorStorageView::BF16(view) => view
            .as_slice_memory_order()
            .and_then(sum_bf16_arch)
            .or_else(|| Some(view.iter().map(|v| v.to_f32()).sum())),
        TensorStorageView::F32(_) => None,
    })
}

fn binary_no_grad(lhs: &Tensor, rhs: &Tensor, op: BinaryOp) -> Tensor {
    let output_device = assert_same_device(lhs, rhs, "binary op");
    let output_dtype = if lhs.dtype() == rhs.dtype() {
        lhs.dtype()
    } else {
        DType::F32
    };

    if let Some(out) = try_binary_no_grad_native_same_shape(lhs, rhs, op)
        .or_else(|| try_binary_no_grad_native_row_broadcast(lhs, rhs, op))
    {
        return out;
    }
    if let Some(data) = try_mixed_add_sub_native_same_shape_f32(lhs, rhs, op)
        .or_else(|| try_mixed_add_sub_native_row_broadcast_f32(lhs, rhs, op))
        .or_else(|| try_mixed_add_sub_native_row_scalar_f32(lhs, rhs, op))
        .or_else(|| try_mixed_add_sub_native_scalar_broadcast_f32(lhs, rhs, op))
        .or_else(|| try_mixed_mul_native_same_shape_f32(lhs, rhs, op))
        .or_else(|| try_mixed_mul_native_row_broadcast_f32(lhs, rhs, op))
        .or_else(|| try_mixed_mul_native_row_scalar_f32(lhs, rhs, op))
        .or_else(|| try_mixed_mul_native_scalar_broadcast_f32(lhs, rhs, op))
    {
        return Tensor::from_f32_data_no_grad_with_device_dtype(data, DType::F32, output_device);
    }

    if output_device == Device::Cuda
        && (cuda::should_accelerate_elementwise(lhs.len()) || is_strict_device_execution())
        && let Some(out) = try_cuda_binary_same_dtype_lowp_typed_output(lhs, rhs, op)
            .or_else(|| try_cuda_binary_row_broadcast_lowp_typed_output(lhs, rhs, op))
            .or_else(|| try_cuda_binary_row_scalar_lowp_typed_output(lhs, rhs, op))
            .or_else(|| try_cuda_binary_b1d_1h1_lowp_typed_output(lhs, rhs, op))
            .or_else(|| try_cuda_binary_b1d_1hd_lowp_typed_output(lhs, rhs, op))
            .or_else(|| try_cuda_binary_broadcast_lowp_typed_output(lhs, rhs, op))
    {
        return out;
    }

    if output_device == Device::Cuda
        && (cuda::should_accelerate_elementwise(lhs.len()) || is_strict_device_execution())
        && let Some(buffer) = try_cuda_binary_native_same_shape_f32_buffer(lhs, rhs, op)
            .or_else(|| try_cuda_binary_native_row_broadcast_f32_buffer(lhs, rhs, op))
            .or_else(|| try_cuda_binary_typed_same_shape_f32_buffer(lhs, rhs, op))
            .or_else(|| try_cuda_binary_typed_row_broadcast_f32_buffer(lhs, rhs, op))
            .or_else(|| try_cuda_binary_typed_row_scalar_f32_buffer(lhs, rhs, op))
            .or_else(|| try_cuda_binary_typed_b1d_1h1_f32_buffer(lhs, rhs, op))
            .or_else(|| try_cuda_binary_typed_b1d_1hd_f32_buffer(lhs, rhs, op))
    {
        let out_shape =
            broadcast_shape(&lhs.shape_vec(), &rhs.shape_vec()).unwrap_or_else(|| lhs.shape_vec());
        return Tensor::from_cuda_f32_buffer_no_host_with_dtype(
            &out_shape,
            buffer,
            output_device,
            output_dtype,
        );
    }

    if output_device == Device::Cuda {
        let lhs_shape = lhs.shape_vec();
        let rhs_shape = rhs.shape_vec();
        if let Some(out_shape) = broadcast_shape(&lhs_shape, &rhs_shape) {
            let out_len = out_shape.iter().product::<usize>();
            if out_len > 0
                && (cuda::should_accelerate_elementwise(out_len) || is_strict_device_execution())
            {
                let cuda_op = cuda_binary_op(op);
                let cuda_out = if lhs_shape == rhs_shape {
                    lhs.with_cuda_f32_buffer(|lhs_buf| {
                        rhs.with_cuda_f32_buffer(|rhs_buf| {
                            cuda::binary_f32_buffer(lhs_buf, rhs_buf, cuda_op)
                        })
                    })
                } else {
                    lhs.with_cuda_f32_buffer(|lhs_buf| {
                        rhs.with_cuda_f32_buffer(|rhs_buf| {
                            cuda::binary_broadcast_f32_buffer(
                                lhs_buf, rhs_buf, &lhs_shape, &rhs_shape, &out_shape, cuda_op,
                            )
                        })
                    })
                };
                match cuda_out {
                    Ok(buffer) => {
                        if output_dtype == DType::F32 {
                            return Tensor::from_cuda_f32_buffer_no_host(
                                &out_shape,
                                buffer,
                                output_device,
                            );
                        }
                        if matches!(output_dtype, DType::F16 | DType::BF16) {
                            match cuda::f32_to_lowp_storage_no_host(&buffer, output_dtype) {
                                Ok(lowp_buffer) => {
                                    return Tensor::from_cuda_native_lowp_buffer_no_host_with_dtype(
                                        &out_shape,
                                        lowp_buffer,
                                        output_device,
                                        output_dtype,
                                        None,
                                    );
                                }
                                Err(err) => {
                                    assert!(
                                        !is_strict_device_execution(),
                                        "binary op CUDA low precision output conversion failed while strict device execution is enabled: {err}"
                                    );
                                }
                            }
                        }
                        if output_dtype == DType::I8 {
                            match cuda::quantize_f32_to_i8_dynamic_no_host(&buffer) {
                                Ok((i8_buffer, scale)) => {
                                    return Tensor::from_cuda_native_lowp_buffer_no_host_with_dtype(
                                        &out_shape,
                                        i8_buffer,
                                        output_device,
                                        output_dtype,
                                        Some(scale),
                                    );
                                }
                                Err(err) => {
                                    assert!(
                                        !is_strict_device_execution(),
                                        "binary op CUDA i8 output quantization failed while strict device execution is enabled: {err}"
                                    );
                                }
                            }
                        }
                    }
                    Err(err) => {
                        assert!(
                            !is_strict_device_execution(),
                            "binary op CUDA forward failed while strict device execution is enabled: {err}"
                        );
                    }
                }
            }
        }
    }

    lhs.with_storage_view_preferring(StoragePreference::F32Compute, |lhs_view| {
        rhs.with_storage_view_preferring(StoragePreference::F32Compute, |rhs_view| {
            let lhs_f32 = match lhs_view {
                TensorStorageView::F32(view) => view,
                TensorStorageView::F16(_) => {
                    unreachable!("f32 compute preference should expose f32 view")
                }
                TensorStorageView::BF16(_) => {
                    unreachable!("f32 compute preference should expose f32 view")
                }
            };
            let rhs_f32 = match rhs_view {
                TensorStorageView::F32(view) => view,
                TensorStorageView::F16(_) => {
                    unreachable!("f32 compute preference should expose f32 view")
                }
                TensorStorageView::BF16(_) => {
                    unreachable!("f32 compute preference should expose f32 view")
                }
            };

            if let Some(out_shape) = broadcast_shape(lhs_f32.shape(), rhs_f32.shape()) {
                let out_len = out_shape.iter().product::<usize>();
                if output_device == crate::autograd::Device::Cuda
                && out_len > 0
                && (cuda::should_accelerate_elementwise(
                    out_len,
                ) || is_strict_device_execution())
            {
                let cuda_op = cuda_binary_op(op);
                if output_dtype == DType::F32 {
                    let cuda_out = if lhs_f32.shape() == rhs_f32.shape() {
                        lhs.with_cuda_f32_buffer(|lhs_buf| {
                            rhs.with_cuda_f32_buffer(|rhs_buf| {
                                cuda::binary_f32_buffer(lhs_buf, rhs_buf, cuda_op)
                            })
                        })
                    } else {
                        lhs.with_cuda_f32_buffer(|lhs_buf| {
                            rhs.with_cuda_f32_buffer(|rhs_buf| {
                                cuda::binary_broadcast_f32_buffer(
                                    lhs_buf,
                                    rhs_buf,
                                    lhs_f32.shape(),
                                    rhs_f32.shape(),
                                    &out_shape,
                                    cuda_op,
                                )
                            })
                        })
                    };
                    match cuda_out {
                        Ok(buffer) => {
                            return Tensor::from_cuda_f32_buffer_no_host(
                                &out_shape,
                                buffer,
                                output_device,
                            );
                        }
                        Err(err) => {
                            assert!(
                                !is_strict_device_execution(),
                                "binary op CUDA forward failed while strict device execution is enabled: {err}"
                            );
                        }
                    }
                } else {
                    let cuda_out = if lhs_f32.shape() == rhs_f32.shape() {
                        lhs.with_cuda_f32_buffer(|lhs_buf| {
                            rhs.with_cuda_f32_buffer(|rhs_buf| {
                                cuda::binary_f32(lhs_buf, rhs_buf, cuda_op)
                            })
                        })
                    } else {
                        lhs.with_cuda_f32_buffer(|lhs_buf| {
                            rhs.with_cuda_f32_buffer(|rhs_buf| {
                                cuda::binary_broadcast_f32(
                                    lhs_buf,
                                    rhs_buf,
                                    lhs_f32.shape(),
                                    rhs_f32.shape(),
                                    &out_shape,
                                    cuda_op,
                                )
                            })
                        })
                    };
                    match cuda_out {
                        Ok((buffer, out)) => {
                            let out = ndarray::Array::from_shape_vec(IxDyn(&out_shape), out)
                                .expect("CUDA binary op output shape build failed")
                                .into_dyn();
                            return Tensor::from_f32_data_no_grad_with_device_dtype_and_cuda_buffer(
                                out,
                                output_dtype,
                                output_device,
                                Some(buffer),
                            );
                        }
                        Err(err) => {
                            assert!(
                                !is_strict_device_execution(),
                                "binary op CUDA forward failed while strict device execution is enabled: {err}"
                            );
                        }
                    }
                }
                }
            }

            Tensor::from_f32_data_no_grad_with_device_dtype(
                apply_binary_views(lhs_f32, rhs_f32, op),
                output_dtype,
                output_device,
            )
        })
    })
}

fn reduce_gradient(grad: ArrayViewD<'_, f32>, target_shape: &[usize]) -> ArrayD<f32> {
    if grad.shape() == target_shape {
        return grad.to_owned().into_dyn();
    }

    if let (Some(&grad_last), Some(&target_last), Some(grad_slice)) = (
        grad.shape().last(),
        target_shape.last(),
        grad.as_slice_memory_order(),
    ) {
        let target_is_pure_lastdim = target_last > 0
            && grad_last == target_last
            && target_shape.iter().product::<usize>() == target_last;
        if target_is_pure_lastdim {
            let mut reduced = vec![0.0f32; target_last];
            for row in grad_slice.chunks_exact(target_last) {
                for (dst, &value) in reduced.iter_mut().zip(row) {
                    *dst += value;
                }
            }
            return ArrayD::from_shape_vec(IxDyn(target_shape), reduced)
                .expect("last-dim reduced gradient shape build failed");
        }
    }

    let mut res = grad.to_owned().into_dyn();
    let g_ndim = res.ndim();
    let t_ndim = target_shape.len();

    if g_ndim > t_ndim {
        for _ in 0..(g_ndim - t_ndim) {
            res = res.sum_axis(ndarray::Axis(0));
        }
    }

    for i in 0..res.ndim() {
        if target_shape[i] == 1 && res.shape()[i] > 1 {
            let summed = res.sum_axis(ndarray::Axis(i));
            res = summed.insert_axis(ndarray::Axis(i));
        } else if target_shape[i] != res.shape()[i] {
            panic!(
                "Gradient shape mismatch. Grad: {:?}, Target: {:?}",
                grad.shape(),
                target_shape
            );
        }
    }

    if res.shape() != target_shape {
        if res.len() == target_shape.iter().product::<usize>() {
            return res.into_shape(target_shape).unwrap();
        }
        panic!("Reduction failed.");
    }

    res
}

fn cuda_binary_op(op: BinaryOp) -> cuda::BinaryOp {
    match op {
        BinaryOp::Add => cuda::BinaryOp::Add,
        BinaryOp::Sub => cuda::BinaryOp::Sub,
        BinaryOp::Mul => cuda::BinaryOp::Mul,
    }
}

fn broadcast_shape(lhs_shape: &[usize], rhs_shape: &[usize]) -> Option<Vec<usize>> {
    let ndim = lhs_shape.len().max(rhs_shape.len());
    let mut out = vec![1usize; ndim];
    for i in 0..ndim {
        let lhs_idx = lhs_shape.len() as isize - 1 - i as isize;
        let rhs_idx = rhs_shape.len() as isize - 1 - i as isize;
        let lhs_dim = if lhs_idx >= 0 {
            lhs_shape[lhs_idx as usize]
        } else {
            1
        };
        let rhs_dim = if rhs_idx >= 0 {
            rhs_shape[rhs_idx as usize]
        } else {
            1
        };
        if lhs_dim != rhs_dim && lhs_dim != 1 && rhs_dim != 1 {
            return None;
        }
        out[ndim - 1 - i] = lhs_dim.max(rhs_dim);
    }
    Some(out)
}

fn array_from_shape_vec(shape: &[usize], data: Vec<f32>) -> Option<ArrayD<f32>> {
    ArrayD::from_shape_vec(IxDyn(shape), data).ok()
}

fn apply_f16_slices_to_f32(lhs: &[f16], rhs: &[f16], out: &mut [f32], op: BinaryOp) {
    let handled = match op {
        BinaryOp::Add => add_f16_f16_to_f32_arch(lhs, rhs, out),
        BinaryOp::Sub => sub_f16_f16_to_f32_arch(lhs, rhs, out),
        BinaryOp::Mul => mul_f16_f16_to_f32_arch(lhs, rhs, out),
    };
    if !handled {
        for ((dst, &a), &b) in out.iter_mut().zip(lhs).zip(rhs) {
            *dst = match op {
                BinaryOp::Add => a.to_f32() + b.to_f32(),
                BinaryOp::Sub => a.to_f32() - b.to_f32(),
                BinaryOp::Mul => a.to_f32() * b.to_f32(),
            };
        }
    }
}

fn apply_bf16_slices_to_f32(lhs: &[bf16], rhs: &[bf16], out: &mut [f32], op: BinaryOp) {
    let handled = match op {
        BinaryOp::Add => add_bf16_bf16_to_f32_arch(lhs, rhs, out),
        BinaryOp::Sub => sub_bf16_bf16_to_f32_arch(lhs, rhs, out),
        BinaryOp::Mul => mul_bf16_bf16_to_f32_arch(lhs, rhs, out),
    };
    if !handled {
        for ((dst, &a), &b) in out.iter_mut().zip(lhs).zip(rhs) {
            *dst = match op {
                BinaryOp::Add => a.to_f32() + b.to_f32(),
                BinaryOp::Sub => a.to_f32() - b.to_f32(),
                BinaryOp::Mul => a.to_f32() * b.to_f32(),
            };
        }
    }
}

fn apply_i8_slices_to_f32(
    lhs: &[i8],
    lhs_scale: f32,
    rhs: &[i8],
    rhs_scale: f32,
    out: &mut [f32],
    op: BinaryOp,
) {
    let handled = match op {
        BinaryOp::Add => add_i8_i8_to_f32_arch(lhs, lhs_scale, rhs, rhs_scale, out),
        BinaryOp::Sub => sub_i8_i8_to_f32_arch(lhs, lhs_scale, rhs, rhs_scale, out),
        BinaryOp::Mul => mul_i8_i8_to_f32_arch(lhs, lhs_scale, rhs, rhs_scale, out),
    };
    if !handled {
        for ((dst, &a), &b) in out.iter_mut().zip(lhs).zip(rhs) {
            let lhs_v = (a as f32) * lhs_scale;
            let rhs_v = (b as f32) * rhs_scale;
            *dst = match op {
                BinaryOp::Add => lhs_v + rhs_v,
                BinaryOp::Sub => lhs_v - rhs_v,
                BinaryOp::Mul => lhs_v * rhs_v,
            };
        }
    }
}

fn apply_mixed_row_broadcast_f16(
    f32_slice: &[f32],
    lowp_slice: &[f16],
    side: RowBroadcastSide,
    rows: usize,
    row_len: usize,
    out: &mut [f32],
) {
    for row in 0..rows {
        let start = row * row_len;
        match side {
            RowBroadcastSide::RhsVector => {
                mul_f32_f16_slice_to_f32(
                    &f32_slice[start..start + row_len],
                    lowp_slice,
                    &mut out[start..start + row_len],
                );
            }
            RowBroadcastSide::LhsVector => {
                mul_f32_f16_slice_to_f32(
                    f32_slice,
                    &lowp_slice[start..start + row_len],
                    &mut out[start..start + row_len],
                );
            }
        }
    }
}

fn apply_mixed_row_broadcast_bf16(
    f32_slice: &[f32],
    lowp_slice: &[bf16],
    side: RowBroadcastSide,
    rows: usize,
    row_len: usize,
    out: &mut [f32],
) {
    for row in 0..rows {
        let start = row * row_len;
        match side {
            RowBroadcastSide::RhsVector => {
                mul_f32_bf16_slice_to_f32(
                    &f32_slice[start..start + row_len],
                    lowp_slice,
                    &mut out[start..start + row_len],
                );
            }
            RowBroadcastSide::LhsVector => {
                mul_f32_bf16_slice_to_f32(
                    f32_slice,
                    &lowp_slice[start..start + row_len],
                    &mut out[start..start + row_len],
                );
            }
        }
    }
}

fn apply_mixed_row_broadcast_i8(
    f32_slice: &[f32],
    lowp_slice: &[i8],
    lowp_scale: f32,
    side: RowBroadcastSide,
    rows: usize,
    row_len: usize,
    out: &mut [f32],
) {
    for row in 0..rows {
        let start = row * row_len;
        match side {
            RowBroadcastSide::RhsVector => {
                mul_f32_i8_slice_to_f32(
                    &f32_slice[start..start + row_len],
                    lowp_slice,
                    lowp_scale,
                    &mut out[start..start + row_len],
                );
            }
            RowBroadcastSide::LhsVector => {
                mul_f32_i8_slice_to_f32(
                    f32_slice,
                    &lowp_slice[start..start + row_len],
                    lowp_scale,
                    &mut out[start..start + row_len],
                );
            }
        }
    }
}

fn mul_grad_by_native_operand(
    grad: ArrayViewD<'_, f32>,
    operand: &Tensor,
    shape: &[usize],
) -> Option<ArrayD<f32>> {
    let len = grad.len();
    let grad_slice = grad.as_slice_memory_order();

    match operand.native_storage_owned() {
        TensorStorageOwned::F32(data) => {
            if let (Some(g), Some(x)) = (grad_slice, data.as_slice_memory_order()) {
                let mut out = vec![0.0f32; len];
                for ((dst, &gv), &xv) in out.iter_mut().zip(g.iter()).zip(x.iter()) {
                    *dst = gv * xv;
                }
                return array_from_shape_vec(shape, out);
            }
            array_from_shape_vec(
                shape,
                grad.iter().zip(data.iter()).map(|(&g, &x)| g * x).collect(),
            )
        }
        TensorStorageOwned::F16(data) => {
            if let (Some(g), Some(x)) = (grad_slice, data.as_slice_memory_order()) {
                let mut out = vec![0.0f32; len];
                if !mul_f32_f16_to_f32_arch(g, x, &mut out) {
                    for ((dst, &gv), &xv) in out.iter_mut().zip(g.iter()).zip(x.iter()) {
                        *dst = gv * xv.to_f32();
                    }
                }
                return array_from_shape_vec(shape, out);
            }
            array_from_shape_vec(
                shape,
                grad.iter()
                    .zip(data.iter())
                    .map(|(&g, &x)| g * x.to_f32())
                    .collect(),
            )
        }
        TensorStorageOwned::BF16(data) => {
            if let (Some(g), Some(x)) = (grad_slice, data.as_slice_memory_order()) {
                let mut out = vec![0.0f32; len];
                if !mul_f32_bf16_to_f32_arch(g, x, &mut out) {
                    for ((dst, &gv), &xv) in out.iter_mut().zip(g.iter()).zip(x.iter()) {
                        *dst = gv * xv.to_f32();
                    }
                }
                return array_from_shape_vec(shape, out);
            }
            array_from_shape_vec(
                shape,
                grad.iter()
                    .zip(data.iter())
                    .map(|(&g, &x)| g * x.to_f32())
                    .collect(),
            )
        }
        TensorStorageOwned::I8(data, scale) => {
            if let (Some(g), Some(x)) = (grad_slice, data.as_slice_memory_order()) {
                let mut out = vec![0.0f32; len];
                if !mul_f32_i8_to_f32_arch(g, x, scale, &mut out) {
                    for ((dst, &gv), &xv) in out.iter_mut().zip(g.iter()).zip(x.iter()) {
                        *dst = gv * (xv as f32) * scale;
                    }
                }
                return array_from_shape_vec(shape, out);
            }
            array_from_shape_vec(
                shape,
                grad.iter()
                    .zip(data.iter())
                    .map(|(&g, &x)| g * (x as f32) * scale)
                    .collect(),
            )
        }
    }
}

fn try_same_shape_mul_grads_native(
    lhs: &Tensor,
    rhs: &Tensor,
    grad: ArrayViewD<'_, f32>,
) -> Option<NativeMulGradResult> {
    let lhs_shape = lhs.shape_vec();
    let rhs_shape = rhs.shape_vec();
    if lhs_shape != rhs_shape || grad.shape() != lhs_shape.as_slice() {
        return None;
    }

    let g_lhs = mul_grad_by_native_operand(grad.view(), rhs, &lhs_shape)?;
    let g_rhs = mul_grad_by_native_operand(grad, lhs, &rhs_shape)?;
    Some((g_lhs, g_rhs, lhs_shape, rhs_shape))
}

fn mul_f32_f16_slice_to_f32(grad: &[f32], operand: &[f16], out: &mut [f32]) {
    if !mul_f32_f16_to_f32_arch(grad, operand, out) {
        for ((dst, &g), &x) in out.iter_mut().zip(grad).zip(operand) {
            *dst = g * x.to_f32();
        }
    }
}

fn mul_f32_bf16_slice_to_f32(grad: &[f32], operand: &[bf16], out: &mut [f32]) {
    if !mul_f32_bf16_to_f32_arch(grad, operand, out) {
        for ((dst, &g), &x) in out.iter_mut().zip(grad).zip(operand) {
            *dst = g * x.to_f32();
        }
    }
}

fn mul_f32_i8_slice_to_f32(grad: &[f32], operand: &[i8], scale: f32, out: &mut [f32]) {
    if !mul_f32_i8_to_f32_arch(grad, operand, scale, out) {
        for ((dst, &g), &x) in out.iter_mut().zip(grad).zip(operand) {
            *dst = g * (x as f32) * scale;
        }
    }
}

fn mul_f32_f32_slice_to_f32(grad: &[f32], operand: &[f32], out: &mut [f32]) {
    for ((dst, &g), &x) in out.iter_mut().zip(grad).zip(operand) {
        *dst = g * x;
    }
}

fn accumulate_slice(dst: &mut [f32], src: &[f32]) {
    for (dst, &src) in dst.iter_mut().zip(src) {
        *dst += src;
    }
}

fn dot_f32_f32_slice(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter().zip(rhs).map(|(&a, &b)| a * b).sum()
}

fn dot_f32_f16_slice(lhs: &[f32], rhs: &[f16]) -> f32 {
    lhs.iter().zip(rhs).map(|(&a, &b)| a * b.to_f32()).sum()
}

fn dot_f32_bf16_slice(lhs: &[f32], rhs: &[bf16]) -> f32 {
    lhs.iter().zip(rhs).map(|(&a, &b)| a * b.to_f32()).sum()
}

fn dot_f32_i8_slice(lhs: &[f32], rhs: &[i8], scale: f32) -> f32 {
    lhs.iter()
        .zip(rhs)
        .map(|(&a, &b)| a * (b as f32) * scale)
        .sum()
}

fn try_row_broadcast_mul_grads_native(
    lhs: &Tensor,
    rhs: &Tensor,
    grad: ArrayViewD<'_, f32>,
) -> Option<NativeMulGradResult> {
    if lhs.device() != Device::Cpu || rhs.device() != Device::Cpu || lhs.dtype() != rhs.dtype() {
        return None;
    }
    let lhs_shape = lhs.shape_vec();
    let rhs_shape = rhs.shape_vec();
    let side = row_broadcast_side(&lhs_shape, &rhs_shape)?;
    let out_shape = match side {
        RowBroadcastSide::RhsVector => lhs_shape.clone(),
        RowBroadcastSide::LhsVector => rhs_shape.clone(),
    };
    if grad.shape() != out_shape.as_slice() {
        return None;
    }
    let row_len = *out_shape.last()?;
    if row_len == 0 {
        return Some((
            ArrayD::zeros(IxDyn(&lhs_shape)),
            ArrayD::zeros(IxDyn(&rhs_shape)),
            lhs_shape,
            rhs_shape,
        ));
    }
    let rows = out_shape.iter().product::<usize>() / row_len;
    let grad_slice = grad.as_slice_memory_order()?;

    match lhs.dtype() {
        DType::F16 => lhs.with_storage_view_preferring(StoragePreference::Native, |lhs_view| {
            rhs.with_storage_view_preferring(StoragePreference::Native, |rhs_view| {
                let (TensorStorageView::F16(lhs_view), TensorStorageView::F16(rhs_view)) =
                    (lhs_view, rhs_view)
                else {
                    return None;
                };
                let (Some(lhs_slice), Some(rhs_slice)) = (
                    lhs_view.as_slice_memory_order(),
                    rhs_view.as_slice_memory_order(),
                ) else {
                    return None;
                };
                let mut lhs_grad = vec![0.0f32; lhs_shape.iter().product()];
                let mut rhs_grad = vec![0.0f32; rhs_shape.iter().product()];
                let mut tmp = vec![0.0f32; row_len];
                match side {
                    RowBroadcastSide::RhsVector => {
                        for row in 0..rows {
                            let start = row * row_len;
                            let g_row = &grad_slice[start..start + row_len];
                            mul_f32_f16_slice_to_f32(
                                g_row,
                                rhs_slice,
                                &mut lhs_grad[start..start + row_len],
                            );
                            mul_f32_f16_slice_to_f32(
                                g_row,
                                &lhs_slice[start..start + row_len],
                                &mut tmp,
                            );
                            for (dst, &v) in rhs_grad.iter_mut().zip(tmp.iter()) {
                                *dst += v;
                            }
                        }
                    }
                    RowBroadcastSide::LhsVector => {
                        for row in 0..rows {
                            let start = row * row_len;
                            let g_row = &grad_slice[start..start + row_len];
                            mul_f32_f16_slice_to_f32(
                                g_row,
                                &rhs_slice[start..start + row_len],
                                &mut tmp,
                            );
                            for (dst, &v) in lhs_grad.iter_mut().zip(tmp.iter()) {
                                *dst += v;
                            }
                            mul_f32_f16_slice_to_f32(
                                g_row,
                                lhs_slice,
                                &mut rhs_grad[start..start + row_len],
                            );
                        }
                    }
                }
                Some((
                    array_from_shape_vec(&lhs_shape, lhs_grad)?,
                    array_from_shape_vec(&rhs_shape, rhs_grad)?,
                    lhs_shape,
                    rhs_shape,
                ))
            })
        }),
        DType::BF16 => lhs.with_storage_view_preferring(StoragePreference::Native, |lhs_view| {
            rhs.with_storage_view_preferring(StoragePreference::Native, |rhs_view| {
                let (TensorStorageView::BF16(lhs_view), TensorStorageView::BF16(rhs_view)) =
                    (lhs_view, rhs_view)
                else {
                    return None;
                };
                let (Some(lhs_slice), Some(rhs_slice)) = (
                    lhs_view.as_slice_memory_order(),
                    rhs_view.as_slice_memory_order(),
                ) else {
                    return None;
                };
                let mut lhs_grad = vec![0.0f32; lhs_shape.iter().product()];
                let mut rhs_grad = vec![0.0f32; rhs_shape.iter().product()];
                let mut tmp = vec![0.0f32; row_len];
                match side {
                    RowBroadcastSide::RhsVector => {
                        for row in 0..rows {
                            let start = row * row_len;
                            let g_row = &grad_slice[start..start + row_len];
                            mul_f32_bf16_slice_to_f32(
                                g_row,
                                rhs_slice,
                                &mut lhs_grad[start..start + row_len],
                            );
                            mul_f32_bf16_slice_to_f32(
                                g_row,
                                &lhs_slice[start..start + row_len],
                                &mut tmp,
                            );
                            for (dst, &v) in rhs_grad.iter_mut().zip(tmp.iter()) {
                                *dst += v;
                            }
                        }
                    }
                    RowBroadcastSide::LhsVector => {
                        for row in 0..rows {
                            let start = row * row_len;
                            let g_row = &grad_slice[start..start + row_len];
                            mul_f32_bf16_slice_to_f32(
                                g_row,
                                &rhs_slice[start..start + row_len],
                                &mut tmp,
                            );
                            for (dst, &v) in lhs_grad.iter_mut().zip(tmp.iter()) {
                                *dst += v;
                            }
                            mul_f32_bf16_slice_to_f32(
                                g_row,
                                lhs_slice,
                                &mut rhs_grad[start..start + row_len],
                            );
                        }
                    }
                }
                Some((
                    array_from_shape_vec(&lhs_shape, lhs_grad)?,
                    array_from_shape_vec(&rhs_shape, rhs_grad)?,
                    lhs_shape,
                    rhs_shape,
                ))
            })
        }),
        DType::I8 => {
            let (
                TensorStorageOwned::I8(lhs_data, lhs_scale),
                TensorStorageOwned::I8(rhs_data, rhs_scale),
            ) = (lhs.native_storage_owned(), rhs.native_storage_owned())
            else {
                return None;
            };
            let (Some(lhs_slice), Some(rhs_slice)) = (
                lhs_data.as_slice_memory_order(),
                rhs_data.as_slice_memory_order(),
            ) else {
                return None;
            };
            let mut lhs_grad = vec![0.0f32; lhs_shape.iter().product()];
            let mut rhs_grad = vec![0.0f32; rhs_shape.iter().product()];
            let mut tmp = vec![0.0f32; row_len];
            match side {
                RowBroadcastSide::RhsVector => {
                    for row in 0..rows {
                        let start = row * row_len;
                        let g_row = &grad_slice[start..start + row_len];
                        mul_f32_i8_slice_to_f32(
                            g_row,
                            rhs_slice,
                            rhs_scale,
                            &mut lhs_grad[start..start + row_len],
                        );
                        mul_f32_i8_slice_to_f32(
                            g_row,
                            &lhs_slice[start..start + row_len],
                            lhs_scale,
                            &mut tmp,
                        );
                        for (dst, &v) in rhs_grad.iter_mut().zip(tmp.iter()) {
                            *dst += v;
                        }
                    }
                }
                RowBroadcastSide::LhsVector => {
                    for row in 0..rows {
                        let start = row * row_len;
                        let g_row = &grad_slice[start..start + row_len];
                        mul_f32_i8_slice_to_f32(
                            g_row,
                            &rhs_slice[start..start + row_len],
                            rhs_scale,
                            &mut tmp,
                        );
                        for (dst, &v) in lhs_grad.iter_mut().zip(tmp.iter()) {
                            *dst += v;
                        }
                        mul_f32_i8_slice_to_f32(
                            g_row,
                            lhs_slice,
                            lhs_scale,
                            &mut rhs_grad[start..start + row_len],
                        );
                    }
                }
            }
            Some((
                array_from_shape_vec(&lhs_shape, lhs_grad)?,
                array_from_shape_vec(&rhs_shape, rhs_grad)?,
                lhs_shape,
                rhs_shape,
            ))
        }
        _ => None,
    }
}

fn try_mixed_row_broadcast_mul_grads_native(
    lhs: &Tensor,
    rhs: &Tensor,
    grad: ArrayViewD<'_, f32>,
) -> Option<NativeMulGradResult> {
    if lhs.device() != Device::Cpu || rhs.device() != Device::Cpu || lhs.dtype() == rhs.dtype() {
        return None;
    }
    let lhs_shape = lhs.shape_vec();
    let rhs_shape = rhs.shape_vec();
    let side = row_broadcast_side(&lhs_shape, &rhs_shape)?;
    let out_shape = match side {
        RowBroadcastSide::RhsVector => lhs_shape.clone(),
        RowBroadcastSide::LhsVector => rhs_shape.clone(),
    };
    if grad.shape() != out_shape.as_slice() {
        return None;
    }
    let row_len = *out_shape.last()?;
    if row_len == 0 {
        return Some((
            ArrayD::zeros(IxDyn(&lhs_shape)),
            ArrayD::zeros(IxDyn(&rhs_shape)),
            lhs_shape,
            rhs_shape,
        ));
    }
    let rows = out_shape.iter().product::<usize>() / row_len;
    let grad_slice = grad.as_slice_memory_order()?;
    let mut lhs_grad = vec![0.0f32; lhs_shape.iter().product()];
    let mut rhs_grad = vec![0.0f32; rhs_shape.iter().product()];
    let mut tmp = vec![0.0f32; row_len];

    match (lhs.native_storage_owned(), rhs.native_storage_owned()) {
        (TensorStorageOwned::F32(lhs_data), TensorStorageOwned::F16(rhs_data)) => {
            let lhs_slice = lhs_data.as_slice_memory_order()?;
            let rhs_slice = rhs_data.as_slice_memory_order()?;
            match side {
                RowBroadcastSide::RhsVector => {
                    for row in 0..rows {
                        let start = row * row_len;
                        let g_row = &grad_slice[start..start + row_len];
                        mul_f32_f16_slice_to_f32(
                            g_row,
                            rhs_slice,
                            &mut lhs_grad[start..start + row_len],
                        );
                        mul_f32_f32_slice_to_f32(
                            g_row,
                            &lhs_slice[start..start + row_len],
                            &mut tmp,
                        );
                        accumulate_slice(&mut rhs_grad, &tmp);
                    }
                }
                RowBroadcastSide::LhsVector => {
                    for row in 0..rows {
                        let start = row * row_len;
                        let g_row = &grad_slice[start..start + row_len];
                        mul_f32_f16_slice_to_f32(
                            g_row,
                            &rhs_slice[start..start + row_len],
                            &mut tmp,
                        );
                        accumulate_slice(&mut lhs_grad, &tmp);
                        mul_f32_f32_slice_to_f32(
                            g_row,
                            lhs_slice,
                            &mut rhs_grad[start..start + row_len],
                        );
                    }
                }
            }
        }
        (TensorStorageOwned::F16(lhs_data), TensorStorageOwned::F32(rhs_data)) => {
            let lhs_slice = lhs_data.as_slice_memory_order()?;
            let rhs_slice = rhs_data.as_slice_memory_order()?;
            match side {
                RowBroadcastSide::RhsVector => {
                    for row in 0..rows {
                        let start = row * row_len;
                        let g_row = &grad_slice[start..start + row_len];
                        mul_f32_f32_slice_to_f32(
                            g_row,
                            rhs_slice,
                            &mut lhs_grad[start..start + row_len],
                        );
                        mul_f32_f16_slice_to_f32(
                            g_row,
                            &lhs_slice[start..start + row_len],
                            &mut tmp,
                        );
                        accumulate_slice(&mut rhs_grad, &tmp);
                    }
                }
                RowBroadcastSide::LhsVector => {
                    for row in 0..rows {
                        let start = row * row_len;
                        let g_row = &grad_slice[start..start + row_len];
                        mul_f32_f32_slice_to_f32(
                            g_row,
                            &rhs_slice[start..start + row_len],
                            &mut tmp,
                        );
                        accumulate_slice(&mut lhs_grad, &tmp);
                        mul_f32_f16_slice_to_f32(
                            g_row,
                            lhs_slice,
                            &mut rhs_grad[start..start + row_len],
                        );
                    }
                }
            }
        }
        (TensorStorageOwned::F32(lhs_data), TensorStorageOwned::BF16(rhs_data)) => {
            let lhs_slice = lhs_data.as_slice_memory_order()?;
            let rhs_slice = rhs_data.as_slice_memory_order()?;
            match side {
                RowBroadcastSide::RhsVector => {
                    for row in 0..rows {
                        let start = row * row_len;
                        let g_row = &grad_slice[start..start + row_len];
                        mul_f32_bf16_slice_to_f32(
                            g_row,
                            rhs_slice,
                            &mut lhs_grad[start..start + row_len],
                        );
                        mul_f32_f32_slice_to_f32(
                            g_row,
                            &lhs_slice[start..start + row_len],
                            &mut tmp,
                        );
                        accumulate_slice(&mut rhs_grad, &tmp);
                    }
                }
                RowBroadcastSide::LhsVector => {
                    for row in 0..rows {
                        let start = row * row_len;
                        let g_row = &grad_slice[start..start + row_len];
                        mul_f32_bf16_slice_to_f32(
                            g_row,
                            &rhs_slice[start..start + row_len],
                            &mut tmp,
                        );
                        accumulate_slice(&mut lhs_grad, &tmp);
                        mul_f32_f32_slice_to_f32(
                            g_row,
                            lhs_slice,
                            &mut rhs_grad[start..start + row_len],
                        );
                    }
                }
            }
        }
        (TensorStorageOwned::BF16(lhs_data), TensorStorageOwned::F32(rhs_data)) => {
            let lhs_slice = lhs_data.as_slice_memory_order()?;
            let rhs_slice = rhs_data.as_slice_memory_order()?;
            match side {
                RowBroadcastSide::RhsVector => {
                    for row in 0..rows {
                        let start = row * row_len;
                        let g_row = &grad_slice[start..start + row_len];
                        mul_f32_f32_slice_to_f32(
                            g_row,
                            rhs_slice,
                            &mut lhs_grad[start..start + row_len],
                        );
                        mul_f32_bf16_slice_to_f32(
                            g_row,
                            &lhs_slice[start..start + row_len],
                            &mut tmp,
                        );
                        accumulate_slice(&mut rhs_grad, &tmp);
                    }
                }
                RowBroadcastSide::LhsVector => {
                    for row in 0..rows {
                        let start = row * row_len;
                        let g_row = &grad_slice[start..start + row_len];
                        mul_f32_f32_slice_to_f32(
                            g_row,
                            &rhs_slice[start..start + row_len],
                            &mut tmp,
                        );
                        accumulate_slice(&mut lhs_grad, &tmp);
                        mul_f32_bf16_slice_to_f32(
                            g_row,
                            lhs_slice,
                            &mut rhs_grad[start..start + row_len],
                        );
                    }
                }
            }
        }
        (TensorStorageOwned::F32(lhs_data), TensorStorageOwned::I8(rhs_data, rhs_scale)) => {
            let lhs_slice = lhs_data.as_slice_memory_order()?;
            let rhs_slice = rhs_data.as_slice_memory_order()?;
            match side {
                RowBroadcastSide::RhsVector => {
                    for row in 0..rows {
                        let start = row * row_len;
                        let g_row = &grad_slice[start..start + row_len];
                        mul_f32_i8_slice_to_f32(
                            g_row,
                            rhs_slice,
                            rhs_scale,
                            &mut lhs_grad[start..start + row_len],
                        );
                        mul_f32_f32_slice_to_f32(
                            g_row,
                            &lhs_slice[start..start + row_len],
                            &mut tmp,
                        );
                        accumulate_slice(&mut rhs_grad, &tmp);
                    }
                }
                RowBroadcastSide::LhsVector => {
                    for row in 0..rows {
                        let start = row * row_len;
                        let g_row = &grad_slice[start..start + row_len];
                        mul_f32_i8_slice_to_f32(
                            g_row,
                            &rhs_slice[start..start + row_len],
                            rhs_scale,
                            &mut tmp,
                        );
                        accumulate_slice(&mut lhs_grad, &tmp);
                        mul_f32_f32_slice_to_f32(
                            g_row,
                            lhs_slice,
                            &mut rhs_grad[start..start + row_len],
                        );
                    }
                }
            }
        }
        (TensorStorageOwned::I8(lhs_data, lhs_scale), TensorStorageOwned::F32(rhs_data)) => {
            let lhs_slice = lhs_data.as_slice_memory_order()?;
            let rhs_slice = rhs_data.as_slice_memory_order()?;
            match side {
                RowBroadcastSide::RhsVector => {
                    for row in 0..rows {
                        let start = row * row_len;
                        let g_row = &grad_slice[start..start + row_len];
                        mul_f32_f32_slice_to_f32(
                            g_row,
                            rhs_slice,
                            &mut lhs_grad[start..start + row_len],
                        );
                        mul_f32_i8_slice_to_f32(
                            g_row,
                            &lhs_slice[start..start + row_len],
                            lhs_scale,
                            &mut tmp,
                        );
                        accumulate_slice(&mut rhs_grad, &tmp);
                    }
                }
                RowBroadcastSide::LhsVector => {
                    for row in 0..rows {
                        let start = row * row_len;
                        let g_row = &grad_slice[start..start + row_len];
                        mul_f32_f32_slice_to_f32(
                            g_row,
                            &rhs_slice[start..start + row_len],
                            &mut tmp,
                        );
                        accumulate_slice(&mut lhs_grad, &tmp);
                        mul_f32_i8_slice_to_f32(
                            g_row,
                            lhs_slice,
                            lhs_scale,
                            &mut rhs_grad[start..start + row_len],
                        );
                    }
                }
            }
        }
        _ => return None,
    }

    Some((
        array_from_shape_vec(&lhs_shape, lhs_grad)?,
        array_from_shape_vec(&rhs_shape, rhs_grad)?,
        lhs_shape,
        rhs_shape,
    ))
}

fn try_mixed_row_scalar_mul_grads_native(
    lhs: &Tensor,
    rhs: &Tensor,
    grad: ArrayViewD<'_, f32>,
) -> Option<NativeMulGradResult> {
    if lhs.device() != Device::Cpu || rhs.device() != Device::Cpu || lhs.dtype() == rhs.dtype() {
        return None;
    }
    let lhs_shape = lhs.shape_vec();
    let rhs_shape = rhs.shape_vec();
    let (rows, last_dim, scalar_on_rhs) = row_scalar_broadcast_side(&lhs_shape, &rhs_shape)?;
    let out_shape = if scalar_on_rhs {
        lhs_shape.clone()
    } else {
        rhs_shape.clone()
    };
    if grad.shape() != out_shape.as_slice() {
        return None;
    }
    let grad_slice = grad.as_slice_memory_order()?;
    let mut lhs_grad = vec![0.0f32; lhs.len()];
    let mut rhs_grad = vec![0.0f32; rhs.len()];

    match (lhs.native_storage_owned(), rhs.native_storage_owned()) {
        (TensorStorageOwned::F32(lhs_data), TensorStorageOwned::F16(rhs_data)) => {
            let lhs_slice = lhs_data.as_slice_memory_order()?;
            let rhs_slice = rhs_data.as_slice_memory_order()?;
            for row in 0..rows {
                let start = row * last_dim;
                let g_row = &grad_slice[start..start + last_dim];
                if scalar_on_rhs {
                    mul_f32_slice_by_scalar_to_f32(
                        g_row,
                        rhs_slice[row].to_f32(),
                        &mut lhs_grad[start..start + last_dim],
                    );
                    rhs_grad[row] = dot_f32_f32_slice(g_row, &lhs_slice[start..start + last_dim]);
                } else {
                    lhs_grad[row] = dot_f32_f16_slice(g_row, &rhs_slice[start..start + last_dim]);
                    mul_f32_slice_by_scalar_to_f32(
                        g_row,
                        lhs_slice[row],
                        &mut rhs_grad[start..start + last_dim],
                    );
                }
            }
        }
        (TensorStorageOwned::F16(lhs_data), TensorStorageOwned::F32(rhs_data)) => {
            let lhs_slice = lhs_data.as_slice_memory_order()?;
            let rhs_slice = rhs_data.as_slice_memory_order()?;
            for row in 0..rows {
                let start = row * last_dim;
                let g_row = &grad_slice[start..start + last_dim];
                if scalar_on_rhs {
                    mul_f32_slice_by_scalar_to_f32(
                        g_row,
                        rhs_slice[row],
                        &mut lhs_grad[start..start + last_dim],
                    );
                    rhs_grad[row] = dot_f32_f16_slice(g_row, &lhs_slice[start..start + last_dim]);
                } else {
                    lhs_grad[row] = dot_f32_f32_slice(g_row, &rhs_slice[start..start + last_dim]);
                    mul_f32_slice_by_scalar_to_f32(
                        g_row,
                        lhs_slice[row].to_f32(),
                        &mut rhs_grad[start..start + last_dim],
                    );
                }
            }
        }
        (TensorStorageOwned::F32(lhs_data), TensorStorageOwned::BF16(rhs_data)) => {
            let lhs_slice = lhs_data.as_slice_memory_order()?;
            let rhs_slice = rhs_data.as_slice_memory_order()?;
            for row in 0..rows {
                let start = row * last_dim;
                let g_row = &grad_slice[start..start + last_dim];
                if scalar_on_rhs {
                    mul_f32_slice_by_scalar_to_f32(
                        g_row,
                        rhs_slice[row].to_f32(),
                        &mut lhs_grad[start..start + last_dim],
                    );
                    rhs_grad[row] = dot_f32_f32_slice(g_row, &lhs_slice[start..start + last_dim]);
                } else {
                    lhs_grad[row] = dot_f32_bf16_slice(g_row, &rhs_slice[start..start + last_dim]);
                    mul_f32_slice_by_scalar_to_f32(
                        g_row,
                        lhs_slice[row],
                        &mut rhs_grad[start..start + last_dim],
                    );
                }
            }
        }
        (TensorStorageOwned::BF16(lhs_data), TensorStorageOwned::F32(rhs_data)) => {
            let lhs_slice = lhs_data.as_slice_memory_order()?;
            let rhs_slice = rhs_data.as_slice_memory_order()?;
            for row in 0..rows {
                let start = row * last_dim;
                let g_row = &grad_slice[start..start + last_dim];
                if scalar_on_rhs {
                    mul_f32_slice_by_scalar_to_f32(
                        g_row,
                        rhs_slice[row],
                        &mut lhs_grad[start..start + last_dim],
                    );
                    rhs_grad[row] = dot_f32_bf16_slice(g_row, &lhs_slice[start..start + last_dim]);
                } else {
                    lhs_grad[row] = dot_f32_f32_slice(g_row, &rhs_slice[start..start + last_dim]);
                    mul_f32_slice_by_scalar_to_f32(
                        g_row,
                        lhs_slice[row].to_f32(),
                        &mut rhs_grad[start..start + last_dim],
                    );
                }
            }
        }
        (TensorStorageOwned::F32(lhs_data), TensorStorageOwned::I8(rhs_data, rhs_scale)) => {
            let lhs_slice = lhs_data.as_slice_memory_order()?;
            let rhs_slice = rhs_data.as_slice_memory_order()?;
            for row in 0..rows {
                let start = row * last_dim;
                let g_row = &grad_slice[start..start + last_dim];
                if scalar_on_rhs {
                    mul_f32_slice_by_scalar_to_f32(
                        g_row,
                        (rhs_slice[row] as f32) * rhs_scale,
                        &mut lhs_grad[start..start + last_dim],
                    );
                    rhs_grad[row] = dot_f32_f32_slice(g_row, &lhs_slice[start..start + last_dim]);
                } else {
                    lhs_grad[row] =
                        dot_f32_i8_slice(g_row, &rhs_slice[start..start + last_dim], rhs_scale);
                    mul_f32_slice_by_scalar_to_f32(
                        g_row,
                        lhs_slice[row],
                        &mut rhs_grad[start..start + last_dim],
                    );
                }
            }
        }
        (TensorStorageOwned::I8(lhs_data, lhs_scale), TensorStorageOwned::F32(rhs_data)) => {
            let lhs_slice = lhs_data.as_slice_memory_order()?;
            let rhs_slice = rhs_data.as_slice_memory_order()?;
            for row in 0..rows {
                let start = row * last_dim;
                let g_row = &grad_slice[start..start + last_dim];
                if scalar_on_rhs {
                    mul_f32_slice_by_scalar_to_f32(
                        g_row,
                        rhs_slice[row],
                        &mut lhs_grad[start..start + last_dim],
                    );
                    rhs_grad[row] =
                        dot_f32_i8_slice(g_row, &lhs_slice[start..start + last_dim], lhs_scale);
                } else {
                    lhs_grad[row] = dot_f32_f32_slice(g_row, &rhs_slice[start..start + last_dim]);
                    mul_f32_slice_by_scalar_to_f32(
                        g_row,
                        (lhs_slice[row] as f32) * lhs_scale,
                        &mut rhs_grad[start..start + last_dim],
                    );
                }
            }
        }
        _ => return None,
    }

    Some((
        array_from_shape_vec(&lhs_shape, lhs_grad)?,
        array_from_shape_vec(&rhs_shape, rhs_grad)?,
        lhs_shape,
        rhs_shape,
    ))
}

fn try_mixed_scalar_mul_grads_native(
    lhs: &Tensor,
    rhs: &Tensor,
    grad: ArrayViewD<'_, f32>,
) -> Option<NativeMulGradResult> {
    if lhs.device() != Device::Cpu || rhs.device() != Device::Cpu || lhs.dtype() == rhs.dtype() {
        return None;
    }
    let lhs_shape = lhs.shape_vec();
    let rhs_shape = rhs.shape_vec();
    let out_shape = broadcast_shape(&lhs_shape, &rhs_shape)?;
    let out_len = out_shape.iter().product::<usize>();
    if out_len == 0
        || grad.len() != out_len
        || !((lhs.len() == 1 && rhs.len() == out_len) || (rhs.len() == 1 && lhs.len() == out_len))
    {
        return None;
    }
    let grad_slice = grad.as_slice_memory_order()?;
    let mut lhs_grad = vec![0.0f32; lhs.len()];
    let mut rhs_grad = vec![0.0f32; rhs.len()];

    match (lhs.native_storage_owned(), rhs.native_storage_owned()) {
        (TensorStorageOwned::F32(lhs_data), TensorStorageOwned::F16(rhs_data)) => {
            let lhs_slice = lhs_data.as_slice_memory_order()?;
            let rhs_slice = rhs_data.as_slice_memory_order()?;
            if lhs_slice.len() == 1 {
                lhs_grad[0] = dot_f32_f16_slice(grad_slice, rhs_slice);
                mul_f32_slice_by_scalar_to_f32(grad_slice, lhs_slice[0], &mut rhs_grad);
            } else {
                mul_f32_slice_by_scalar_to_f32(grad_slice, rhs_slice[0].to_f32(), &mut lhs_grad);
                rhs_grad[0] = dot_f32_f32_slice(grad_slice, lhs_slice);
            }
        }
        (TensorStorageOwned::F16(lhs_data), TensorStorageOwned::F32(rhs_data)) => {
            let lhs_slice = lhs_data.as_slice_memory_order()?;
            let rhs_slice = rhs_data.as_slice_memory_order()?;
            if lhs_slice.len() == 1 {
                lhs_grad[0] = dot_f32_f32_slice(grad_slice, rhs_slice);
                mul_f32_slice_by_scalar_to_f32(grad_slice, lhs_slice[0].to_f32(), &mut rhs_grad);
            } else {
                mul_f32_slice_by_scalar_to_f32(grad_slice, rhs_slice[0], &mut lhs_grad);
                rhs_grad[0] = dot_f32_f16_slice(grad_slice, lhs_slice);
            }
        }
        (TensorStorageOwned::F32(lhs_data), TensorStorageOwned::BF16(rhs_data)) => {
            let lhs_slice = lhs_data.as_slice_memory_order()?;
            let rhs_slice = rhs_data.as_slice_memory_order()?;
            if lhs_slice.len() == 1 {
                lhs_grad[0] = dot_f32_bf16_slice(grad_slice, rhs_slice);
                mul_f32_slice_by_scalar_to_f32(grad_slice, lhs_slice[0], &mut rhs_grad);
            } else {
                mul_f32_slice_by_scalar_to_f32(grad_slice, rhs_slice[0].to_f32(), &mut lhs_grad);
                rhs_grad[0] = dot_f32_f32_slice(grad_slice, lhs_slice);
            }
        }
        (TensorStorageOwned::BF16(lhs_data), TensorStorageOwned::F32(rhs_data)) => {
            let lhs_slice = lhs_data.as_slice_memory_order()?;
            let rhs_slice = rhs_data.as_slice_memory_order()?;
            if lhs_slice.len() == 1 {
                lhs_grad[0] = dot_f32_f32_slice(grad_slice, rhs_slice);
                mul_f32_slice_by_scalar_to_f32(grad_slice, lhs_slice[0].to_f32(), &mut rhs_grad);
            } else {
                mul_f32_slice_by_scalar_to_f32(grad_slice, rhs_slice[0], &mut lhs_grad);
                rhs_grad[0] = dot_f32_bf16_slice(grad_slice, lhs_slice);
            }
        }
        (TensorStorageOwned::F32(lhs_data), TensorStorageOwned::I8(rhs_data, rhs_scale)) => {
            let lhs_slice = lhs_data.as_slice_memory_order()?;
            let rhs_slice = rhs_data.as_slice_memory_order()?;
            if lhs_slice.len() == 1 {
                lhs_grad[0] = dot_f32_i8_slice(grad_slice, rhs_slice, rhs_scale);
                mul_f32_slice_by_scalar_to_f32(grad_slice, lhs_slice[0], &mut rhs_grad);
            } else {
                mul_f32_slice_by_scalar_to_f32(
                    grad_slice,
                    (rhs_slice[0] as f32) * rhs_scale,
                    &mut lhs_grad,
                );
                rhs_grad[0] = dot_f32_f32_slice(grad_slice, lhs_slice);
            }
        }
        (TensorStorageOwned::I8(lhs_data, lhs_scale), TensorStorageOwned::F32(rhs_data)) => {
            let lhs_slice = lhs_data.as_slice_memory_order()?;
            let rhs_slice = rhs_data.as_slice_memory_order()?;
            if lhs_slice.len() == 1 {
                lhs_grad[0] = dot_f32_f32_slice(grad_slice, rhs_slice);
                mul_f32_slice_by_scalar_to_f32(
                    grad_slice,
                    (lhs_slice[0] as f32) * lhs_scale,
                    &mut rhs_grad,
                );
            } else {
                mul_f32_slice_by_scalar_to_f32(grad_slice, rhs_slice[0], &mut lhs_grad);
                rhs_grad[0] = dot_f32_i8_slice(grad_slice, lhs_slice, lhs_scale);
            }
        }
        _ => return None,
    }

    Some((
        array_from_shape_vec(&lhs_shape, lhs_grad)?,
        array_from_shape_vec(&rhs_shape, rhs_grad)?,
        lhs_shape,
        rhs_shape,
    ))
}

fn add_cpu_binary_grads(lhs: &Tensor, rhs: &Tensor, grad: ArrayViewD<'_, f32>, op: BinaryOp) {
    let l_shape = lhs.shape_vec();
    let r_shape = rhs.shape_vec();
    match op {
        BinaryOp::Add => {
            lhs.add_grad(reduce_gradient(grad.view(), &l_shape));
            rhs.add_grad(reduce_gradient(grad.view(), &r_shape));
        }
        BinaryOp::Sub => {
            lhs.add_grad(reduce_gradient(grad.view(), &l_shape));
            let grad_neg = Zip::from(&grad).par_map_collect(|&x| -x);
            rhs.add_grad(reduce_gradient(grad_neg.view(), &r_shape));
        }
        BinaryOp::Mul => {
            let (g_lhs, g_rhs, lhs_shape, rhs_shape) = {
                if let Some(native_grads) = try_same_shape_mul_grads_native(lhs, rhs, grad.view())
                    .or_else(|| try_row_broadcast_mul_grads_native(lhs, rhs, grad.view()))
                    .or_else(|| try_mixed_row_broadcast_mul_grads_native(lhs, rhs, grad.view()))
                    .or_else(|| try_mixed_row_scalar_mul_grads_native(lhs, rhs, grad.view()))
                    .or_else(|| try_mixed_scalar_mul_grads_native(lhs, rhs, grad.view()))
                {
                    lhs.add_grad(reduce_gradient(native_grads.0.view(), &native_grads.2));
                    rhs.add_grad(reduce_gradient(native_grads.1.view(), &native_grads.3));
                    return;
                }

                let lhs_data = lhs.data_ref();
                let rhs_data = rhs.data_ref();

                let (g_lhs, g_rhs) =
                    if grad.shape() == lhs_data.shape() && grad.shape() == rhs_data.shape() {
                        let gl = Zip::from(&grad)
                            .and(&*rhs_data)
                            .par_map_collect(|&g, &b| g * b);
                        let gr = Zip::from(&grad)
                            .and(&*lhs_data)
                            .par_map_collect(|&g, &a| g * a);
                        (gl, gr)
                    } else {
                        (grad.to_owned() * &*rhs_data, grad.to_owned() * &*lhs_data)
                    };

                (
                    g_lhs,
                    g_rhs,
                    lhs_data.shape().to_vec(),
                    rhs_data.shape().to_vec(),
                )
            };
            lhs.add_grad(reduce_gradient(g_lhs.view(), &lhs_shape));
            rhs.add_grad(reduce_gradient(g_rhs.view(), &rhs_shape));
        }
    }
}

fn try_cuda_binary_training(
    lhs: &Tensor,
    rhs: &Tensor,
    op: BinaryOp,
    op_name: &'static str,
) -> Option<Tensor> {
    if lhs.device() != Device::Cuda {
        return None;
    }

    let lhs_shape = lhs.shape_vec();
    let rhs_shape = rhs.shape_vec();
    let out_shape = broadcast_shape(&lhs_shape, &rhs_shape)?;
    if out_shape.iter().product::<usize>() == 0 {
        return None;
    }
    let cuda_op = cuda_binary_op(op);
    let cuda_out = if let Some(buffer) = try_cuda_binary_native_same_shape_f32_buffer(lhs, rhs, op)
        .or_else(|| try_cuda_binary_native_row_broadcast_f32_buffer(lhs, rhs, op))
        .or_else(|| try_cuda_binary_typed_same_shape_f32_buffer(lhs, rhs, op))
        .or_else(|| try_cuda_binary_typed_row_broadcast_f32_buffer(lhs, rhs, op))
        .or_else(|| try_cuda_binary_typed_row_scalar_f32_buffer(lhs, rhs, op))
        .or_else(|| try_cuda_binary_typed_b1d_1h1_f32_buffer(lhs, rhs, op))
        .or_else(|| try_cuda_binary_typed_b1d_1hd_f32_buffer(lhs, rhs, op))
        .or_else(|| try_cuda_binary_typed_broadcast_f32_buffer(lhs, rhs, op))
    {
        Ok(buffer)
    } else if lhs_shape == rhs_shape {
        lhs.with_cuda_f32_buffer(|lhs_buf| {
            rhs.with_cuda_f32_buffer(|rhs_buf| cuda::binary_f32_buffer(lhs_buf, rhs_buf, cuda_op))
        })
    } else {
        lhs.with_cuda_f32_buffer(|lhs_buf| {
            rhs.with_cuda_f32_buffer(|rhs_buf| {
                cuda::binary_broadcast_f32_buffer(
                    lhs_buf, rhs_buf, &lhs_shape, &rhs_shape, &out_shape, cuda_op,
                )
            })
        })
    };

    match cuda_out {
        Ok(buffer) => {
            let lhs_clone = lhs.clone();
            let rhs_clone = rhs.clone();
            let lhs_shape_for_backward = lhs_shape.clone();
            let rhs_shape_for_backward = rhs_shape.clone();
            let out_shape_for_backward = out_shape.clone();
            let output_self = Rc::new(RefCell::new(None::<Tensor>));
            let output_self_for_backward = output_self.clone();

            let tensor = Tensor(Rc::new(RefCell::new(TensorData {
                data: ndarray::ArrayD::<f32>::zeros(IxDyn(&out_shape)).into_shared(),
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
                parents: vec![lhs.clone(), rhs.clone()],
                backward_op: Some(std::rc::Rc::new(move |grad| {
                    let upstream_cuda_grad = output_self_for_backward
                        .borrow()
                        .as_ref()
                        .and_then(|output| output.cloned_cuda_f32_grad())
                        .filter(|grad_buf| grad_buf.len() == grad.len());
                    if is_strict_device_execution() {
                        let grad_buf = match upstream_cuda_grad.clone() {
                            Some(buffer) => Ok(buffer),
                            None => {
                                let grad_host = grad.iter().copied().collect::<Vec<_>>();
                                cuda::upload_f32(&grad_host)
                            }
                        };
                        let cuda_grad_buffers = grad_buf.and_then(|grad_buf| {
                            if let Some(shape_only_grads) =
                                try_cuda_add_sub_grads_shape_only_buffers(
                                    &lhs_clone, &rhs_clone, &grad_buf, op,
                                )
                            {
                                return shape_only_grads;
                            }

                            if matches!(op, BinaryOp::Mul)
                                && let Some(native_grads) =
                                    try_cuda_same_shape_mul_grads_typed_buffers(
                                        &lhs_clone, &rhs_clone, &grad_buf,
                                    )
                                    .or_else(|| {
                                        try_cuda_same_shape_mul_grads_native_lowp_buffers(
                                            &lhs_clone, &rhs_clone, &grad_buf,
                                        )
                                    })
                                    .or_else(|| {
                                        try_cuda_row_broadcast_mul_grads_native_lowp_buffers(
                                            &lhs_clone, &rhs_clone, &grad_buf,
                                        )
                                    })
                                    .or_else(|| {
                                        try_cuda_row_broadcast_mul_grads_typed_buffers(
                                            &lhs_clone, &rhs_clone, &grad_buf,
                                        )
                                    })
                                    .or_else(|| {
                                        try_cuda_row_scalar_mul_grads_typed_buffers(
                                            &lhs_clone, &rhs_clone, &grad_buf,
                                        )
                                    })
                                    .or_else(|| {
                                        try_cuda_scalar_mul_grads_typed_buffers(
                                            &lhs_clone, &rhs_clone, &grad_buf,
                                        )
                                    })
                                    .or_else(|| {
                                        try_cuda_b1d_1hd_mul_grads_typed_buffers(
                                            &lhs_clone, &rhs_clone, &grad_buf,
                                        )
                                    })
                                    .or_else(|| {
                                        try_cuda_b1d_1h1_mul_grads_typed_buffers(
                                            &lhs_clone, &rhs_clone, &grad_buf,
                                        )
                                    })
                                    .or_else(|| {
                                        try_cuda_broadcast_mul_grads_typed_buffers(
                                            &lhs_clone, &rhs_clone, &grad_buf,
                                        )
                                    })
                            {
                                return native_grads;
                            }

                            lhs_clone.with_cuda_f32_buffer(|lhs_buf| {
                                rhs_clone.with_cuda_f32_buffer(|rhs_buf| {
                                    if lhs_shape_for_backward == rhs_shape_for_backward {
                                        cuda::binary_backward_f32_buffers(
                                            lhs_buf, rhs_buf, &grad_buf, cuda_op,
                                        )
                                    } else {
                                        cuda::binary_broadcast_backward_f32_buffers(
                                            lhs_buf,
                                            rhs_buf,
                                            &grad_buf,
                                            &lhs_shape_for_backward,
                                            &rhs_shape_for_backward,
                                            &out_shape_for_backward,
                                            cuda_op,
                                        )
                                    }
                                })
                            })
                        });

                        match cuda_grad_buffers {
                            Ok((lhs_buffer, rhs_buffer)) => {
                                lhs_clone.add_cuda_grad_buffer_only(lhs_buffer);
                                rhs_clone.add_cuda_grad_buffer_only(rhs_buffer);
                                return;
                            }
                            Err(err) => {
                                panic!(
                                    "{op_name} CUDA backward failed while strict device execution is enabled: {err}"
                                );
                            }
                        }
                    }
                    let cuda_grad = if let Some(grad_buf) = upstream_cuda_grad {
                        if let Some(shape_only_grads) = try_cuda_add_sub_grads_shape_only_buffers(
                            &lhs_clone, &rhs_clone, &grad_buf, op,
                        ) {
                            shape_only_grads.and_then(|(lhs_buffer, rhs_buffer)| {
                                let lhs_host = cuda::download_f32(&lhs_buffer)?;
                                let rhs_host = cuda::download_f32(&rhs_buffer)?;
                                Ok(((lhs_buffer, lhs_host), (rhs_buffer, rhs_host)))
                            })
                        } else if matches!(op, BinaryOp::Mul)
                            && let Some(native_grads) = try_cuda_same_shape_mul_grads_typed_buffers(
                                &lhs_clone, &rhs_clone, &grad_buf,
                            )
                            .or_else(|| {
                                try_cuda_same_shape_mul_grads_native_lowp_buffers(
                                    &lhs_clone, &rhs_clone, &grad_buf,
                                )
                            })
                            .or_else(|| {
                                try_cuda_row_broadcast_mul_grads_native_lowp_buffers(
                                    &lhs_clone, &rhs_clone, &grad_buf,
                                )
                            })
                            .or_else(|| {
                                try_cuda_row_broadcast_mul_grads_typed_buffers(
                                    &lhs_clone, &rhs_clone, &grad_buf,
                                )
                            })
                            .or_else(|| {
                                try_cuda_row_scalar_mul_grads_typed_buffers(
                                    &lhs_clone, &rhs_clone, &grad_buf,
                                )
                            })
                            .or_else(|| {
                                try_cuda_scalar_mul_grads_typed_buffers(
                                    &lhs_clone, &rhs_clone, &grad_buf,
                                )
                            })
                            .or_else(|| {
                                try_cuda_b1d_1hd_mul_grads_typed_buffers(
                                    &lhs_clone, &rhs_clone, &grad_buf,
                                )
                            })
                            .or_else(|| {
                                try_cuda_b1d_1h1_mul_grads_typed_buffers(
                                    &lhs_clone, &rhs_clone, &grad_buf,
                                )
                            })
                            .or_else(|| {
                                try_cuda_broadcast_mul_grads_typed_buffers(
                                    &lhs_clone, &rhs_clone, &grad_buf,
                                )
                            })
                        {
                            native_grads.and_then(|(lhs_buffer, rhs_buffer)| {
                                let lhs_host = cuda::download_f32(&lhs_buffer)?;
                                let rhs_host = cuda::download_f32(&rhs_buffer)?;
                                Ok(((lhs_buffer, lhs_host), (rhs_buffer, rhs_host)))
                            })
                        } else {
                            lhs_clone.with_cuda_f32_buffer(|lhs_buf| {
                                rhs_clone.with_cuda_f32_buffer(|rhs_buf| {
                                    if lhs_shape_for_backward == rhs_shape_for_backward {
                                        cuda::binary_backward_f32(
                                            lhs_buf, rhs_buf, &grad_buf, cuda_op,
                                        )
                                    } else {
                                        cuda::binary_broadcast_backward_f32(
                                            lhs_buf,
                                            rhs_buf,
                                            &grad_buf,
                                            &lhs_shape_for_backward,
                                            &rhs_shape_for_backward,
                                            &out_shape_for_backward,
                                            cuda_op,
                                        )
                                    }
                                })
                            })
                        }
                    } else {
                        let grad_host = grad.iter().copied().collect::<Vec<_>>();
                        cuda::upload_f32(&grad_host).and_then(|grad_buf| {
                            if let Some(shape_only_grads) =
                                try_cuda_add_sub_grads_shape_only_buffers(
                                    &lhs_clone, &rhs_clone, &grad_buf, op,
                                )
                            {
                                return shape_only_grads.and_then(|(lhs_buffer, rhs_buffer)| {
                                    let lhs_host = cuda::download_f32(&lhs_buffer)?;
                                    let rhs_host = cuda::download_f32(&rhs_buffer)?;
                                    Ok(((lhs_buffer, lhs_host), (rhs_buffer, rhs_host)))
                                });
                            }

                            if matches!(op, BinaryOp::Mul)
                                && let Some(native_grads) =
                                    try_cuda_same_shape_mul_grads_typed_buffers(
                                        &lhs_clone, &rhs_clone, &grad_buf,
                                    )
                                    .or_else(|| {
                                        try_cuda_same_shape_mul_grads_native_lowp_buffers(
                                            &lhs_clone, &rhs_clone, &grad_buf,
                                        )
                                    })
                                    .or_else(|| {
                                        try_cuda_row_broadcast_mul_grads_native_lowp_buffers(
                                            &lhs_clone, &rhs_clone, &grad_buf,
                                        )
                                    })
                                    .or_else(|| {
                                        try_cuda_row_broadcast_mul_grads_typed_buffers(
                                            &lhs_clone, &rhs_clone, &grad_buf,
                                        )
                                    })
                                    .or_else(|| {
                                        try_cuda_row_scalar_mul_grads_typed_buffers(
                                            &lhs_clone, &rhs_clone, &grad_buf,
                                        )
                                    })
                                    .or_else(|| {
                                        try_cuda_scalar_mul_grads_typed_buffers(
                                            &lhs_clone, &rhs_clone, &grad_buf,
                                        )
                                    })
                                    .or_else(|| {
                                        try_cuda_b1d_1hd_mul_grads_typed_buffers(
                                            &lhs_clone, &rhs_clone, &grad_buf,
                                        )
                                    })
                                    .or_else(|| {
                                        try_cuda_b1d_1h1_mul_grads_typed_buffers(
                                            &lhs_clone, &rhs_clone, &grad_buf,
                                        )
                                    })
                                    .or_else(|| {
                                        try_cuda_broadcast_mul_grads_typed_buffers(
                                            &lhs_clone, &rhs_clone, &grad_buf,
                                        )
                                    })
                            {
                                return native_grads.and_then(|(lhs_buffer, rhs_buffer)| {
                                    let lhs_host = cuda::download_f32(&lhs_buffer)?;
                                    let rhs_host = cuda::download_f32(&rhs_buffer)?;
                                    Ok(((lhs_buffer, lhs_host), (rhs_buffer, rhs_host)))
                                });
                            }

                            lhs_clone.with_cuda_f32_buffer(|lhs_buf| {
                                rhs_clone.with_cuda_f32_buffer(|rhs_buf| {
                                    if lhs_shape_for_backward == rhs_shape_for_backward {
                                        cuda::binary_backward_f32(
                                            lhs_buf, rhs_buf, &grad_buf, cuda_op,
                                        )
                                    } else {
                                        cuda::binary_broadcast_backward_f32(
                                            lhs_buf,
                                            rhs_buf,
                                            &grad_buf,
                                            &lhs_shape_for_backward,
                                            &rhs_shape_for_backward,
                                            &out_shape_for_backward,
                                            cuda_op,
                                        )
                                    }
                                })
                            })
                        })
                    };

                    match cuda_grad {
                        Ok(((lhs_buffer, lhs_host), (rhs_buffer, rhs_host))) => {
                            let grad_lhs = ndarray::Array::from_shape_vec(
                                IxDyn(&lhs_shape_for_backward),
                                lhs_host,
                            )
                            .expect("CUDA binary lhs grad shape build failed")
                            .into_dyn();
                            let grad_rhs = ndarray::Array::from_shape_vec(
                                IxDyn(&rhs_shape_for_backward),
                                rhs_host,
                            )
                            .expect("CUDA binary rhs grad shape build failed")
                            .into_dyn();
                            lhs_clone.add_grad_with_cuda_buffer(grad_lhs, Some(lhs_buffer));
                            rhs_clone.add_grad_with_cuda_buffer(grad_rhs, Some(rhs_buffer));
                        }
                        Err(err) => {
                            assert!(
                                !is_strict_device_execution(),
                                "{op_name} CUDA backward failed while strict device execution is enabled: {err}"
                            );
                            add_cpu_binary_grads(&lhs_clone, &rhs_clone, grad.view(), op);
                        }
                    }
                })),
                requires_grad: true,
                device: Device::Cuda,
            })));
            *output_self.borrow_mut() = Some(tensor.clone());
            Some(tensor)
        }
        Err(err) => {
            assert!(
                !is_strict_device_execution(),
                "{op_name} CUDA forward failed while strict device execution is enabled: {err}"
            );
            None
        }
    }
}

impl Add for Tensor {
    type Output = Tensor;
    fn add(self, rhs: Tensor) -> Tensor {
        let output_device = assert_same_device(&self, &rhs, "add");
        let build_graph = !is_no_grad() && (self.requires_grad() || rhs.requires_grad());
        let lhs_shape = self.shape_vec();
        let rhs_shape = rhs.shape_vec();
        let cuda_native_supported =
            output_device == Device::Cuda && broadcast_shape(&lhs_shape, &rhs_shape).is_some();
        assert_native_device_support(output_device, "add", cuda_native_supported);

        if !build_graph {
            return binary_no_grad(&self, &rhs, BinaryOp::Add);
        }

        if output_device == Device::Cuda
            && let Some(output) = try_cuda_binary_training(&self, &rhs, BinaryOp::Add, "add")
        {
            return output;
        }

        let data = if let Some(data) = try_binary_training_native_f32(&self, &rhs, BinaryOp::Add) {
            data
        } else {
            let lhs_data = self.data_ref();
            let rhs_data = rhs.data_ref();
            apply_binary_views(lhs_data.view(), rhs_data.view(), BinaryOp::Add)
        };

        let lhs = self.clone();
        let rhs = rhs.clone();

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
            parents: vec![self.clone(), rhs.clone()],
            backward_op: Some(std::rc::Rc::new(move |grad| {
                add_cpu_binary_grads(&lhs, &rhs, grad.view(), BinaryOp::Add);
            })),
            requires_grad: true,
            device: output_device,
        })))
    }
}
impl<'b> Add<&'b Tensor> for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: &'b Tensor) -> Tensor {
        self.clone() + rhs.clone()
    }
}

impl Sub for Tensor {
    type Output = Tensor;
    fn sub(self, rhs: Tensor) -> Tensor {
        let output_device = assert_same_device(&self, &rhs, "sub");
        let build_graph = !is_no_grad() && (self.requires_grad() || rhs.requires_grad());
        let lhs_shape = self.shape_vec();
        let rhs_shape = rhs.shape_vec();
        let cuda_native_supported =
            output_device == Device::Cuda && broadcast_shape(&lhs_shape, &rhs_shape).is_some();
        assert_native_device_support(output_device, "sub", cuda_native_supported);

        if !build_graph {
            return binary_no_grad(&self, &rhs, BinaryOp::Sub);
        }

        if output_device == Device::Cuda
            && let Some(output) = try_cuda_binary_training(&self, &rhs, BinaryOp::Sub, "sub")
        {
            return output;
        }

        let data = if let Some(data) = try_binary_training_native_f32(&self, &rhs, BinaryOp::Sub) {
            data
        } else {
            let lhs_data = self.data_ref();
            let rhs_data = rhs.data_ref();
            apply_binary_views(lhs_data.view(), rhs_data.view(), BinaryOp::Sub)
        };

        let lhs = self.clone();
        let rhs = rhs.clone();

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
            parents: vec![self.clone(), rhs.clone()],
            backward_op: Some(std::rc::Rc::new(move |grad| {
                add_cpu_binary_grads(&lhs, &rhs, grad.view(), BinaryOp::Sub);
            })),
            requires_grad: true,
            device: output_device,
        })))
    }
}
impl<'b> Sub<&'b Tensor> for &Tensor {
    type Output = Tensor;
    fn sub(self, rhs: &'b Tensor) -> Tensor {
        self.clone() - rhs.clone()
    }
}

impl Mul for Tensor {
    type Output = Tensor;
    fn mul(self, rhs: Tensor) -> Tensor {
        let output_device = assert_same_device(&self, &rhs, "mul");
        let build_graph = !is_no_grad() && (self.requires_grad() || rhs.requires_grad());
        let lhs_shape = self.shape_vec();
        let rhs_shape = rhs.shape_vec();
        let cuda_native_supported =
            output_device == Device::Cuda && broadcast_shape(&lhs_shape, &rhs_shape).is_some();
        assert_native_device_support(output_device, "mul", cuda_native_supported);

        if !build_graph {
            return binary_no_grad(&self, &rhs, BinaryOp::Mul);
        }

        if output_device == Device::Cuda
            && let Some(output) = try_cuda_binary_training(&self, &rhs, BinaryOp::Mul, "mul")
        {
            return output;
        }

        let data = if let Some(data) = try_binary_training_native_f32(&self, &rhs, BinaryOp::Mul) {
            data
        } else {
            let lhs_data = self.data_ref();
            let rhs_data = rhs.data_ref();
            apply_binary_views(lhs_data.view(), rhs_data.view(), BinaryOp::Mul)
        };

        let lhs = self.clone();
        let rhs = rhs.clone();

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
            parents: vec![self.clone(), rhs.clone()],
            backward_op: Some(std::rc::Rc::new(move |grad| {
                add_cpu_binary_grads(&lhs, &rhs, grad.view(), BinaryOp::Mul);
            })),
            requires_grad: true,
            device: output_device,
        })))
    }
}
impl<'b> Mul<&'b Tensor> for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: &'b Tensor) -> Tensor {
        self.clone() * rhs.clone()
    }
}

pub fn sum(input: &Tensor) -> Tensor {
    let output_device = input.device();
    let build_graph = !is_no_grad() && input.requires_grad();
    let cuda_native_supported = output_device == Device::Cuda;
    assert_native_device_support(output_device, "sum", cuda_native_supported);

    if !build_graph {
        if output_device == Device::Cuda && !input.is_empty() {
            let cuda_out = try_cuda_sum_native_lowp(input)
                .unwrap_or_else(|| input.with_cuda_f32_buffer(cuda::sum_f32));
            match cuda_out {
                Ok((buffer, out)) => {
                    let result = ndarray::arr0(out[0]).into_dyn();
                    return Tensor::from_f32_data_no_grad_with_device_dtype_and_cuda_buffer(
                        result,
                        DType::F32,
                        output_device,
                        Some(buffer),
                    );
                }
                Err(err) => {
                    assert!(
                        !is_strict_device_execution(),
                        "sum CUDA forward failed while strict device execution is enabled: {err}"
                    );
                }
            }
        }

        if let Some(sum_val) = try_sum_no_grad_native(input) {
            let result = ndarray::arr0(sum_val).into_dyn();
            return Tensor::from_f32_data_no_grad_with_device_dtype(
                result,
                DType::F32,
                output_device,
            );
        }

        let sum_val =
            input.with_storage_view_preferring(StoragePreference::F32Compute, |view| match view {
                TensorStorageView::F32(view) => view.sum(),
                TensorStorageView::F16(_) => {
                    unreachable!("f32 compute preference should expose f32 view")
                }
                TensorStorageView::BF16(_) => {
                    unreachable!("f32 compute preference should expose f32 view")
                }
            });
        let result = ndarray::arr0(sum_val).into_dyn();
        return Tensor::from_f32_data_no_grad_with_device_dtype(result, DType::F32, output_device);
    }

    if output_device == Device::Cuda && !input.is_empty() {
        let input_shape = input.shape_vec();
        let input_len = input.len();
        let cuda_out = try_cuda_sum_native_lowp(input)
            .unwrap_or_else(|| input.with_cuda_f32_buffer(cuda::sum_f32));
        match cuda_out {
            Ok((buffer, out)) => {
                let result = ndarray::arr0(out[0]).into_dyn();
                let input_clone = input.clone();
                return Tensor(Rc::new(RefCell::new(TensorData {
                    data: result.into_shared(),
                    f16_data: None,
                    bf16_data: None,
                    i8_data: None,
                    cuda_f32_data: Some(buffer),
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
                        let g = grad.first().copied().unwrap_or(0.0);
                        if is_strict_device_execution() {
                            match cuda::fill_scalar_f32_buffer(input_len, g) {
                                Ok(grad_buffer) => {
                                    input_clone.add_cuda_grad_buffer_only(grad_buffer);
                                    return;
                                }
                                Err(err) => {
                                    panic!(
                                        "sum CUDA backward failed while strict device execution is enabled: {err}"
                                    );
                                }
                            }
                        }
                        match cuda::fill_scalar_f32(input_len, g) {
                            Ok((grad_buffer, grad_host)) => {
                                let grad_input =
                                    ndarray::Array::from_shape_vec(input_shape.clone(), grad_host)
                                        .expect("CUDA sum backward grad shape build failed")
                                        .into_dyn();
                                input_clone
                                    .add_grad_with_cuda_buffer(grad_input, Some(grad_buffer));
                            }
                            Err(err) => {
                                assert!(
                                    !is_strict_device_execution(),
                                    "sum CUDA backward failed while strict device execution is enabled: {err}"
                                );
                                let grad_input = ndarray::ArrayD::from_elem(input_shape.clone(), g);
                                input_clone.add_grad(grad_input);
                            }
                        }
                    })),
                    requires_grad: true,
                    device: output_device,
                })));
            }
            Err(err) => {
                assert!(
                    !is_strict_device_execution(),
                    "sum CUDA forward failed while strict device execution is enabled: {err}"
                );
            }
        }
    }

    let sum_val = input.data_ref().sum();
    let result = ndarray::arr0(sum_val).into_dyn();

    let input_clone = input.clone();

    Tensor(Rc::new(RefCell::new(TensorData {
        data: result.into_shared(),
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
            let g = grad.first().copied().unwrap_or(0.0);
            let input_shape = input_clone.shape_vec();
            let grad_input = ndarray::ArrayD::from_elem(input_shape, g);
            input_clone.add_grad(grad_input);
        })),
        requires_grad: true,
        device: output_device,
    })))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::no_grad;
    use crate::precision::DType;
    use ndarray::{Array, IxDyn};

    fn make_tensor(shape: &[usize], data: Vec<f32>, dtype: DType) -> Tensor {
        let t = Tensor::from_array_no_grad(
            Array::from_shape_vec(IxDyn(shape), data)
                .expect("test tensor shape mismatch")
                .into_dyn(),
        );
        t.cast_inplace(dtype);
        t
    }

    fn make_training_tensor_with_dtype(shape: &[usize], data: Vec<f32>, dtype: DType) -> Tensor {
        let t = Tensor::from_data_with_grad_flag(
            Array::from_shape_vec(IxDyn(shape), data)
                .expect("test tensor shape mismatch")
                .into_dyn(),
            true,
        );
        t.cast_inplace(dtype);
        t
    }

    #[cfg(feature = "cuda")]
    fn make_training_tensor(shape: &[usize], data: Vec<f32>) -> Tensor {
        Tensor::from_data_with_grad_flag(
            Array::from_shape_vec(IxDyn(shape), data)
                .expect("test tensor shape mismatch")
                .into_dyn(),
            true,
        )
    }

    #[cfg(feature = "cuda")]
    fn assert_cuda_lowp_typed_output(out: &Tensor, dtype: DType, shape: &[usize]) {
        assert!(out.is_cuda());
        assert_eq!(out.dtype(), dtype);
        assert_eq!(out.shape_vec(), shape);
        assert!(!out.has_host_f32_data());
        let inner = out.0.borrow();
        assert!(inner.cuda_f32_data.is_none());
        match dtype {
            DType::F16 => assert!(inner.cuda_f16_data.is_some()),
            DType::BF16 => assert!(inner.cuda_bf16_data.is_some()),
            DType::I8 => {
                assert!(inner.cuda_i8_data.is_some());
                assert!(inner.i8_scale.is_some());
            }
            DType::F32 => panic!("expected lowp typed output, got F32"),
        }
    }

    #[test]
    fn bf16_add_no_grad_preserves_dtype_and_inputs() {
        let lhs = make_tensor(&[2], vec![1.0, -2.0], DType::BF16);
        let rhs = make_tensor(&[2], vec![0.5, 3.0], DType::BF16);

        let out = no_grad(|| lhs.clone() + rhs.clone());

        assert_eq!(lhs.dtype(), DType::BF16);
        assert_eq!(rhs.dtype(), DType::BF16);
        assert_eq!(out.dtype(), DType::BF16);
        out.with_storage_view(|view| match view {
            TensorStorageView::BF16(view) => {
                let vals = view.iter().map(|v| v.to_f32()).collect::<Vec<_>>();
                let expected = vec![bf16::from_f32(1.5).to_f32(), bf16::from_f32(1.0).to_f32()];
                assert_eq!(vals, expected);
            }
            TensorStorageView::F16(_) => panic!("bf16 add output should stay bf16 in no-grad"),
            TensorStorageView::F32(_) => panic!("bf16 add output should stay bf16 in no-grad"),
        });
    }

    #[test]
    fn mixed_add_no_grad_promotes_to_f32_without_mutating_bf16_input() {
        let lhs = make_tensor(&[2], vec![1.0, -2.0], DType::BF16);
        let rhs = make_tensor(&[2], vec![0.5, 3.0], DType::F32);

        let out = no_grad(|| lhs.clone() + rhs.clone());

        assert_eq!(lhs.dtype(), DType::BF16);
        assert_eq!(rhs.dtype(), DType::F32);
        assert_eq!(out.dtype(), DType::F32);
        let vals = out.data_ref().iter().copied().collect::<Vec<_>>();
        assert_eq!(vals, vec![1.5, 1.0]);
    }

    #[test]
    fn mixed_add_sub_no_grad_reads_native_low_precision_without_f32_cache() {
        for dtype in [DType::F16, DType::BF16, DType::I8] {
            let lowp = make_tensor(&[2, 3], vec![1.0, -2.0, 0.5, 3.0, -4.0, 2.0], dtype);
            let f32_same = make_tensor(&[2, 3], vec![0.5, 3.0, -1.5, 0.25, -0.75, 2.0], DType::F32);
            let f32_row = make_tensor(&[3], vec![0.5, 3.0, -1.5], DType::F32);
            let f32_row_scalar = make_tensor(&[2, 1], vec![0.5, -1.25], DType::F32);
            let f32_scalar = make_tensor(&[1], vec![-0.75], DType::F32);

            assert!(!lowp.has_host_f32_data());
            let same_add = no_grad(|| f32_same.clone() + lowp.clone());
            let same_sub = no_grad(|| lowp.clone() - f32_same.clone());
            let row_sub = no_grad(|| f32_row.clone() - lowp.clone());
            let row_scalar_add = no_grad(|| lowp.clone() + f32_row_scalar);
            let scalar_sub = no_grad(|| f32_scalar - lowp.clone());
            assert!(!lowp.has_host_f32_data());

            for out in [&same_add, &same_sub, &row_sub, &row_scalar_add, &scalar_sub] {
                assert_eq!(out.dtype(), DType::F32);
                assert_eq!(out.shape_vec(), vec![2, 3]);
            }

            for (got, expected) in same_add
                .data_ref()
                .iter()
                .zip([1.5f32, 1.0, -1.0, 3.25, -4.75, 4.0])
            {
                assert!((got - expected).abs() <= 0.08);
            }
            for (got, expected) in same_sub
                .data_ref()
                .iter()
                .zip([0.5f32, -5.0, 2.0, 2.75, -3.25, 0.0])
            {
                assert!((got - expected).abs() <= 0.08);
            }
            for (got, expected) in row_sub
                .data_ref()
                .iter()
                .zip([-0.5f32, 5.0, -2.0, -2.5, 7.0, -3.5])
            {
                assert!((got - expected).abs() <= 0.08);
            }
            for (got, expected) in row_scalar_add
                .data_ref()
                .iter()
                .zip([1.5f32, -1.5, 1.0, 1.75, -5.25, 0.75])
            {
                assert!((got - expected).abs() <= 0.08);
            }
            for (got, expected) in scalar_sub
                .data_ref()
                .iter()
                .zip([-1.75f32, 1.25, -1.25, -3.75, 3.25, -2.75])
            {
                assert!((got - expected).abs() <= 0.08);
            }
        }
    }

    #[test]
    fn mixed_add_sub_training_forward_reads_native_low_precision_and_outputs_f32() {
        for dtype in [DType::F16, DType::BF16, DType::I8] {
            let lowp = make_training_tensor_with_dtype(
                &[2, 3],
                vec![1.0, -2.0, 0.5, 3.0, -4.0, 2.0],
                dtype,
            );
            let f32_same = make_training_tensor_with_dtype(
                &[2, 3],
                vec![0.5, 3.0, -1.5, 0.25, -0.75, 2.0],
                DType::F32,
            );
            let f32_row = make_training_tensor_with_dtype(&[3], vec![0.5, 3.0, -1.5], DType::F32);
            let f32_scalar = make_training_tensor_with_dtype(&[1], vec![-0.75], DType::F32);

            assert!(!lowp.has_host_f32_data());
            let same_add = lowp.clone() + f32_same.clone();
            let same_sub = f32_same - lowp.clone();
            let row_sub = f32_row - lowp.clone();
            let scalar_add = lowp.clone() + f32_scalar;
            assert!(!lowp.has_host_f32_data());

            for out in [&same_add, &same_sub, &row_sub, &scalar_add] {
                assert_eq!(out.dtype(), DType::F32);
                assert!(out.has_host_f32_data());
                assert_eq!(out.shape_vec(), vec![2, 3]);
            }
        }
    }

    #[test]
    fn mixed_add_sub_directional_broadcasts_match_native_reference() {
        fn decoded_lowp(t: &Tensor) -> Vec<f32> {
            match t.native_storage_owned() {
                TensorStorageOwned::F16(data) => data.iter().map(|v| v.to_f32()).collect(),
                TensorStorageOwned::BF16(data) => data.iter().map(|v| v.to_f32()).collect(),
                TensorStorageOwned::I8(data, scale) => {
                    data.iter().map(|&v| (v as f32) * scale).collect()
                }
                TensorStorageOwned::F32(_) => panic!("tensor should stay native low precision"),
            }
        }

        fn assert_close(out: &Tensor, expected: &[f32]) {
            assert_eq!(out.dtype(), DType::F32);
            let vals = out.data_ref();
            assert_eq!(vals.len(), expected.len());
            for (&got, &want) in vals.iter().zip(expected) {
                assert!((got - want).abs() <= 1e-6, "got {got}, expected {want}");
            }
        }

        fn matrix_minus_row(matrix: &[f32], row: &[f32]) -> Vec<f32> {
            matrix
                .chunks_exact(row.len())
                .flat_map(|chunk| chunk.iter().zip(row).map(|(&a, &b)| a - b))
                .collect()
        }

        fn row_minus_matrix(row: &[f32], matrix: &[f32]) -> Vec<f32> {
            matrix
                .chunks_exact(row.len())
                .flat_map(|chunk| row.iter().zip(chunk).map(|(&a, &b)| a - b))
                .collect()
        }

        fn matrix_minus_row_scalar(matrix: &[f32], scalars: &[f32], row_len: usize) -> Vec<f32> {
            matrix
                .chunks_exact(row_len)
                .zip(scalars)
                .flat_map(|(row, &scalar)| row.iter().map(move |&v| v - scalar))
                .collect()
        }

        fn row_scalar_minus_matrix(scalars: &[f32], matrix: &[f32], row_len: usize) -> Vec<f32> {
            matrix
                .chunks_exact(row_len)
                .zip(scalars)
                .flat_map(|(row, &scalar)| row.iter().map(move |&v| scalar - v))
                .collect()
        }

        let matrix_vals = vec![1.0f32, -2.0, 0.5, 3.0, -4.0, 2.0];
        let f32_matrix_vals = vec![0.25f32, -1.5, 2.0, 0.75, 3.0, -0.5];
        let row_vals = vec![0.5f32, 3.0, -1.5];
        let f32_row_vals = vec![1.25f32, -0.75, 2.5];
        let row_scalar_vals = vec![0.5f32, -1.25];
        let f32_row_scalar_vals = vec![-0.25f32, 2.0];
        let vector_vals = vec![1.0f32, -2.0, 0.5, 3.0];
        let f32_vector_vals = vec![0.25f32, -1.5, 2.0, 0.75];
        let f32_scalar_val = -0.75f32;

        for dtype in [DType::F16, DType::BF16, DType::I8] {
            let lowp_matrix = make_tensor(&[2, 3], matrix_vals.clone(), dtype);
            let f32_matrix = make_tensor(&[2, 3], f32_matrix_vals.clone(), DType::F32);
            let lowp_matrix_vals = decoded_lowp(&lowp_matrix);
            assert!(!lowp_matrix.has_host_f32_data());
            assert_close(
                &no_grad(|| lowp_matrix.clone() - f32_matrix.clone()),
                &lowp_matrix_vals
                    .iter()
                    .zip(f32_matrix_vals.iter())
                    .map(|(&a, &b)| a - b)
                    .collect::<Vec<_>>(),
            );
            assert_close(
                &no_grad(|| f32_matrix.clone() - lowp_matrix.clone()),
                &f32_matrix_vals
                    .iter()
                    .zip(lowp_matrix_vals.iter())
                    .map(|(&a, &b)| a - b)
                    .collect::<Vec<_>>(),
            );
            assert!(!lowp_matrix.has_host_f32_data());

            let lowp_row = make_tensor(&[3], row_vals.clone(), dtype);
            let lowp_row_vals = decoded_lowp(&lowp_row);
            assert!(!lowp_row.has_host_f32_data());
            assert_close(
                &no_grad(|| {
                    lowp_matrix.clone() - make_tensor(&[3], f32_row_vals.clone(), DType::F32)
                }),
                &matrix_minus_row(&lowp_matrix_vals, &f32_row_vals),
            );
            assert_close(
                &no_grad(|| {
                    make_tensor(&[3], f32_row_vals.clone(), DType::F32) - lowp_matrix.clone()
                }),
                &row_minus_matrix(&f32_row_vals, &lowp_matrix_vals),
            );
            assert_close(
                &no_grad(|| f32_matrix.clone() - lowp_row.clone()),
                &matrix_minus_row(&f32_matrix_vals, &lowp_row_vals),
            );
            assert_close(
                &no_grad(|| lowp_row.clone() - f32_matrix.clone()),
                &row_minus_matrix(&lowp_row_vals, &f32_matrix_vals),
            );
            assert!(!lowp_matrix.has_host_f32_data());
            assert!(!lowp_row.has_host_f32_data());

            let lowp_row_scalar = make_tensor(&[2, 1], row_scalar_vals.clone(), dtype);
            let lowp_row_scalar_vals = decoded_lowp(&lowp_row_scalar);
            assert!(!lowp_row_scalar.has_host_f32_data());
            assert_close(
                &no_grad(|| {
                    lowp_matrix.clone()
                        - make_tensor(&[2, 1], f32_row_scalar_vals.clone(), DType::F32)
                }),
                &matrix_minus_row_scalar(&lowp_matrix_vals, &f32_row_scalar_vals, 3),
            );
            assert_close(
                &no_grad(|| {
                    make_tensor(&[2, 1], f32_row_scalar_vals.clone(), DType::F32)
                        - lowp_matrix.clone()
                }),
                &row_scalar_minus_matrix(&f32_row_scalar_vals, &lowp_matrix_vals, 3),
            );
            assert_close(
                &no_grad(|| f32_matrix.clone() - lowp_row_scalar.clone()),
                &matrix_minus_row_scalar(&f32_matrix_vals, &lowp_row_scalar_vals, 3),
            );
            assert_close(
                &no_grad(|| lowp_row_scalar.clone() - f32_matrix.clone()),
                &row_scalar_minus_matrix(&lowp_row_scalar_vals, &f32_matrix_vals, 3),
            );
            assert!(!lowp_row_scalar.has_host_f32_data());

            let lowp_vector = make_tensor(&[4], vector_vals.clone(), dtype);
            let lowp_scalar = make_tensor(&[1], vec![1.25], dtype);
            let lowp_vector_vals = decoded_lowp(&lowp_vector);
            let lowp_scalar_val = decoded_lowp(&lowp_scalar)[0];
            assert_close(
                &no_grad(|| {
                    lowp_vector.clone() - make_tensor(&[1], vec![f32_scalar_val], DType::F32)
                }),
                &lowp_vector_vals
                    .iter()
                    .map(|&v| v - f32_scalar_val)
                    .collect::<Vec<_>>(),
            );
            assert_close(
                &no_grad(|| {
                    make_tensor(&[1], vec![f32_scalar_val], DType::F32) - lowp_vector.clone()
                }),
                &lowp_vector_vals
                    .iter()
                    .map(|&v| f32_scalar_val - v)
                    .collect::<Vec<_>>(),
            );
            assert_close(
                &no_grad(|| {
                    make_tensor(&[4], f32_vector_vals.clone(), DType::F32) - lowp_scalar.clone()
                }),
                &f32_vector_vals
                    .iter()
                    .map(|&v| v - lowp_scalar_val)
                    .collect::<Vec<_>>(),
            );
            assert_close(
                &no_grad(|| {
                    lowp_scalar.clone() - make_tensor(&[4], f32_vector_vals.clone(), DType::F32)
                }),
                &f32_vector_vals
                    .iter()
                    .map(|&v| lowp_scalar_val - v)
                    .collect::<Vec<_>>(),
            );
            assert!(!lowp_vector.has_host_f32_data());
            assert!(!lowp_scalar.has_host_f32_data());
        }
    }

    #[test]
    #[ignore]
    fn cpu_mixed_lowp_binary_forward_perf_smoke() {
        fn bench<F>(iters: usize, mut f: F) -> (f64, f32)
        where
            F: FnMut() -> Tensor,
        {
            let mut sink = 0.0f32;
            let start = std::time::Instant::now();
            for _ in 0..iters {
                let out = f();
                let vals = out.data_ref();
                let first = *vals.iter().next().expect("perf output should be non-empty");
                let last = *vals.iter().last().expect("perf output should be non-empty");
                sink += first + last;
            }
            (start.elapsed().as_secs_f64() * 1.0e6 / iters as f64, sink)
        }

        let len = 1 << 20;
        let iters = 24;
        let f32_vals = (0..len)
            .map(|i| (i as f32 % 251.0) / 37.0 - 3.0)
            .collect::<Vec<_>>();
        let lowp_vals = (0..len)
            .map(|i| ((i * 17) as f32 % 241.0) / 41.0 - 2.5)
            .collect::<Vec<_>>();
        let rows = 4096;
        let row_len = len / rows;
        let row_vals = lowp_vals[..row_len].to_vec();

        for dtype in [DType::F16, DType::BF16, DType::I8] {
            let f32_same = make_tensor(&[len], f32_vals.clone(), DType::F32);
            let lowp_same = make_tensor(&[len], lowp_vals.clone(), dtype);
            let f32_matrix = make_tensor(&[rows, row_len], f32_vals.clone(), DType::F32);
            let lowp_row = make_tensor(&[row_len], row_vals.clone(), dtype);
            let lowp_matrix = make_tensor(&[rows, row_len], lowp_vals.clone(), dtype);
            let f32_row = make_tensor(&[row_len], f32_vals[..row_len].to_vec(), DType::F32);
            let f32_row_scalar = make_tensor(
                &[rows, 1],
                (0..rows).map(|i| (i as f32 % 127.0) / 31.0 - 2.0).collect(),
                DType::F32,
            );
            let f32_scalar = make_tensor(&[1], vec![-0.375], DType::F32);

            assert!(!lowp_same.has_host_f32_data());
            assert!(!lowp_row.has_host_f32_data());
            assert!(!lowp_matrix.has_host_f32_data());

            let (same_add_us, same_add_sink) =
                bench(iters, || no_grad(|| f32_same.clone() + lowp_same.clone()));
            let (same_sub_us, same_sub_sink) =
                bench(iters, || no_grad(|| lowp_same.clone() - f32_same.clone()));
            let (same_mul_us, same_mul_sink) =
                bench(iters, || no_grad(|| f32_same.clone() * lowp_same.clone()));
            let (row_add_us, row_add_sink) =
                bench(iters, || no_grad(|| f32_matrix.clone() + lowp_row.clone()));
            let (row_sub_us, row_sub_sink) =
                bench(iters, || no_grad(|| f32_row.clone() - lowp_matrix.clone()));
            let (row_mul_us, row_mul_sink) =
                bench(iters, || no_grad(|| f32_matrix.clone() * lowp_row.clone()));
            let (row_scalar_sub_us, row_scalar_sub_sink) = bench(iters, || {
                no_grad(|| lowp_matrix.clone() - f32_row_scalar.clone())
            });
            let (row_scalar_mul_us, row_scalar_mul_sink) = bench(iters, || {
                no_grad(|| lowp_matrix.clone() * f32_row_scalar.clone())
            });
            let (scalar_add_us, scalar_add_sink) =
                bench(iters, || no_grad(|| f32_scalar.clone() + lowp_same.clone()));
            let (scalar_sub_us, scalar_sub_sink) =
                bench(iters, || no_grad(|| f32_scalar.clone() - lowp_same.clone()));

            assert!(!lowp_same.has_host_f32_data());
            assert!(!lowp_row.has_host_f32_data());
            assert!(!lowp_matrix.has_host_f32_data());

            println!(
                "cpu mixed lowp binary dtype={dtype:?} len={len} rows={rows} row_len={row_len} iters={iters} same_add={same_add_us:.3}us same_sub={same_sub_us:.3}us same_mul={same_mul_us:.3}us row_add={row_add_us:.3}us row_sub={row_sub_us:.3}us row_mul={row_mul_us:.3}us row_scalar_sub={row_scalar_sub_us:.3}us row_scalar_mul={row_scalar_mul_us:.3}us scalar_add={scalar_add_us:.3}us scalar_sub={scalar_sub_us:.3}us sink={:.5}",
                same_add_sink
                    + same_sub_sink
                    + same_mul_sink
                    + row_add_sink
                    + row_sub_sink
                    + row_mul_sink
                    + row_scalar_sub_sink
                    + row_scalar_mul_sink
                    + scalar_add_sink
                    + scalar_sub_sink
            );
        }
    }

    #[test]
    fn mixed_mul_no_grad_reads_native_low_precision_without_f32_cache() {
        for dtype in [DType::F16, DType::BF16, DType::I8] {
            let lowp = make_tensor(&[2, 3], vec![1.0, -2.0, 0.5, 3.0, -4.0, 2.0], dtype);
            let f32_same = make_tensor(&[2, 3], vec![0.5, 3.0, -1.5, 0.25, -0.75, 2.0], DType::F32);
            let f32_row = make_tensor(&[3], vec![0.5, 3.0, -1.5], DType::F32);

            assert!(!lowp.has_host_f32_data());
            let same_out = no_grad(|| lowp.clone() * f32_same.clone());
            let row_out = no_grad(|| f32_row.clone() * lowp.clone());
            assert!(!lowp.has_host_f32_data());

            assert_eq!(same_out.dtype(), DType::F32);
            assert_eq!(row_out.dtype(), DType::F32);
            assert_eq!(same_out.shape_vec(), vec![2, 3]);
            assert_eq!(row_out.shape_vec(), vec![2, 3]);

            let same_vals = same_out.data_ref().iter().copied().collect::<Vec<_>>();
            let row_vals = row_out.data_ref().iter().copied().collect::<Vec<_>>();
            for (got, expected) in same_vals.iter().zip([0.5f32, -6.0, -0.75, 0.75, 3.0, 4.0]) {
                assert!((got - expected).abs() <= 0.08);
            }
            for (got, expected) in row_vals.iter().zip([0.5f32, -6.0, -0.75, 1.5, -12.0, -3.0]) {
                assert!((got - expected).abs() <= 0.08);
            }
        }
    }

    #[test]
    fn mixed_mul_training_forward_reads_native_low_precision_and_outputs_f32() {
        for dtype in [DType::F16, DType::BF16, DType::I8] {
            let lowp = make_training_tensor_with_dtype(
                &[2, 3],
                vec![1.0, -2.0, 0.5, 3.0, -4.0, 2.0],
                dtype,
            );
            let f32_same = make_training_tensor_with_dtype(
                &[2, 3],
                vec![0.5, 3.0, -1.5, 0.25, -0.75, 2.0],
                DType::F32,
            );
            let f32_row = make_training_tensor_with_dtype(&[3], vec![0.5, 3.0, -1.5], DType::F32);

            assert!(!lowp.has_host_f32_data());
            let same_out = lowp.clone() * f32_same;
            let row_out = f32_row * lowp.clone();
            assert!(!lowp.has_host_f32_data());

            for out in [&same_out, &row_out] {
                assert_eq!(out.dtype(), DType::F32);
                assert!(out.has_host_f32_data());
                assert_eq!(out.shape_vec(), vec![2, 3]);
            }
        }
    }

    #[test]
    fn mixed_mul_scalar_broadcast_forward_reads_native_low_precision() {
        for dtype in [DType::F16, DType::BF16, DType::I8] {
            let lowp = make_tensor(&[2, 3], vec![1.0, -2.0, 0.5, 3.0, -4.0, 2.0], dtype);
            let f32_row_scalar = make_tensor(&[2, 1], vec![0.5, -1.25], DType::F32);

            assert!(!lowp.has_host_f32_data());
            let row_scalar_out = no_grad(|| lowp.clone() * f32_row_scalar);
            assert!(!lowp.has_host_f32_data());

            assert_eq!(row_scalar_out.dtype(), DType::F32);
            assert_eq!(row_scalar_out.shape_vec(), vec![2, 3]);
            for (got, expected) in row_scalar_out
                .data_ref()
                .iter()
                .zip([0.5f32, -1.0, 0.25, -3.75, 5.0, -2.5])
            {
                assert!((got - expected).abs() <= 0.08);
            }

            let lowp = make_tensor(&[2, 3], vec![1.0, -2.0, 0.5, 3.0, -4.0, 2.0], dtype);
            let f32_scalar = make_tensor(&[1], vec![-0.75], DType::F32);

            assert!(!lowp.has_host_f32_data());
            let scalar_out = no_grad(|| f32_scalar * lowp.clone());
            assert!(!lowp.has_host_f32_data());

            assert_eq!(scalar_out.dtype(), DType::F32);
            assert_eq!(scalar_out.shape_vec(), vec![2, 3]);
            for (got, expected) in scalar_out
                .data_ref()
                .iter()
                .zip([-0.75f32, 1.5, -0.375, -2.25, 3.0, -1.5])
            {
                assert!((got - expected).abs() <= 0.08);
            }
        }
    }

    #[test]
    fn mixed_mul_backward_same_shape_reads_native_low_precision_and_writes_f32_grads() {
        let grad = Array::from_shape_vec(IxDyn(&[2, 3]), vec![1.0f32, -0.5, 0.25, 2.0, -1.0, 0.5])
            .expect("grad shape mismatch")
            .into_dyn();
        let grad_slice = grad.as_slice_memory_order().unwrap();
        let lowp_vals = vec![1.0f32, -2.0, 0.5, 3.0, -4.0, 2.0];
        let f32_vals = vec![0.25f32, -1.5, 2.0, 0.75, 3.0, -0.5];

        fn decoded_lowp(t: &Tensor) -> Vec<f32> {
            match t.native_storage_owned() {
                TensorStorageOwned::F16(data) => data.iter().map(|v| v.to_f32()).collect(),
                TensorStorageOwned::BF16(data) => data.iter().map(|v| v.to_f32()).collect(),
                TensorStorageOwned::I8(data, scale) => {
                    data.iter().map(|&v| (v as f32) * scale).collect()
                }
                TensorStorageOwned::F32(_) => panic!("tensor should stay native low precision"),
            }
        }

        for dtype in [DType::F16, DType::BF16, DType::I8] {
            let lowp = make_tensor(&[2, 3], lowp_vals.clone(), dtype);
            let f32_operand = make_tensor(&[2, 3], f32_vals.clone(), DType::F32);
            assert!(!lowp.has_host_f32_data());
            add_cpu_binary_grads(&lowp, &f32_operand, grad.view(), BinaryOp::Mul);
            assert!(!lowp.has_host_f32_data());

            let lowp_decoded = decoded_lowp(&lowp);
            let lowp_grad = lowp
                .grad_ref()
                .as_ref()
                .expect("lowp grad missing")
                .to_owned();
            let f32_grad = f32_operand
                .grad_ref()
                .as_ref()
                .expect("f32 grad missing")
                .to_owned();
            for (((&lowp_got, &f32_got), &g), (&f32_x, &lowp_x)) in lowp_grad
                .iter()
                .zip(f32_grad.iter())
                .zip(grad_slice.iter())
                .zip(f32_vals.iter().zip(lowp_decoded.iter()))
            {
                assert!((lowp_got - g * f32_x).abs() <= 0.05);
                assert!((f32_got - g * lowp_x).abs() <= 0.05);
            }

            let f32_operand = make_tensor(&[2, 3], f32_vals.clone(), DType::F32);
            let lowp = make_tensor(&[2, 3], lowp_vals.clone(), dtype);
            assert!(!lowp.has_host_f32_data());
            add_cpu_binary_grads(&f32_operand, &lowp, grad.view(), BinaryOp::Mul);
            assert!(!lowp.has_host_f32_data());

            let lowp_decoded = decoded_lowp(&lowp);
            let f32_grad = f32_operand
                .grad_ref()
                .as_ref()
                .expect("f32 grad missing")
                .to_owned();
            let lowp_grad = lowp
                .grad_ref()
                .as_ref()
                .expect("lowp grad missing")
                .to_owned();
            for (((&f32_got, &lowp_got), &g), (&lowp_x, &f32_x)) in f32_grad
                .iter()
                .zip(lowp_grad.iter())
                .zip(grad_slice.iter())
                .zip(lowp_decoded.iter().zip(f32_vals.iter()))
            {
                assert!((f32_got - g * lowp_x).abs() <= 0.05);
                assert!((lowp_got - g * f32_x).abs() <= 0.05);
            }
        }
    }

    #[test]
    fn mixed_mul_scalar_broadcast_backward_reads_native_low_precision_and_writes_f32_grads() {
        let grad = Array::from_shape_vec(IxDyn(&[2, 3]), vec![1.0f32, -0.5, 0.25, 2.0, -1.0, 0.5])
            .expect("grad shape mismatch")
            .into_dyn();
        let grad_slice = grad.as_slice_memory_order().unwrap();
        let lowp_matrix_vals = vec![1.0f32, -2.0, 0.5, 3.0, -4.0, 2.0];
        let f32_matrix_vals = vec![0.25f32, -1.5, 2.0, 0.75, 3.0, -0.5];
        let f32_row_scalar_vals = vec![0.5f32, -1.25];
        let lowp_row_scalar_vals = vec![1.25f32, -0.75];

        fn decoded_lowp(t: &Tensor) -> Vec<f32> {
            match t.native_storage_owned() {
                TensorStorageOwned::F16(data) => data.iter().map(|v| v.to_f32()).collect(),
                TensorStorageOwned::BF16(data) => data.iter().map(|v| v.to_f32()).collect(),
                TensorStorageOwned::I8(data, scale) => {
                    data.iter().map(|&v| (v as f32) * scale).collect()
                }
                TensorStorageOwned::F32(_) => panic!("tensor should stay native low precision"),
            }
        }

        fn assert_matrix_grad_by_row_scalar(got: &Tensor, row_scalars: &[f32], grad_slice: &[f32]) {
            let grad = got
                .grad_ref()
                .as_ref()
                .expect("matrix grad missing")
                .to_owned();
            assert_eq!(grad.shape(), &[2, 3]);
            for (row, &row_scalar) in row_scalars.iter().enumerate().take(2) {
                for col in 0..3 {
                    let idx = row * 3 + col;
                    let expected = grad_slice[idx] * row_scalar;
                    assert!((grad.as_slice_memory_order().unwrap()[idx] - expected).abs() <= 0.05);
                }
            }
        }

        fn assert_row_scalar_grad(got: &Tensor, matrix_vals: &[f32], grad_slice: &[f32]) {
            let grad = got
                .grad_ref()
                .as_ref()
                .expect("row scalar grad missing")
                .to_owned();
            assert_eq!(grad.shape(), &[2, 1]);
            for row in 0..2 {
                let expected = (0..3)
                    .map(|col| {
                        let idx = row * 3 + col;
                        grad_slice[idx] * matrix_vals[idx]
                    })
                    .sum::<f32>();
                assert!((grad.as_slice_memory_order().unwrap()[row] - expected).abs() <= 0.05);
            }
        }

        for dtype in [DType::F16, DType::BF16, DType::I8] {
            let lowp_matrix = make_tensor(&[2, 3], lowp_matrix_vals.clone(), dtype);
            let f32_row_scalar = make_tensor(&[2, 1], f32_row_scalar_vals.clone(), DType::F32);
            assert!(!lowp_matrix.has_host_f32_data());
            add_cpu_binary_grads(&lowp_matrix, &f32_row_scalar, grad.view(), BinaryOp::Mul);
            assert!(!lowp_matrix.has_host_f32_data());
            let lowp_matrix_decoded = decoded_lowp(&lowp_matrix);
            assert_matrix_grad_by_row_scalar(&lowp_matrix, &f32_row_scalar_vals, grad_slice);
            assert_row_scalar_grad(&f32_row_scalar, &lowp_matrix_decoded, grad_slice);

            let f32_matrix = make_tensor(&[2, 3], f32_matrix_vals.clone(), DType::F32);
            let lowp_row_scalar = make_tensor(&[2, 1], lowp_row_scalar_vals.clone(), dtype);
            assert!(!lowp_row_scalar.has_host_f32_data());
            add_cpu_binary_grads(&f32_matrix, &lowp_row_scalar, grad.view(), BinaryOp::Mul);
            assert!(!lowp_row_scalar.has_host_f32_data());
            let lowp_row_scalar_decoded = decoded_lowp(&lowp_row_scalar);
            assert_matrix_grad_by_row_scalar(&f32_matrix, &lowp_row_scalar_decoded, grad_slice);
            assert_row_scalar_grad(&lowp_row_scalar, &f32_matrix_vals, grad_slice);

            let lowp_vector = make_tensor(&[2, 3], lowp_matrix_vals.clone(), dtype);
            let f32_scalar = make_tensor(&[1], vec![-0.75], DType::F32);
            assert!(!lowp_vector.has_host_f32_data());
            add_cpu_binary_grads(&lowp_vector, &f32_scalar, grad.view(), BinaryOp::Mul);
            assert!(!lowp_vector.has_host_f32_data());
            let lowp_vector_decoded = decoded_lowp(&lowp_vector);

            let lowp_grad = lowp_vector
                .grad_ref()
                .as_ref()
                .expect("lowp grad missing")
                .to_owned();
            for (&got, &g) in lowp_grad.iter().zip(grad_slice) {
                assert!((got - g * -0.75).abs() <= 0.05);
            }

            let scalar_grad = f32_scalar
                .grad_ref()
                .as_ref()
                .expect("scalar grad missing")
                .to_owned();
            let expected_scalar_grad = grad_slice
                .iter()
                .zip(lowp_vector_decoded.iter())
                .map(|(&g, &x)| g * x)
                .sum::<f32>();
            assert!(
                (scalar_grad.as_slice_memory_order().unwrap()[0] - expected_scalar_grad).abs()
                    <= 0.05
            );
        }
    }

    #[test]
    fn bf16_mul_no_grad_preserves_dtype() {
        let lhs = make_tensor(&[2], vec![2.0, -1.5], DType::BF16);
        let rhs = make_tensor(&[2], vec![0.25, 2.0], DType::BF16);

        let out = no_grad(|| lhs * rhs);
        assert_eq!(out.dtype(), DType::BF16);
    }

    #[test]
    fn f16_same_shape_binary_no_grad_preserves_native_storage() {
        let lhs = make_tensor(&[2, 2], vec![1.0, -2.0, 0.5, 3.0], DType::F16);
        let rhs = make_tensor(&[2, 2], vec![0.5, 3.0, -1.5, 0.25], DType::F16);

        let add_out = no_grad(|| lhs.clone() + rhs.clone());
        let sub_out = no_grad(|| lhs.clone() - rhs.clone());
        let mul_out = no_grad(|| lhs * rhs);

        for out in [&add_out, &sub_out, &mul_out] {
            assert_eq!(out.dtype(), DType::F16);
            out.with_storage_view_preferring(StoragePreference::Native, |view| match view {
                TensorStorageView::F16(_) => {}
                TensorStorageView::F32(_) => {
                    panic!("same-shape f16 binary should keep native f16 storage")
                }
                TensorStorageView::BF16(_) => {
                    panic!("same-shape f16 binary should not produce bf16 storage")
                }
            });
        }

        let add_vals = add_out.data_ref().iter().copied().collect::<Vec<_>>();
        assert_eq!(
            add_vals,
            vec![
                f16::from_f32(1.5).to_f32(),
                f16::from_f32(1.0).to_f32(),
                f16::from_f32(-1.0).to_f32(),
                f16::from_f32(3.25).to_f32(),
            ]
        );
    }

    #[test]
    fn bf16_same_shape_binary_no_grad_preserves_native_storage() {
        let lhs = make_tensor(&[2, 2], vec![1.0, -2.0, 0.5, 3.0], DType::BF16);
        let rhs = make_tensor(&[2, 2], vec![0.5, 3.0, -1.5, 0.25], DType::BF16);

        let add_out = no_grad(|| lhs.clone() + rhs.clone());
        let sub_out = no_grad(|| lhs.clone() - rhs.clone());
        let mul_out = no_grad(|| lhs * rhs);

        for out in [&add_out, &sub_out, &mul_out] {
            assert_eq!(out.dtype(), DType::BF16);
            out.with_storage_view_preferring(StoragePreference::Native, |view| match view {
                TensorStorageView::BF16(_) => {}
                TensorStorageView::F32(_) => {
                    panic!("same-shape bf16 binary should keep native bf16 storage")
                }
                TensorStorageView::F16(_) => {
                    panic!("same-shape bf16 binary should not produce f16 storage")
                }
            });
        }

        let add_vals = add_out.data_ref().iter().copied().collect::<Vec<_>>();
        assert_eq!(
            add_vals,
            vec![
                bf16::from_f32(1.5).to_f32(),
                bf16::from_f32(1.0).to_f32(),
                bf16::from_f32(-1.0).to_f32(),
                bf16::from_f32(3.25).to_f32(),
            ]
        );
    }

    #[test]
    fn i8_same_shape_binary_no_grad_preserves_native_storage_with_dynamic_scale() {
        let lhs = make_tensor(&[2, 2], vec![1.0, -2.0, 0.5, 3.0], DType::I8);
        let rhs = make_tensor(&[2, 2], vec![0.5, 3.0, -1.5, 0.25], DType::I8);

        let add_out = no_grad(|| lhs.clone() + rhs.clone());
        let sub_out = no_grad(|| lhs.clone() - rhs.clone());
        let mul_out = no_grad(|| lhs.clone() * rhs.clone());

        for out in [&add_out, &sub_out, &mul_out] {
            assert_eq!(out.dtype(), DType::I8);
            assert!(!out.has_host_f32_data());
            match out.native_storage_owned() {
                TensorStorageOwned::I8(data, scale) => {
                    assert!(scale.is_finite() && scale > 0.0);
                    assert_eq!(data.len(), 4);
                }
                _ => panic!("same-shape i8 binary should keep native i8 storage"),
            }
        }

        let add_vals = add_out.data_ref().iter().copied().collect::<Vec<_>>();
        for (got, expected) in add_vals.iter().zip([1.5f32, 1.0, -1.0, 3.25]) {
            assert!((got - expected).abs() <= 0.03);
        }
    }

    #[test]
    fn row_broadcast_binary_no_grad_preserves_native_low_precision_storage() {
        for dtype in [DType::F16, DType::BF16, DType::I8] {
            let matrix = make_tensor(&[2, 3], vec![1.0, -2.0, 0.5, 3.0, -4.0, 2.0], dtype);
            let row = make_tensor(&[3], vec![0.5, 3.0, -1.5], dtype);

            assert!(!matrix.has_host_f32_data());
            assert!(!row.has_host_f32_data());
            let add_out = no_grad(|| matrix.clone() + row.clone());
            let sub_out = no_grad(|| row.clone() - matrix.clone());
            let mul_out = no_grad(|| matrix.clone() * row.clone());
            assert!(!matrix.has_host_f32_data());
            assert!(!row.has_host_f32_data());

            for out in [&add_out, &sub_out, &mul_out] {
                assert_eq!(out.dtype(), dtype);
                assert_eq!(out.shape_vec(), vec![2, 3]);
                assert!(!out.has_host_f32_data());
            }

            let add_vals = add_out.data_ref().iter().copied().collect::<Vec<_>>();
            let sub_vals = sub_out.data_ref().iter().copied().collect::<Vec<_>>();
            let mul_vals = mul_out.data_ref().iter().copied().collect::<Vec<_>>();
            for (got, expected) in add_vals.iter().zip([1.5f32, 1.0, -1.0, 3.5, -1.0, 0.5]) {
                assert!((got - expected).abs() <= 0.05);
            }
            for (got, expected) in sub_vals.iter().zip([-0.5f32, 5.0, -2.0, -2.5, 7.0, -3.5]) {
                assert!((got - expected).abs() <= 0.08);
            }
            for (got, expected) in mul_vals.iter().zip([0.5f32, -6.0, -0.75, 1.5, -12.0, -3.0]) {
                assert!((got - expected).abs() <= 0.08);
            }
        }
    }

    #[test]
    fn same_shape_binary_training_reads_native_low_precision_and_outputs_f32() {
        for dtype in [DType::F16, DType::BF16, DType::I8] {
            let lhs = make_training_tensor_with_dtype(&[2, 2], vec![1.0, -2.0, 0.5, 3.0], dtype);
            let rhs = make_training_tensor_with_dtype(&[2, 2], vec![0.5, 3.0, -1.5, 0.25], dtype);

            assert!(!lhs.has_host_f32_data());
            assert!(!rhs.has_host_f32_data());
            let add_out = lhs.clone() + rhs.clone();
            let sub_out = lhs.clone() - rhs.clone();
            let mul_out = lhs.clone() * rhs.clone();
            assert!(!lhs.has_host_f32_data());
            assert!(!rhs.has_host_f32_data());

            for out in [&add_out, &sub_out, &mul_out] {
                assert_eq!(out.dtype(), DType::F32);
                assert!(out.has_host_f32_data());
            }

            let add_vals = add_out.data_ref().iter().copied().collect::<Vec<_>>();
            let sub_vals = sub_out.data_ref().iter().copied().collect::<Vec<_>>();
            let mul_vals = mul_out.data_ref().iter().copied().collect::<Vec<_>>();
            for (got, expected) in add_vals.iter().zip([1.5f32, 1.0, -1.0, 3.25]) {
                assert!((got - expected).abs() <= 0.03);
            }
            for (got, expected) in sub_vals.iter().zip([0.5f32, -5.0, 2.0, 2.75]) {
                assert!((got - expected).abs() <= 0.03);
            }
            for (got, expected) in mul_vals.iter().zip([0.5f32, -6.0, -0.75, 0.75]) {
                assert!((got - expected).abs() <= 0.03);
            }
        }
    }

    #[test]
    fn row_broadcast_binary_training_reads_native_low_precision_and_outputs_f32() {
        for dtype in [DType::F16, DType::BF16, DType::I8] {
            let matrix = make_training_tensor_with_dtype(
                &[2, 3],
                vec![1.0, -2.0, 0.5, 3.0, -4.0, 2.0],
                dtype,
            );
            let row = make_training_tensor_with_dtype(&[3], vec![0.5, 3.0, -1.5], dtype);

            assert!(!matrix.has_host_f32_data());
            assert!(!row.has_host_f32_data());
            let add_out = matrix.clone() + row.clone();
            let sub_out = row.clone() - matrix.clone();
            let mul_out = matrix.clone() * row.clone();
            assert!(!matrix.has_host_f32_data());
            assert!(!row.has_host_f32_data());

            for out in [&add_out, &sub_out, &mul_out] {
                assert_eq!(out.dtype(), DType::F32);
                assert_eq!(out.shape_vec(), vec![2, 3]);
            }

            let add_vals = add_out.data_ref().iter().copied().collect::<Vec<_>>();
            let sub_vals = sub_out.data_ref().iter().copied().collect::<Vec<_>>();
            let mul_vals = mul_out.data_ref().iter().copied().collect::<Vec<_>>();
            for (got, expected) in add_vals.iter().zip([1.5f32, 1.0, -1.0, 3.5, -1.0, 0.5]) {
                assert!((got - expected).abs() <= 0.03);
            }
            for (got, expected) in sub_vals.iter().zip([-0.5f32, 5.0, -2.0, -2.5, 7.0, -3.5]) {
                assert!((got - expected).abs() <= 0.05);
            }
            for (got, expected) in mul_vals.iter().zip([0.5f32, -6.0, -0.75, 1.5, -12.0, -3.0]) {
                assert!((got - expected).abs() <= 0.05);
            }
        }
    }

    #[test]
    fn mul_backward_same_shape_reads_native_low_precision_and_writes_f32_grads() {
        let grad = Array::from_shape_vec(IxDyn(&[2, 2]), vec![1.0f32, -0.5, 0.25, 2.0])
            .expect("grad shape mismatch")
            .into_dyn();

        for dtype in [DType::F16, DType::BF16, DType::I8] {
            let lhs = make_tensor(&[2, 2], vec![1.0, -2.0, 0.5, 3.0], dtype);
            let rhs = make_tensor(&[2, 2], vec![0.5, 3.0, -1.5, 0.25], dtype);

            assert!(!lhs.has_host_f32_data());
            assert!(!rhs.has_host_f32_data());
            add_cpu_binary_grads(&lhs, &rhs, grad.view(), BinaryOp::Mul);
            assert!(!lhs.has_host_f32_data());
            assert!(!rhs.has_host_f32_data());

            let rhs_decoded = match rhs.native_storage_owned() {
                TensorStorageOwned::F16(data) => {
                    data.iter().map(|v| v.to_f32()).collect::<Vec<_>>()
                }
                TensorStorageOwned::BF16(data) => {
                    data.iter().map(|v| v.to_f32()).collect::<Vec<_>>()
                }
                TensorStorageOwned::I8(data, scale) => {
                    data.iter().map(|&v| (v as f32) * scale).collect::<Vec<_>>()
                }
                TensorStorageOwned::F32(_) => panic!("rhs should stay native low precision"),
            };
            let lhs_decoded = match lhs.native_storage_owned() {
                TensorStorageOwned::F16(data) => {
                    data.iter().map(|v| v.to_f32()).collect::<Vec<_>>()
                }
                TensorStorageOwned::BF16(data) => {
                    data.iter().map(|v| v.to_f32()).collect::<Vec<_>>()
                }
                TensorStorageOwned::I8(data, scale) => {
                    data.iter().map(|&v| (v as f32) * scale).collect::<Vec<_>>()
                }
                TensorStorageOwned::F32(_) => panic!("lhs should stay native low precision"),
            };

            let lhs_grad = lhs
                .grad_ref()
                .as_ref()
                .expect("lhs grad missing")
                .to_owned();
            let rhs_grad = rhs
                .grad_ref()
                .as_ref()
                .expect("rhs grad missing")
                .to_owned();
            assert_eq!(lhs_grad.shape(), &[2, 2]);
            assert_eq!(rhs_grad.shape(), &[2, 2]);
            for ((&got, &g), &x) in lhs_grad.iter().zip(grad.iter()).zip(rhs_decoded.iter()) {
                assert!((got - g * x).abs() <= 0.03);
            }
            for ((&got, &g), &x) in rhs_grad.iter().zip(grad.iter()).zip(lhs_decoded.iter()) {
                assert!((got - g * x).abs() <= 0.03);
            }
        }
    }

    #[test]
    fn mul_backward_row_broadcast_reads_native_low_precision_and_writes_f32_grads() {
        let grad = Array::from_shape_vec(IxDyn(&[2, 3]), vec![1.0f32, -0.5, 0.25, 2.0, -1.0, 0.5])
            .expect("grad shape mismatch")
            .into_dyn();

        fn decoded(t: &Tensor) -> Vec<f32> {
            match t.native_storage_owned() {
                TensorStorageOwned::F16(data) => data.iter().map(|v| v.to_f32()).collect(),
                TensorStorageOwned::BF16(data) => data.iter().map(|v| v.to_f32()).collect(),
                TensorStorageOwned::I8(data, scale) => {
                    data.iter().map(|&v| (v as f32) * scale).collect()
                }
                TensorStorageOwned::F32(_) => panic!("tensor should stay native low precision"),
            }
        }

        for dtype in [DType::F16, DType::BF16, DType::I8] {
            let matrix = make_tensor(&[2, 3], vec![1.0, -2.0, 0.5, 3.0, -4.0, 2.0], dtype);
            let row = make_tensor(&[3], vec![0.5, 3.0, -1.5], dtype);

            assert!(!matrix.has_host_f32_data());
            assert!(!row.has_host_f32_data());
            add_cpu_binary_grads(&matrix, &row, grad.view(), BinaryOp::Mul);
            assert!(!matrix.has_host_f32_data());
            assert!(!row.has_host_f32_data());

            let matrix_vals = decoded(&matrix);
            let row_vals = decoded(&row);
            let matrix_grad = matrix
                .grad_ref()
                .as_ref()
                .expect("matrix grad missing")
                .to_owned();
            let row_grad = row
                .grad_ref()
                .as_ref()
                .expect("row grad missing")
                .to_owned();

            for row_idx in 0..2 {
                for (col, &row_val) in row_vals.iter().enumerate().take(3) {
                    let idx = row_idx * 3 + col;
                    let expected = grad.as_slice_memory_order().unwrap()[idx] * row_val;
                    assert!(
                        (matrix_grad.as_slice_memory_order().unwrap()[idx] - expected).abs()
                            <= 0.05
                    );
                }
            }
            for col in 0..3 {
                let expected = (0..2)
                    .map(|row_idx| {
                        let idx = row_idx * 3 + col;
                        grad.as_slice_memory_order().unwrap()[idx] * matrix_vals[idx]
                    })
                    .sum::<f32>();
                assert!((row_grad.as_slice_memory_order().unwrap()[col] - expected).abs() <= 0.05);
            }

            let row = make_tensor(&[3], vec![0.5, 3.0, -1.5], dtype);
            let matrix = make_tensor(&[2, 3], vec![1.0, -2.0, 0.5, 3.0, -4.0, 2.0], dtype);
            add_cpu_binary_grads(&row, &matrix, grad.view(), BinaryOp::Mul);
            assert!(!row.has_host_f32_data());
            assert!(!matrix.has_host_f32_data());

            let row_vals = decoded(&row);
            let matrix_vals = decoded(&matrix);
            let row_grad = row
                .grad_ref()
                .as_ref()
                .expect("row grad missing")
                .to_owned();
            let matrix_grad = matrix
                .grad_ref()
                .as_ref()
                .expect("matrix grad missing")
                .to_owned();
            for col in 0..3 {
                let expected = (0..2)
                    .map(|row_idx| {
                        let idx = row_idx * 3 + col;
                        grad.as_slice_memory_order().unwrap()[idx] * matrix_vals[idx]
                    })
                    .sum::<f32>();
                assert!((row_grad.as_slice_memory_order().unwrap()[col] - expected).abs() <= 0.05);
            }
            for row_idx in 0..2 {
                for (col, &row_val) in row_vals.iter().enumerate().take(3) {
                    let idx = row_idx * 3 + col;
                    let expected = grad.as_slice_memory_order().unwrap()[idx] * row_val;
                    assert!(
                        (matrix_grad.as_slice_memory_order().unwrap()[idx] - expected).abs()
                            <= 0.05
                    );
                }
            }
        }
    }

    #[test]
    fn mixed_mul_backward_row_broadcast_reads_native_low_precision_and_writes_f32_grads() {
        let grad = Array::from_shape_vec(IxDyn(&[2, 3]), vec![1.0f32, -0.5, 0.25, 2.0, -1.0, 0.5])
            .expect("grad shape mismatch")
            .into_dyn();
        let grad_slice = grad.as_slice_memory_order().unwrap();
        let f32_matrix_vals = vec![0.25f32, -1.5, 2.0, 0.75, 3.0, -0.5];
        let f32_row_vals = vec![0.5f32, 3.0, -1.5];
        let lowp_matrix_vals = vec![1.0f32, -2.0, 0.5, 3.0, -4.0, 2.0];
        let lowp_row_vals = vec![1.25f32, -0.75, 2.5];

        fn decoded_lowp(t: &Tensor) -> Vec<f32> {
            match t.native_storage_owned() {
                TensorStorageOwned::F16(data) => data.iter().map(|v| v.to_f32()).collect(),
                TensorStorageOwned::BF16(data) => data.iter().map(|v| v.to_f32()).collect(),
                TensorStorageOwned::I8(data, scale) => {
                    data.iter().map(|&v| (v as f32) * scale).collect()
                }
                TensorStorageOwned::F32(_) => panic!("tensor should stay native low precision"),
            }
        }

        fn assert_matrix_grad(got: &Tensor, row_vals: &[f32], grad_slice: &[f32]) {
            let grad = got
                .grad_ref()
                .as_ref()
                .expect("matrix grad missing")
                .to_owned();
            assert_eq!(grad.shape(), &[2, 3]);
            for row_idx in 0..2 {
                for (col, &row_val) in row_vals.iter().enumerate().take(3) {
                    let idx = row_idx * 3 + col;
                    let expected = grad_slice[idx] * row_val;
                    assert!((grad.as_slice_memory_order().unwrap()[idx] - expected).abs() <= 0.05);
                }
            }
        }

        fn assert_row_grad(got: &Tensor, matrix_vals: &[f32], grad_slice: &[f32]) {
            let grad = got
                .grad_ref()
                .as_ref()
                .expect("row grad missing")
                .to_owned();
            assert_eq!(grad.shape(), &[3]);
            for col in 0..3 {
                let expected = (0..2)
                    .map(|row_idx| {
                        let idx = row_idx * 3 + col;
                        grad_slice[idx] * matrix_vals[idx]
                    })
                    .sum::<f32>();
                assert!((grad.as_slice_memory_order().unwrap()[col] - expected).abs() <= 0.05);
            }
        }

        for dtype in [DType::F16, DType::BF16, DType::I8] {
            let lowp_matrix = make_tensor(&[2, 3], lowp_matrix_vals.clone(), dtype);
            let f32_row = make_tensor(&[3], f32_row_vals.clone(), DType::F32);
            assert!(!lowp_matrix.has_host_f32_data());
            add_cpu_binary_grads(&lowp_matrix, &f32_row, grad.view(), BinaryOp::Mul);
            assert!(!lowp_matrix.has_host_f32_data());
            let lowp_matrix_decoded = decoded_lowp(&lowp_matrix);
            assert_matrix_grad(&lowp_matrix, &f32_row_vals, grad_slice);
            assert_row_grad(&f32_row, &lowp_matrix_decoded, grad_slice);

            let f32_row = make_tensor(&[3], f32_row_vals.clone(), DType::F32);
            let lowp_matrix = make_tensor(&[2, 3], lowp_matrix_vals.clone(), dtype);
            assert!(!lowp_matrix.has_host_f32_data());
            add_cpu_binary_grads(&f32_row, &lowp_matrix, grad.view(), BinaryOp::Mul);
            assert!(!lowp_matrix.has_host_f32_data());
            let lowp_matrix_decoded = decoded_lowp(&lowp_matrix);
            assert_row_grad(&f32_row, &lowp_matrix_decoded, grad_slice);
            assert_matrix_grad(&lowp_matrix, &f32_row_vals, grad_slice);

            let f32_matrix = make_tensor(&[2, 3], f32_matrix_vals.clone(), DType::F32);
            let lowp_row = make_tensor(&[3], lowp_row_vals.clone(), dtype);
            assert!(!lowp_row.has_host_f32_data());
            add_cpu_binary_grads(&f32_matrix, &lowp_row, grad.view(), BinaryOp::Mul);
            assert!(!lowp_row.has_host_f32_data());
            let lowp_row_decoded = decoded_lowp(&lowp_row);
            assert_matrix_grad(&f32_matrix, &lowp_row_decoded, grad_slice);
            assert_row_grad(&lowp_row, &f32_matrix_vals, grad_slice);

            let lowp_row = make_tensor(&[3], lowp_row_vals.clone(), dtype);
            let f32_matrix = make_tensor(&[2, 3], f32_matrix_vals.clone(), DType::F32);
            assert!(!lowp_row.has_host_f32_data());
            add_cpu_binary_grads(&lowp_row, &f32_matrix, grad.view(), BinaryOp::Mul);
            assert!(!lowp_row.has_host_f32_data());
            let lowp_row_decoded = decoded_lowp(&lowp_row);
            assert_row_grad(&lowp_row, &f32_matrix_vals, grad_slice);
            assert_matrix_grad(&f32_matrix, &lowp_row_decoded, grad_slice);
        }
    }

    #[test]
    fn bf16_sum_no_grad_keeps_input_dtype() {
        let input = make_tensor(&[2, 2], vec![1.0, 2.0, 3.0, 4.0], DType::BF16);
        let out = no_grad(|| sum(&input));
        assert_eq!(input.dtype(), DType::BF16);
        assert_eq!(out.dtype(), DType::F32);
        assert_eq!(out.data_ref().first().copied(), Some(10.0));
    }

    #[test]
    fn f16_sum_no_grad_reads_native_storage_without_materializing_input_f32() {
        let input = make_tensor(&[2, 2], vec![1.0, 2.0, -3.0, 4.0], DType::F16);
        assert!(!input.has_host_f32_data());

        let out = no_grad(|| sum(&input));

        assert_eq!(input.dtype(), DType::F16);
        assert!(!input.has_host_f32_data());
        assert_eq!(out.dtype(), DType::F32);
        assert_eq!(out.data_ref().first().copied(), Some(4.0));
    }

    #[test]
    fn bf16_sum_no_grad_reads_native_storage_without_materializing_input_f32() {
        let input = make_tensor(&[2, 2], vec![1.0, 2.0, -3.0, 4.0], DType::BF16);
        assert!(!input.has_host_f32_data());

        let out = no_grad(|| sum(&input));

        assert_eq!(input.dtype(), DType::BF16);
        assert!(!input.has_host_f32_data());
        assert_eq!(out.dtype(), DType::F32);
        assert_eq!(out.data_ref().first().copied(), Some(4.0));
    }

    #[test]
    fn i8_sum_no_grad_reads_native_storage_without_materializing_input_f32() {
        let input = make_tensor(&[2, 2], vec![1.0, 2.0, -3.0, 4.0], DType::I8);
        let expected = match input.native_storage_owned() {
            TensorStorageOwned::I8(data, scale) => {
                assert!(!input.has_host_f32_data());
                data.iter().map(|&v| (v as f32) * scale).sum::<f32>()
            }
            _ => panic!("test input should use i8 native storage"),
        };

        let out = no_grad(|| sum(&input));

        assert_eq!(input.dtype(), DType::I8);
        assert!(!input.has_host_f32_data());
        assert_eq!(out.dtype(), DType::F32);
        assert!((out.data_ref().first().copied().unwrap() - expected).abs() <= 1e-6);
    }

    #[test]
    fn i8_add_no_grad_preserves_dtype() {
        let lhs = make_tensor(&[2], vec![1.0, -2.0], DType::I8);
        let rhs = make_tensor(&[2], vec![0.5, 3.0], DType::I8);

        let out = no_grad(|| lhs.clone() + rhs.clone());

        assert_eq!(lhs.dtype(), DType::I8);
        assert_eq!(rhs.dtype(), DType::I8);
        assert_eq!(out.dtype(), DType::I8);
        let vals = out.data_ref().iter().copied().collect::<Vec<_>>();
        assert!((vals[0] - 1.5).abs() <= 0.02);
        assert!((vals[1] - 1.0).abs() <= 0.02);
    }

    #[test]
    fn i8_mul_no_grad_preserves_dtype() {
        let lhs = make_tensor(&[2], vec![2.0, -1.5], DType::I8);
        let rhs = make_tensor(&[2], vec![0.25, 2.0], DType::I8);

        let out = no_grad(|| lhs * rhs);
        assert_eq!(out.dtype(), DType::I8);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_add_matches_cpu_reference() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        crate::ops::cuda::set_enabled(true);
        let lhs = make_tensor(
            &[16384],
            (0..16384).map(|i| i as f32 * 0.25).collect(),
            DType::F32,
        )
        .to_cuda();
        let rhs = make_tensor(
            &[16384],
            (0..16384).map(|i| -(i as f32) * 0.5).collect(),
            DType::F32,
        )
        .to_cuda();

        let out = no_grad(|| lhs.clone() + rhs.clone());
        assert!(out.is_cuda());

        crate::ops::cuda::set_enabled(false);
        let reference = no_grad(|| lhs.to_cpu() + rhs.to_cpu());

        let out_vals = out.data_ref().iter().copied().collect::<Vec<_>>();
        let ref_vals = reference.data_ref().iter().copied().collect::<Vec<_>>();
        for (got, expect) in out_vals.iter().zip(ref_vals.iter()) {
            assert!((got - expect).abs() < 1e-5, "got {got}, expect {expect}");
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_same_shape_lowp_binary_uses_native_forward_buffer_in_strict_mode() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        crate::ops::cuda::set_enabled(true);
        crate::autograd::set_strict_device_execution(true);

        for dtype in [DType::F16, DType::BF16, DType::I8] {
            let lhs = make_tensor(&[2, 2], vec![1.0, -2.0, 0.5, 3.0], dtype).to_cuda();
            let rhs = make_tensor(&[2, 2], vec![0.5, 3.0, -1.5, 0.25], dtype).to_cuda();

            assert!(!lhs.has_host_f32_data());
            assert!(!rhs.has_host_f32_data());
            assert!(lhs.cloned_cuda_native_lowp_buffer().is_some());
            assert!(rhs.cloned_cuda_native_lowp_buffer().is_some());

            let add_out = no_grad(|| lhs.clone() + rhs.clone());
            let sub_out = no_grad(|| lhs.clone() - rhs.clone());
            let mul_out = no_grad(|| lhs.clone() * rhs.clone());

            for out in [&add_out, &sub_out, &mul_out] {
                assert!(out.is_cuda());
                assert_eq!(out.dtype(), dtype);
                assert!(!out.has_host_f32_data());
                let inner = out.0.borrow();
                match dtype {
                    DType::F16 => {
                        assert!(inner.cuda_f32_data.is_none());
                        assert!(inner.cuda_f16_data.is_some());
                    }
                    DType::BF16 => {
                        assert!(inner.cuda_f32_data.is_none());
                        assert!(inner.cuda_bf16_data.is_some());
                    }
                    DType::I8 => {
                        assert!(inner.cuda_f32_data.is_none());
                        assert!(inner.cuda_i8_data.is_some());
                        assert!(inner.i8_scale.is_some());
                    }
                    DType::F32 => unreachable!(),
                }
            }

            let add_vals = add_out.data_ref().iter().copied().collect::<Vec<_>>();
            let sub_vals = sub_out.data_ref().iter().copied().collect::<Vec<_>>();
            let mul_vals = mul_out.data_ref().iter().copied().collect::<Vec<_>>();
            for (got, expected) in add_vals.iter().zip([1.5f32, 1.0, -1.0, 3.25]) {
                assert!((got - expected).abs() <= 0.04);
            }
            for (got, expected) in sub_vals.iter().zip([0.5f32, -5.0, 2.0, 2.75]) {
                assert!((got - expected).abs() <= 0.04);
            }
            for (got, expected) in mul_vals.iter().zip([0.5f32, -6.0, -0.75, 0.75]) {
                assert!((got - expected).abs() <= 0.04);
            }
        }

        crate::autograd::set_strict_device_execution(false);
        crate::ops::cuda::set_enabled(false);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_same_shape_lowp_binary_training_forward_outputs_f32_buffer() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        crate::ops::cuda::set_enabled(true);
        crate::autograd::set_strict_device_execution(true);

        for dtype in [DType::F16, DType::BF16, DType::I8] {
            let lhs = make_training_tensor_with_dtype(&[2, 2], vec![1.0, -2.0, 0.5, 3.0], dtype)
                .to_cuda();
            let rhs = make_training_tensor_with_dtype(&[2, 2], vec![0.5, 3.0, -1.5, 0.25], dtype)
                .to_cuda();
            assert!(lhs.cloned_cuda_native_lowp_buffer().is_some());
            assert!(rhs.cloned_cuda_native_lowp_buffer().is_some());

            let out = lhs.clone() * rhs.clone();
            assert!(out.is_cuda());
            assert_eq!(out.dtype(), DType::F32);
            assert!(!out.has_host_f32_data());
            assert!(out.cloned_cuda_f32_buffer().is_some());

            let loss = sum(&out);
            loss.backward();
            assert!(lhs.cloned_cuda_f32_grad().is_some());
            assert!(rhs.cloned_cuda_f32_grad().is_some());
            assert!(!lhs.has_host_f32_data());
            assert!(!rhs.has_host_f32_data());

            let lhs_grad = lhs.grad().expect("CUDA lowp lhs grad");
            let rhs_grad = rhs.grad().expect("CUDA lowp rhs grad");
            for (got, expected) in lhs_grad.iter().zip([0.5f32, 3.0, -1.5, 0.25]) {
                assert!(
                    (got - expected).abs() <= 0.04,
                    "{dtype:?} lhs grad got {got}, expected {expected}"
                );
            }
            for (got, expected) in rhs_grad.iter().zip([1.0f32, -2.0, 0.5, 3.0]) {
                assert!(
                    (got - expected).abs() <= 0.04,
                    "{dtype:?} rhs grad got {got}, expected {expected}"
                );
            }
        }

        crate::autograd::set_strict_device_execution(false);
        crate::ops::cuda::set_enabled(false);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_b1d_1hd_lowp_binary_forward_uses_resident_buffers_in_strict_mode() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        crate::ops::cuda::set_enabled(true);
        crate::autograd::set_strict_device_execution(true);

        let lhs = make_tensor(
            &[2, 1, 3],
            vec![1.0, -2.0, 0.5, 3.0, -4.0, 2.0],
            DType::BF16,
        )
        .to_cuda();
        let rhs = make_tensor(
            &[1, 4, 3],
            vec![
                0.5, 3.0, -1.5, 2.0, 0.25, -0.75, 1.25, -2.5, 4.0, -1.0, 0.0, 1.5,
            ],
            DType::F32,
        )
        .to_cuda();
        assert!(lhs.0.borrow().cuda_f32_data.is_none());
        assert!(lhs.0.borrow().cuda_bf16_data.is_some());

        let add_out = no_grad(|| lhs.clone() + rhs.clone());
        let sub_out = no_grad(|| lhs.clone() - rhs.clone());
        let mul_out = no_grad(|| lhs.clone() * rhs.clone());

        for out in [&add_out, &sub_out, &mul_out] {
            assert!(out.is_cuda());
            assert_eq!(out.dtype(), DType::F32);
            assert!(!out.has_host_f32_data());
            assert!(out.cloned_cuda_f32_buffer().is_some());
        }
        assert!(lhs.0.borrow().cuda_f32_data.is_none());
        assert!(lhs.0.borrow().cuda_bf16_data.is_some());

        crate::autograd::set_strict_device_execution(false);
        crate::ops::cuda::set_enabled(false);

        let add_vals = add_out.data_ref().iter().copied().collect::<Vec<_>>();
        let sub_vals = sub_out.data_ref().iter().copied().collect::<Vec<_>>();
        let mul_vals = mul_out.data_ref().iter().copied().collect::<Vec<_>>();
        let expected_add = [
            1.5f32, 1.0, -1.0, 3.0, -1.75, -0.25, 2.25, -4.5, 4.5, 0.0, -2.0, 2.0, 3.5, -1.0, 0.5,
            5.0, -3.75, 1.25, 4.25, -6.5, 6.0, 2.0, -4.0, 3.5,
        ];
        let expected_sub = [
            0.5f32, -5.0, 2.0, -1.0, -2.25, 1.25, -0.25, 0.5, -3.5, 2.0, -2.0, -1.0, 2.5, -7.0,
            3.5, 1.0, -4.25, 2.75, 1.75, -1.5, -2.0, 4.0, -4.0, 0.5,
        ];
        let expected_mul = [
            0.5f32, -6.0, -0.75, 2.0, -0.5, -0.375, 1.25, 5.0, 2.0, -1.0, -0.0, 0.75, 1.5, -12.0,
            -3.0, 6.0, -1.0, -1.5, 3.75, 10.0, 8.0, -3.0, -0.0, 3.0,
        ];
        for (got, expected) in add_vals.iter().zip(expected_add) {
            assert!(
                (got - expected).abs() <= 0.04,
                "add got {got}, expected {expected}"
            );
        }
        for (got, expected) in sub_vals.iter().zip(expected_sub) {
            assert!(
                (got - expected).abs() <= 0.04,
                "sub got {got}, expected {expected}"
            );
        }
        for (got, expected) in mul_vals.iter().zip(expected_mul) {
            assert!(
                (got - expected).abs() <= 0.04,
                "mul got {got}, expected {expected}"
            );
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_row_scalar_lowp_binary_forward_uses_resident_buffers_in_strict_mode() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        crate::ops::cuda::set_enabled(true);
        crate::autograd::set_strict_device_execution(true);

        let lhs = make_tensor(
            &[2, 2, 3],
            vec![
                1.0, -2.0, 0.5, 3.0, -4.0, 2.0, 0.25, 1.5, -1.0, 2.5, -0.5, 4.0,
            ],
            DType::BF16,
        )
        .to_cuda();
        let rhs = make_tensor(&[2, 2, 1], vec![0.5, -1.5, 2.0, -0.25], DType::F32).to_cuda();
        assert!(lhs.0.borrow().cuda_f32_data.is_none());
        assert!(lhs.0.borrow().cuda_bf16_data.is_some());

        let add_out = no_grad(|| lhs.clone() + rhs.clone());
        let sub_out = no_grad(|| lhs.clone() - rhs.clone());
        let mul_out = no_grad(|| lhs.clone() * rhs.clone());

        for out in [&add_out, &sub_out, &mul_out] {
            assert!(out.is_cuda());
            assert_eq!(out.dtype(), DType::F32);
            assert!(!out.has_host_f32_data());
            assert!(out.cloned_cuda_f32_buffer().is_some());
        }
        assert!(lhs.0.borrow().cuda_f32_data.is_none());
        assert!(lhs.0.borrow().cuda_bf16_data.is_some());

        crate::autograd::set_strict_device_execution(false);
        crate::ops::cuda::set_enabled(false);

        let add_vals = add_out.data_ref().iter().copied().collect::<Vec<_>>();
        let sub_vals = sub_out.data_ref().iter().copied().collect::<Vec<_>>();
        let mul_vals = mul_out.data_ref().iter().copied().collect::<Vec<_>>();
        let expected_add = [
            1.5f32, -1.5, 1.0, 1.5, -5.5, 0.5, 2.25, 3.5, 1.0, 2.25, -0.75, 3.75,
        ];
        let expected_sub = [
            0.5f32, -2.5, 0.0, 4.5, -2.5, 3.5, -1.75, -0.5, -3.0, 2.75, -0.25, 4.25,
        ];
        let expected_mul = [
            0.5f32, -1.0, 0.25, -4.5, 6.0, -3.0, 0.5, 3.0, -2.0, -0.625, 0.125, -1.0,
        ];
        for (got, expected) in add_vals.iter().zip(expected_add) {
            assert!(
                (got - expected).abs() <= 0.04,
                "add got {got}, expected {expected}"
            );
        }
        for (got, expected) in sub_vals.iter().zip(expected_sub) {
            assert!(
                (got - expected).abs() <= 0.04,
                "sub got {got}, expected {expected}"
            );
        }
        for (got, expected) in mul_vals.iter().zip(expected_mul) {
            assert!(
                (got - expected).abs() <= 0.04,
                "mul got {got}, expected {expected}"
            );
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_special_lowp_binary_forward_uses_typed_output_buffers_in_strict_mode() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        crate::ops::cuda::set_enabled(true);
        crate::autograd::set_strict_device_execution(true);

        for dtype in [DType::F16, DType::BF16, DType::I8] {
            let row_lhs = make_tensor(
                &[2, 2, 3],
                vec![
                    1.0, -2.0, 0.5, 3.0, -4.0, 2.0, 0.25, 1.5, -1.0, 2.5, -0.5, 4.0,
                ],
                dtype,
            )
            .to_cuda();
            let row_rhs = make_tensor(&[2, 2, 1], vec![0.5, -1.5, 2.0, -0.25], dtype).to_cuda();
            let row_out = no_grad(|| row_lhs.clone() * row_rhs.clone());
            assert_cuda_lowp_typed_output(&row_out, dtype, &[2, 2, 3]);

            let b1d_lhs =
                make_tensor(&[2, 1, 3], vec![1.0, -2.0, 0.5, 3.0, -4.0, 2.0], dtype).to_cuda();
            let h1_rhs = make_tensor(&[1, 4, 1], vec![0.5, 3.0, -1.5, 2.0], dtype).to_cuda();
            let b1d_1h1_out = no_grad(|| b1d_lhs.clone() + h1_rhs.clone());
            assert_cuda_lowp_typed_output(&b1d_1h1_out, dtype, &[2, 4, 3]);

            let hd_rhs = make_tensor(
                &[1, 4, 3],
                vec![
                    0.5, 3.0, -1.5, 2.0, 0.25, -0.75, 1.25, -2.5, 4.0, -1.0, 0.0, 1.5,
                ],
                dtype,
            )
            .to_cuda();
            let b1d_1hd_out = no_grad(|| b1d_lhs.clone() - hd_rhs.clone());
            assert_cuda_lowp_typed_output(&b1d_1hd_out, dtype, &[2, 4, 3]);

            let general_lhs =
                make_tensor(&[2, 1, 1, 3], vec![1.0, -2.0, 0.5, 3.0, -4.0, 2.0], dtype).to_cuda();
            let general_rhs = make_tensor(
                &[1, 4, 2, 3],
                vec![
                    0.5, 3.0, -1.5, 2.0, 0.25, -0.75, 1.25, -2.5, 4.0, -1.0, 0.0, 1.5, -0.5, 0.75,
                    2.25, -3.0, 1.0, -1.25, 0.125, -0.375, 0.875, 1.75, -2.25, 3.5,
                ],
                dtype,
            )
            .to_cuda();
            let general_out = no_grad(|| general_lhs.clone() * general_rhs.clone());
            assert_cuda_lowp_typed_output(&general_out, dtype, &[2, 4, 2, 3]);
        }

        crate::autograd::set_strict_device_execution(false);
        crate::ops::cuda::set_enabled(false);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_row_scalar_mixed_lowp_f32_backward_keeps_lowp_resident() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        crate::ops::cuda::set_enabled(true);
        crate::autograd::set_strict_device_execution(true);

        let lhs = make_training_tensor_with_dtype(
            &[2, 2, 3],
            vec![
                1.0, -2.0, 0.5, 3.0, -4.0, 2.0, 0.25, 1.5, -1.0, 2.5, -0.5, 4.0,
            ],
            DType::BF16,
        )
        .to_cuda();
        let rhs =
            make_training_tensor_with_dtype(&[2, 2, 1], vec![0.5, -1.5, 2.0, -0.25], DType::F32)
                .to_cuda();
        assert!(lhs.0.borrow().cuda_f32_data.is_none());
        assert!(lhs.0.borrow().cuda_bf16_data.is_some());

        let loss = sum(&(lhs.clone() * rhs.clone()));
        loss.backward();

        assert!(lhs.0.borrow().cuda_f32_data.is_none());
        assert!(lhs.0.borrow().cuda_bf16_data.is_some());
        assert!(lhs.cloned_cuda_f32_grad().is_some());
        assert!(rhs.cloned_cuda_f32_grad().is_some());
        assert!(!lhs.has_host_grad());
        assert!(!rhs.has_host_grad());

        crate::autograd::set_strict_device_execution(false);
        crate::ops::cuda::set_enabled(false);

        let lhs_grad = lhs.grad().expect("CUDA row-scalar mul lhs grad");
        let rhs_grad = rhs.grad().expect("CUDA row-scalar mul rhs grad");
        let expected_lhs = [
            0.5f32, 0.5, 0.5, -1.5, -1.5, -1.5, 2.0, 2.0, 2.0, -0.25, -0.25, -0.25,
        ];
        let expected_rhs = [-0.5f32, 1.0, 0.75, 6.0];
        for (got, expected) in lhs_grad.iter().zip(expected_lhs) {
            assert!(
                (got - expected).abs() <= 0.04,
                "lhs grad got {got}, expected {expected}"
            );
        }
        for (got, expected) in rhs_grad.iter().zip(expected_rhs) {
            assert!(
                (got - expected).abs() <= 0.08,
                "rhs grad got {got}, expected {expected}"
            );
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_row_scalar_mixed_lowp_f32_sub_backward_is_shape_only() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        crate::ops::cuda::set_enabled(true);
        crate::autograd::set_strict_device_execution(true);

        let lhs = make_training_tensor_with_dtype(
            &[2, 2, 3],
            vec![
                1.0, -2.0, 0.5, 3.0, -4.0, 2.0, 0.25, 1.5, -1.0, 2.5, -0.5, 4.0,
            ],
            DType::BF16,
        )
        .to_cuda();
        let rhs =
            make_training_tensor_with_dtype(&[2, 2, 1], vec![0.5, -1.5, 2.0, -0.25], DType::F32)
                .to_cuda();

        let loss = sum(&(lhs.clone() - rhs.clone()));
        loss.backward();

        assert!(lhs.0.borrow().cuda_f32_data.is_none());
        assert!(lhs.0.borrow().cuda_bf16_data.is_some());
        assert!(lhs.cloned_cuda_f32_grad().is_some());
        assert!(rhs.cloned_cuda_f32_grad().is_some());
        assert!(!lhs.has_host_grad());
        assert!(!rhs.has_host_grad());

        crate::autograd::set_strict_device_execution(false);
        crate::ops::cuda::set_enabled(false);

        let lhs_grad = lhs.grad().expect("CUDA row-scalar sub lhs grad");
        let rhs_grad = rhs.grad().expect("CUDA row-scalar sub rhs grad");
        for got in lhs_grad.iter() {
            assert!((*got - 1.0).abs() <= 1e-6);
        }
        for got in rhs_grad.iter() {
            assert!((*got + 3.0).abs() <= 1e-6);
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_lowp_sum_uses_resident_forward_buffer_in_strict_mode() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        crate::ops::cuda::set_enabled(true);
        crate::autograd::set_strict_device_execution(true);

        for dtype in [DType::F16, DType::BF16, DType::I8] {
            let input = make_tensor(&[2, 2], vec![1.0, -2.0, 0.5, 3.0], dtype).to_cuda();
            assert!(!input.has_host_f32_data());
            assert!(input.cloned_cuda_native_lowp_buffer().is_some());

            let out = no_grad(|| sum(&input));

            assert!(out.is_cuda());
            assert_eq!(out.dtype(), DType::F32);
            assert!(out.cloned_cuda_f32_buffer().is_some());
            assert!(!input.has_host_f32_data());
            let got = out.data_ref().first().copied().unwrap();
            assert!(
                (got - 2.5).abs() <= 0.05,
                "{dtype:?} CUDA sum got {got}, expected 2.5"
            );
        }

        crate::autograd::set_strict_device_execution(false);
        crate::ops::cuda::set_enabled(false);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_lowp_sum_backward_keeps_f32_grad_and_no_operand_f32_materialization() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        crate::ops::cuda::set_enabled(true);
        crate::autograd::set_strict_device_execution(true);

        for dtype in [DType::F16, DType::BF16, DType::I8] {
            let input = make_training_tensor_with_dtype(&[2, 2], vec![1.0, -2.0, 0.5, 3.0], dtype)
                .to_cuda();
            assert!(input.cloned_cuda_native_lowp_buffer().is_some());

            let loss = sum(&input);
            loss.backward();

            assert!(input.cloned_cuda_f32_grad().is_some());
            assert!(!input.has_host_f32_data());
            let grad = input.grad().expect("CUDA lowp sum grad");
            for got in grad.iter() {
                assert!((*got - 1.0).abs() <= 1e-6);
            }
        }

        crate::autograd::set_strict_device_execution(false);
        crate::ops::cuda::set_enabled(false);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_resident_i8_sum_matches_quantized_reference() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        let len = 1 << 16;
        let scale = 0.05f32;
        let values = (0..len)
            .map(|i| (((i * 31 + 7) % 255) - 127) as i8)
            .collect::<Vec<_>>();
        let expected = values.iter().map(|&v| v as i64).sum::<i64>() as f32 * scale;
        let buffer = crate::ops::cuda::upload_i8_storage(&values).expect("upload i8 sum input");

        let (_, got) =
            crate::ops::cuda::sum_i8_buffer(&buffer, scale).expect("CUDA resident i8 sum");

        assert!(
            (got[0] - expected).abs() <= 1e-5,
            "resident i8 sum kernel drifted: got {}, expected {}",
            got[0],
            expected
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_row_broadcast_lowp_binary_uses_resident_forward_buffer_in_strict_mode() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        crate::ops::cuda::set_enabled(true);
        crate::autograd::set_strict_device_execution(true);

        for dtype in [DType::F16, DType::BF16, DType::I8] {
            let matrix =
                make_tensor(&[2, 3], vec![1.0, -2.0, 0.5, 3.0, -4.0, 2.0], dtype).to_cuda();
            let row = make_tensor(&[3], vec![0.5, 3.0, -1.5], dtype).to_cuda();
            assert!(matrix.cloned_cuda_native_lowp_buffer().is_some());
            assert!(row.cloned_cuda_native_lowp_buffer().is_some());

            let add_out = no_grad(|| matrix.clone() + row.clone());
            let sub_out = no_grad(|| row.clone() - matrix.clone());
            let mul_out = no_grad(|| matrix.clone() * row.clone());

            for out in [&add_out, &sub_out, &mul_out] {
                assert!(out.is_cuda());
                assert_eq!(out.dtype(), dtype);
                assert_eq!(out.shape_vec(), vec![2, 3]);
                assert!(!out.has_host_f32_data());
                let inner = out.0.borrow();
                match dtype {
                    DType::F16 => {
                        assert!(inner.cuda_f32_data.is_none());
                        assert!(inner.cuda_f16_data.is_some());
                    }
                    DType::BF16 => {
                        assert!(inner.cuda_f32_data.is_none());
                        assert!(inner.cuda_bf16_data.is_some());
                    }
                    DType::I8 => {
                        assert!(inner.cuda_f32_data.is_none());
                        assert!(inner.cuda_i8_data.is_some());
                        assert!(inner.i8_scale.is_some());
                    }
                    DType::F32 => unreachable!(),
                }
            }

            let add_vals = add_out.data_ref().iter().copied().collect::<Vec<_>>();
            let sub_vals = sub_out.data_ref().iter().copied().collect::<Vec<_>>();
            let mul_vals = mul_out.data_ref().iter().copied().collect::<Vec<_>>();
            for (got, expected) in add_vals.iter().zip([1.5f32, 1.0, -1.0, 3.5, -1.0, 0.5]) {
                assert!((got - expected).abs() <= 0.05);
            }
            for (got, expected) in sub_vals.iter().zip([-0.5f32, 5.0, -2.0, -2.5, 7.0, -3.5]) {
                assert!((got - expected).abs() <= 0.08);
            }
            for (got, expected) in mul_vals.iter().zip([0.5f32, -6.0, -0.75, 1.5, -12.0, -3.0]) {
                assert!((got - expected).abs() <= 0.08);
            }
        }

        crate::autograd::set_strict_device_execution(false);
        crate::ops::cuda::set_enabled(false);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_row_broadcast_mixed_lowp_f32_uses_resident_forward_buffer_in_strict_mode() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        crate::ops::cuda::set_enabled(true);
        crate::autograd::set_strict_device_execution(true);

        let matrix =
            make_tensor(&[2, 3], vec![1.0, -2.0, 0.5, 3.0, -4.0, 2.0], DType::BF16).to_cuda();
        let row = make_tensor(&[3], vec![0.5, 3.0, -1.5], DType::F32).to_cuda();
        assert!(matrix.0.borrow().cuda_f32_data.is_none());
        assert!(matrix.0.borrow().cuda_bf16_data.is_some());

        let add_out = no_grad(|| matrix.clone() + row.clone());
        let mul_out = no_grad(|| matrix.clone() * row.clone());

        assert!(matrix.0.borrow().cuda_f32_data.is_none());
        assert!(matrix.0.borrow().cuda_bf16_data.is_some());
        for out in [&add_out, &mul_out] {
            assert!(out.is_cuda());
            assert_eq!(out.dtype(), DType::F32);
            assert_eq!(out.shape_vec(), vec![2, 3]);
            assert!(out.cloned_cuda_f32_buffer().is_some());
        }

        let add_vals = add_out.data_ref().iter().copied().collect::<Vec<_>>();
        let mul_vals = mul_out.data_ref().iter().copied().collect::<Vec<_>>();
        for (got, expected) in add_vals.iter().zip([1.5f32, 1.0, -1.0, 3.5, -1.0, 0.5]) {
            assert!((got - expected).abs() <= 0.05);
        }
        for (got, expected) in mul_vals.iter().zip([0.5f32, -6.0, -0.75, 1.5, -12.0, -3.0]) {
            assert!((got - expected).abs() <= 0.08);
        }

        crate::autograd::set_strict_device_execution(false);
        crate::ops::cuda::set_enabled(false);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_same_shape_mixed_lowp_f32_mul_backward_keeps_lowp_resident() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        crate::ops::cuda::set_enabled(true);
        crate::autograd::set_strict_device_execution(true);

        let lhs = make_training_tensor_with_dtype(
            &[2, 3],
            vec![1.0, -2.0, 0.5, 3.0, -4.0, 2.0],
            DType::BF16,
        )
        .to_cuda();
        let rhs = make_training_tensor_with_dtype(
            &[2, 3],
            vec![0.5, 3.0, -1.5, 2.0, 0.25, -0.75],
            DType::F32,
        )
        .to_cuda();
        assert!(lhs.0.borrow().cuda_f32_data.is_none());
        assert!(lhs.0.borrow().cuda_bf16_data.is_some());

        let loss = sum(&(lhs.clone() * rhs.clone()));
        loss.backward();

        assert!(lhs.0.borrow().cuda_f32_data.is_none());
        assert!(lhs.0.borrow().cuda_bf16_data.is_some());
        assert!(lhs.cloned_cuda_f32_grad().is_some());
        assert!(rhs.cloned_cuda_f32_grad().is_some());
        assert!(!lhs.has_host_grad());
        assert!(!rhs.has_host_grad());

        crate::autograd::set_strict_device_execution(false);
        crate::ops::cuda::set_enabled(false);

        let lhs_grad = lhs.grad().expect("CUDA same-shape mixed mul lhs grad");
        let rhs_grad = rhs.grad().expect("CUDA same-shape mixed mul rhs grad");
        for (got, expected) in lhs_grad.iter().zip([0.5f32, 3.0, -1.5, 2.0, 0.25, -0.75]) {
            assert!((got - expected).abs() <= 0.05);
        }
        for (got, expected) in rhs_grad.iter().zip([1.0f32, -2.0, 0.5, 3.0, -4.0, 2.0]) {
            assert!((got - expected).abs() <= 0.08);
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_row_broadcast_mixed_lowp_f32_mul_backward_keeps_lowp_resident() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        crate::ops::cuda::set_enabled(true);
        crate::autograd::set_strict_device_execution(true);

        let matrix = make_training_tensor_with_dtype(
            &[2, 3],
            vec![1.0, -2.0, 0.5, 3.0, -4.0, 2.0],
            DType::BF16,
        )
        .to_cuda();
        let row = make_training_tensor_with_dtype(&[3], vec![0.5, 3.0, -1.5], DType::F32).to_cuda();
        assert!(matrix.0.borrow().cuda_f32_data.is_none());
        assert!(matrix.0.borrow().cuda_bf16_data.is_some());

        let loss = sum(&(matrix.clone() * row.clone()));
        loss.backward();

        assert!(matrix.0.borrow().cuda_f32_data.is_none());
        assert!(matrix.0.borrow().cuda_bf16_data.is_some());
        assert!(matrix.cloned_cuda_f32_grad().is_some());
        assert!(row.cloned_cuda_f32_grad().is_some());
        assert!(!matrix.has_host_grad());
        assert!(!row.has_host_grad());

        crate::autograd::set_strict_device_execution(false);
        crate::ops::cuda::set_enabled(false);

        let matrix_grad = matrix.grad().expect("CUDA mixed row-broadcast matrix grad");
        let row_grad = row.grad().expect("CUDA mixed row-broadcast row grad");
        for (got, expected) in matrix_grad.iter().zip([0.5f32, 3.0, -1.5, 0.5, 3.0, -1.5]) {
            assert!((got - expected).abs() <= 0.05);
        }
        for (got, expected) in row_grad.iter().zip([4.0f32, -6.0, 2.5]) {
            assert!(
                (got - expected).abs() <= 0.08,
                "row grad got {got}, expected {expected}"
            );
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_mixed_lowp_f32_add_sub_backward_is_shape_only() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        crate::ops::cuda::set_enabled(true);
        crate::autograd::set_strict_device_execution(true);

        let lhs = make_training_tensor_with_dtype(&[2, 2], vec![1.0, -2.0, 0.5, 3.0], DType::BF16)
            .to_cuda();
        let rhs = make_training_tensor_with_dtype(&[2, 2], vec![0.5, 3.0, -1.5, 0.25], DType::F32)
            .to_cuda();
        assert!(lhs.0.borrow().cuda_f32_data.is_none());
        assert!(lhs.0.borrow().cuda_bf16_data.is_some());

        let loss = sum(&(lhs.clone() + rhs.clone()));
        loss.backward();

        assert!(lhs.0.borrow().cuda_f32_data.is_none());
        assert!(lhs.0.borrow().cuda_bf16_data.is_some());
        assert!(lhs.cloned_cuda_f32_grad().is_some());
        assert!(rhs.cloned_cuda_f32_grad().is_some());
        assert!(!lhs.has_host_grad());
        assert!(!rhs.has_host_grad());

        crate::autograd::set_strict_device_execution(false);
        crate::ops::cuda::set_enabled(false);

        let lhs_grad = lhs.grad().expect("CUDA add lhs grad");
        let rhs_grad = rhs.grad().expect("CUDA add rhs grad");
        for got in lhs_grad.iter().chain(rhs_grad.iter()) {
            assert!((*got - 1.0).abs() <= 1e-6);
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_row_broadcast_mixed_lowp_f32_sub_backward_is_shape_only() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        crate::ops::cuda::set_enabled(true);
        crate::autograd::set_strict_device_execution(true);

        let matrix = make_training_tensor_with_dtype(
            &[2, 3],
            vec![1.0, -2.0, 0.5, 3.0, -4.0, 2.0],
            DType::BF16,
        )
        .to_cuda();
        let row = make_training_tensor_with_dtype(&[3], vec![0.5, 3.0, -1.5], DType::F32).to_cuda();
        assert!(matrix.0.borrow().cuda_f32_data.is_none());
        assert!(matrix.0.borrow().cuda_bf16_data.is_some());

        let loss = sum(&(matrix.clone() - row.clone()));
        loss.backward();

        assert!(matrix.0.borrow().cuda_f32_data.is_none());
        assert!(matrix.0.borrow().cuda_bf16_data.is_some());
        assert!(matrix.cloned_cuda_f32_grad().is_some());
        assert!(row.cloned_cuda_f32_grad().is_some());
        assert!(!matrix.has_host_grad());
        assert!(!row.has_host_grad());

        crate::autograd::set_strict_device_execution(false);
        crate::ops::cuda::set_enabled(false);

        let matrix_grad = matrix.grad().expect("CUDA row-broadcast sub matrix grad");
        let row_grad = row.grad().expect("CUDA row-broadcast sub row grad");
        for got in matrix_grad.iter() {
            assert!((*got - 1.0).abs() <= 1e-6);
        }
        for got in row_grad.iter() {
            assert!((*got + 2.0).abs() <= 1e-6);
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_general_broadcast_mixed_lowp_f32_sub_backward_is_shape_only() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        crate::ops::cuda::set_enabled(true);
        crate::autograd::set_strict_device_execution(true);

        let lhs = make_training_tensor_with_dtype(
            &[2, 1, 3],
            vec![1.0, -2.0, 0.5, 3.0, -4.0, 2.0],
            DType::BF16,
        )
        .to_cuda();
        let rhs =
            make_training_tensor_with_dtype(&[1, 4, 1], vec![0.5, 3.0, -1.5, 2.0], DType::F32)
                .to_cuda();
        assert!(lhs.0.borrow().cuda_f32_data.is_none());
        assert!(lhs.0.borrow().cuda_bf16_data.is_some());

        let loss = sum(&(lhs.clone() - rhs.clone()));
        loss.backward();

        assert!(lhs.0.borrow().cuda_f32_data.is_none());
        assert!(lhs.0.borrow().cuda_bf16_data.is_some());
        assert!(lhs.cloned_cuda_f32_grad().is_some());
        assert!(rhs.cloned_cuda_f32_grad().is_some());
        assert!(!lhs.has_host_grad());
        assert!(!rhs.has_host_grad());

        crate::autograd::set_strict_device_execution(false);
        crate::ops::cuda::set_enabled(false);

        let lhs_grad = lhs.grad().expect("CUDA general-broadcast sub lhs grad");
        let rhs_grad = rhs.grad().expect("CUDA general-broadcast sub rhs grad");
        for got in lhs_grad.iter() {
            assert!((*got - 4.0).abs() <= 1e-6);
        }
        for got in rhs_grad.iter() {
            assert!((*got + 6.0).abs() <= 1e-6);
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_general_broadcast_non_special_mixed_lowp_f32_add_backward_is_shape_only() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        crate::ops::cuda::set_enabled(true);
        crate::autograd::set_strict_device_execution(true);

        let lhs = make_training_tensor_with_dtype(
            &[2, 1, 1, 3],
            vec![1.0, -2.0, 0.5, 3.0, -4.0, 2.0],
            DType::BF16,
        )
        .to_cuda();
        let rhs = make_training_tensor_with_dtype(
            &[1, 4, 2, 3],
            vec![
                0.5, 3.0, -1.5, 2.0, 0.25, -0.75, 1.25, -2.5, 4.0, -1.0, 0.0, 1.5, -3.0, 0.75,
                2.25, 1.0, -1.25, 3.5, 0.125, -0.5, 1.75, 2.5, -2.0, 0.625,
            ],
            DType::F32,
        )
        .to_cuda();
        assert!(lhs.0.borrow().cuda_f32_data.is_none());
        assert!(lhs.0.borrow().cuda_bf16_data.is_some());

        let loss = sum(&(lhs.clone() + rhs.clone()));
        loss.backward();

        assert!(lhs.0.borrow().cuda_f32_data.is_none());
        assert!(lhs.0.borrow().cuda_bf16_data.is_some());
        assert!(lhs.cloned_cuda_f32_grad().is_some());
        assert!(rhs.cloned_cuda_f32_grad().is_some());
        assert!(!lhs.has_host_grad());
        assert!(!rhs.has_host_grad());

        crate::autograd::set_strict_device_execution(false);
        crate::ops::cuda::set_enabled(false);

        let lhs_grad = lhs.grad().expect("CUDA non-special broadcast add lhs grad");
        let rhs_grad = rhs.grad().expect("CUDA non-special broadcast add rhs grad");
        for got in lhs_grad.iter() {
            assert!((*got - 8.0).abs() <= 1e-6);
        }
        for got in rhs_grad.iter() {
            assert!((*got - 2.0).abs() <= 1e-6);
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_b1d_1hd_broadcast_mixed_lowp_f32_add_backward_is_shape_only() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        crate::ops::cuda::set_enabled(true);
        crate::autograd::set_strict_device_execution(true);

        let lhs = make_training_tensor_with_dtype(
            &[2, 1, 3],
            vec![1.0, -2.0, 0.5, 3.0, -4.0, 2.0],
            DType::BF16,
        )
        .to_cuda();
        let rhs = make_training_tensor_with_dtype(
            &[1, 4, 3],
            vec![
                0.5, 3.0, -1.5, 2.0, 0.25, -0.75, 1.25, -2.5, 4.0, -1.0, 0.0, 1.5,
            ],
            DType::F32,
        )
        .to_cuda();
        assert!(lhs.0.borrow().cuda_f32_data.is_none());
        assert!(lhs.0.borrow().cuda_bf16_data.is_some());

        let loss = sum(&(lhs.clone() + rhs.clone()));
        loss.backward();

        assert!(lhs.0.borrow().cuda_f32_data.is_none());
        assert!(lhs.0.borrow().cuda_bf16_data.is_some());
        assert!(lhs.cloned_cuda_f32_grad().is_some());
        assert!(rhs.cloned_cuda_f32_grad().is_some());
        assert!(!lhs.has_host_grad());
        assert!(!rhs.has_host_grad());

        crate::autograd::set_strict_device_execution(false);
        crate::ops::cuda::set_enabled(false);

        let lhs_grad = lhs.grad().expect("CUDA b1d/1hd broadcast add lhs grad");
        let rhs_grad = rhs.grad().expect("CUDA b1d/1hd broadcast add rhs grad");
        for got in lhs_grad.iter() {
            assert!((*got - 4.0).abs() <= 1e-6);
        }
        for got in rhs_grad.iter() {
            assert!((*got - 2.0).abs() <= 1e-6);
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_b1d_1hd_mixed_lowp_f32_mul_backward_keeps_lowp_resident() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        crate::ops::cuda::set_enabled(true);
        crate::autograd::set_strict_device_execution(true);

        let lhs = make_training_tensor_with_dtype(
            &[2, 1, 3],
            vec![1.0, -2.0, 0.5, 3.0, -4.0, 2.0],
            DType::BF16,
        )
        .to_cuda();
        let rhs = make_training_tensor_with_dtype(
            &[1, 4, 3],
            vec![
                0.5, 3.0, -1.5, 2.0, 0.25, -0.75, 1.25, -2.5, 4.0, -1.0, 0.0, 1.5,
            ],
            DType::F32,
        )
        .to_cuda();
        assert!(lhs.0.borrow().cuda_f32_data.is_none());
        assert!(lhs.0.borrow().cuda_bf16_data.is_some());

        let loss = sum(&(lhs.clone() * rhs.clone()));
        loss.backward();

        assert!(lhs.0.borrow().cuda_f32_data.is_none());
        assert!(lhs.0.borrow().cuda_bf16_data.is_some());
        assert!(lhs.cloned_cuda_f32_grad().is_some());
        assert!(rhs.cloned_cuda_f32_grad().is_some());
        assert!(!lhs.has_host_grad());
        assert!(!rhs.has_host_grad());

        crate::autograd::set_strict_device_execution(false);
        crate::ops::cuda::set_enabled(false);

        let lhs_grad = lhs.grad().expect("CUDA b1d/1hd mul lhs grad");
        let rhs_grad = rhs.grad().expect("CUDA b1d/1hd mul rhs grad");
        for (got, expected) in lhs_grad.iter().zip([2.75f32, 0.75, 3.25, 2.75, 0.75, 3.25]) {
            assert!(
                (got - expected).abs() <= 0.05,
                "lhs grad got {got}, expected {expected}"
            );
        }
        for (got, expected) in rhs_grad.iter().zip([4.0f32, -6.0, 2.5].into_iter().cycle()) {
            assert!(
                (got - expected).abs() <= 0.08,
                "rhs grad got {got}, expected {expected}"
            );
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_general_broadcast_mixed_lowp_f32_mul_backward_keeps_lowp_resident() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        crate::ops::cuda::set_enabled(true);
        crate::autograd::set_strict_device_execution(true);

        let lhs = make_training_tensor_with_dtype(
            &[2, 1, 1, 3],
            vec![1.0, -2.0, 0.5, 3.0, -4.0, 2.0],
            DType::BF16,
        )
        .to_cuda();
        let rhs = make_training_tensor_with_dtype(
            &[1, 4, 2, 3],
            vec![
                0.5, 3.0, -1.5, 2.0, 0.25, -0.75, 1.25, -2.5, 4.0, -1.0, 0.0, 1.5, 0.75, -1.25,
                2.25, 3.5, -0.5, 1.0, -2.0, 0.5, 0.25, 1.75, -3.0, 2.5,
            ],
            DType::F32,
        )
        .to_cuda();
        assert!(lhs.0.borrow().cuda_f32_data.is_none());
        assert!(lhs.0.borrow().cuda_bf16_data.is_some());

        let loss = sum(&(lhs.clone() * rhs.clone()));
        loss.backward();

        assert!(lhs.0.borrow().cuda_f32_data.is_none());
        assert!(lhs.0.borrow().cuda_bf16_data.is_some());
        assert!(lhs.cloned_cuda_f32_grad().is_some());
        assert!(rhs.cloned_cuda_f32_grad().is_some());
        assert!(!lhs.has_host_grad());
        assert!(!rhs.has_host_grad());

        crate::autograd::set_strict_device_execution(false);
        crate::ops::cuda::set_enabled(false);

        let lhs_grad = lhs.grad().expect("CUDA general-broadcast mul lhs grad");
        let rhs_grad = rhs.grad().expect("CUDA general-broadcast mul rhs grad");
        for (got, expected) in lhs_grad.iter().zip([6.75f32, -3.5, 9.25, 6.75, -3.5, 9.25]) {
            assert!(
                (got - expected).abs() <= 0.08,
                "lhs grad got {got}, expected {expected}"
            );
        }
        for (got, expected) in rhs_grad.iter().zip([4.0f32, -6.0, 2.5].into_iter().cycle()) {
            assert!(
                (got - expected).abs() <= 0.08,
                "rhs grad got {got}, expected {expected}"
            );
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_b1d_1h1_mixed_lowp_f32_mul_backward_keeps_lowp_resident() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        crate::ops::cuda::set_enabled(true);
        crate::autograd::set_strict_device_execution(true);

        let lhs = make_training_tensor_with_dtype(
            &[2, 1, 3],
            vec![1.0, -2.0, 0.5, 3.0, -4.0, 2.0],
            DType::BF16,
        )
        .to_cuda();
        let rhs =
            make_training_tensor_with_dtype(&[1, 4, 1], vec![0.5, 3.0, -1.5, 2.0], DType::F32)
                .to_cuda();
        assert!(lhs.0.borrow().cuda_f32_data.is_none());
        assert!(lhs.0.borrow().cuda_bf16_data.is_some());

        let loss = sum(&(lhs.clone() * rhs.clone()));
        loss.backward();

        assert!(lhs.0.borrow().cuda_f32_data.is_none());
        assert!(lhs.0.borrow().cuda_bf16_data.is_some());
        assert!(lhs.cloned_cuda_f32_grad().is_some());
        assert!(rhs.cloned_cuda_f32_grad().is_some());
        assert!(!lhs.has_host_grad());
        assert!(!rhs.has_host_grad());

        crate::autograd::set_strict_device_execution(false);
        crate::ops::cuda::set_enabled(false);

        let lhs_grad = lhs.grad().expect("CUDA b1d/1h1 mul lhs grad");
        let rhs_grad = rhs.grad().expect("CUDA b1d/1h1 mul rhs grad");
        for got in lhs_grad.iter() {
            assert!((*got - 4.0).abs() <= 0.05);
        }
        for got in rhs_grad.iter() {
            assert!((*got - 0.5).abs() <= 0.08);
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_scalar_broadcast_mixed_lowp_f32_sub_backward_is_shape_only() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        crate::ops::cuda::set_enabled(true);
        crate::autograd::set_strict_device_execution(true);

        let lhs = make_training_tensor_with_dtype(&[2, 2], vec![1.0, -2.0, 0.5, 3.0], DType::BF16)
            .to_cuda();
        let rhs = make_training_tensor_with_dtype(&[1], vec![0.5], DType::F32).to_cuda();
        assert!(lhs.0.borrow().cuda_f32_data.is_none());
        assert!(lhs.0.borrow().cuda_bf16_data.is_some());

        let loss = sum(&(lhs.clone() - rhs.clone()));
        loss.backward();

        assert!(lhs.0.borrow().cuda_f32_data.is_none());
        assert!(lhs.0.borrow().cuda_bf16_data.is_some());
        assert!(lhs.cloned_cuda_f32_grad().is_some());
        assert!(rhs.cloned_cuda_f32_grad().is_some());
        assert!(!lhs.has_host_grad());
        assert!(!rhs.has_host_grad());

        crate::autograd::set_strict_device_execution(false);
        crate::ops::cuda::set_enabled(false);

        let lhs_grad = lhs.grad().expect("CUDA scalar-broadcast sub lhs grad");
        let rhs_grad = rhs.grad().expect("CUDA scalar-broadcast sub rhs grad");
        for got in lhs_grad.iter() {
            assert!((*got - 1.0).abs() <= 1e-6);
        }
        assert!((rhs_grad[IxDyn(&[0])] + 4.0).abs() <= 1e-6);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_scalar_broadcast_mixed_lowp_f32_mul_backward_keeps_lowp_resident() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        crate::ops::cuda::set_enabled(true);
        crate::autograd::set_strict_device_execution(true);

        let lhs = make_training_tensor_with_dtype(&[2, 2], vec![1.0, -2.0, 0.5, 3.0], DType::BF16)
            .to_cuda();
        let rhs = make_training_tensor_with_dtype(&[1], vec![0.5], DType::F32).to_cuda();
        assert!(lhs.0.borrow().cuda_f32_data.is_none());
        assert!(lhs.0.borrow().cuda_bf16_data.is_some());

        let loss = sum(&(lhs.clone() * rhs.clone()));
        loss.backward();

        assert!(lhs.0.borrow().cuda_f32_data.is_none());
        assert!(lhs.0.borrow().cuda_bf16_data.is_some());
        assert!(lhs.cloned_cuda_f32_grad().is_some());
        assert!(rhs.cloned_cuda_f32_grad().is_some());
        assert!(!lhs.has_host_grad());
        assert!(!rhs.has_host_grad());

        crate::autograd::set_strict_device_execution(false);
        crate::ops::cuda::set_enabled(false);

        let lhs_grad = lhs.grad().expect("CUDA scalar-broadcast mul lhs grad");
        let rhs_grad = rhs.grad().expect("CUDA scalar-broadcast mul rhs grad");
        for got in lhs_grad.iter() {
            assert!((*got - 0.5).abs() <= 0.05);
        }
        assert!((rhs_grad[IxDyn(&[0])] - 2.5).abs() <= 0.08);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_row_broadcast_lowp_mul_backward_outputs_f32_grads() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        crate::ops::cuda::set_enabled(true);
        crate::autograd::set_strict_device_execution(true);

        for dtype in [DType::F16, DType::BF16, DType::I8] {
            let matrix = make_training_tensor_with_dtype(
                &[2, 3],
                vec![1.0, -2.0, 0.5, 3.0, -4.0, 2.0],
                dtype,
            )
            .to_cuda();
            let row = make_training_tensor_with_dtype(&[3], vec![0.5, 3.0, -1.5], dtype).to_cuda();

            let loss = sum(&(matrix.clone() * row.clone()));
            loss.backward();
            assert!(matrix.cloned_cuda_f32_grad().is_some());
            assert!(row.cloned_cuda_f32_grad().is_some());
            assert!(!matrix.has_host_f32_data());
            assert!(!row.has_host_f32_data());

            let matrix_grad = matrix.grad().expect("CUDA row-broadcast matrix grad");
            let row_grad = row.grad().expect("CUDA row-broadcast row grad");
            for (got, expected) in matrix_grad.iter().zip([0.5f32, 3.0, -1.5, 0.5, 3.0, -1.5]) {
                assert!((got - expected).abs() <= 0.05);
            }
            for (got, expected) in row_grad.iter().zip([4.0f32, -6.0, 2.5]) {
                assert!(
                    (got - expected).abs() <= 0.08,
                    "{dtype:?} row grad got {got}, expected {expected}"
                );
            }
        }

        crate::autograd::set_strict_device_execution(false);
        crate::ops::cuda::set_enabled(false);
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore]
    fn cuda_lowp_binary_forward_perf_smoke() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        let len = 1 << 20;
        let lhs = (0..len)
            .map(|i| (i as f32 % 251.0) / 37.0 - 3.0)
            .collect::<Vec<_>>();
        let rhs = (0..len)
            .map(|i| ((i * 17) as f32 % 241.0) / 41.0 - 2.5)
            .collect::<Vec<_>>();
        let lhs_f32 = crate::ops::cuda::upload_f32(&lhs).expect("upload lhs f32");
        let rhs_f32 = crate::ops::cuda::upload_f32(&rhs).expect("upload rhs f32");

        let _f32_warmup = crate::ops::cuda::binary_f32_buffer(
            &lhs_f32,
            &rhs_f32,
            crate::ops::cuda::BinaryOp::Mul,
        )
        .expect("warm up CUDA f32 binary perf path");
        crate::ops::cuda::synchronize().expect("sync f32 binary warmup");

        let start = std::time::Instant::now();
        let f32_out = crate::ops::cuda::binary_f32_buffer(
            &lhs_f32,
            &rhs_f32,
            crate::ops::cuda::BinaryOp::Mul,
        )
        .expect("CUDA f32 binary perf path");
        crate::ops::cuda::synchronize().expect("sync f32 binary");
        let f32_us = start.elapsed().as_secs_f64() * 1.0e6;
        let f32_ref = crate::ops::cuda::download_f32(&f32_out).expect("download f32 binary");

        let lhs_f16 = lhs
            .iter()
            .map(|&v| f16::from_f32(v).to_bits())
            .collect::<Vec<_>>();
        let rhs_f16 = rhs
            .iter()
            .map(|&v| f16::from_f32(v).to_bits())
            .collect::<Vec<_>>();
        let lhs_f16_buf = crate::ops::cuda::upload_u16_storage(&lhs_f16).expect("upload lhs f16");
        let rhs_f16_buf = crate::ops::cuda::upload_u16_storage(&rhs_f16).expect("upload rhs f16");
        let start = std::time::Instant::now();
        let f16_out = crate::ops::cuda::binary_f16_buffer_no_host(
            &lhs_f16_buf,
            &rhs_f16_buf,
            crate::ops::cuda::BinaryOp::Mul,
        )
        .expect("CUDA resident f16 binary perf path");
        crate::ops::cuda::synchronize().expect("sync f16 binary");
        let f16_us = start.elapsed().as_secs_f64() * 1.0e6;
        let f16_vals = crate::ops::cuda::download_f32(&f16_out).expect("download f16 binary");

        let lhs_bf16 = lhs
            .iter()
            .map(|&v| bf16::from_f32(v).to_bits())
            .collect::<Vec<_>>();
        let rhs_bf16 = rhs
            .iter()
            .map(|&v| bf16::from_f32(v).to_bits())
            .collect::<Vec<_>>();
        let lhs_bf16_buf =
            crate::ops::cuda::upload_u16_storage(&lhs_bf16).expect("upload lhs bf16");
        let rhs_bf16_buf =
            crate::ops::cuda::upload_u16_storage(&rhs_bf16).expect("upload rhs bf16");
        let start = std::time::Instant::now();
        let bf16_out = crate::ops::cuda::binary_bf16_buffer_no_host(
            &lhs_bf16_buf,
            &rhs_bf16_buf,
            crate::ops::cuda::BinaryOp::Mul,
        )
        .expect("CUDA resident bf16 binary perf path");
        crate::ops::cuda::synchronize().expect("sync bf16 binary");
        let bf16_us = start.elapsed().as_secs_f64() * 1.0e6;
        let bf16_vals = crate::ops::cuda::download_f32(&bf16_out).expect("download bf16 binary");

        let lhs_i8 = lhs
            .iter()
            .map(|&v| (v / 0.05).round().clamp(-127.0, 127.0) as i8)
            .collect::<Vec<_>>();
        let rhs_i8 = rhs
            .iter()
            .map(|&v| (v / 0.05).round().clamp(-127.0, 127.0) as i8)
            .collect::<Vec<_>>();
        let lhs_i8_buf = crate::ops::cuda::upload_i8_storage(&lhs_i8).expect("upload lhs i8");
        let rhs_i8_buf = crate::ops::cuda::upload_i8_storage(&rhs_i8).expect("upload rhs i8");
        let start = std::time::Instant::now();
        let i8_out = crate::ops::cuda::binary_i8_buffer_no_host(
            &lhs_i8_buf,
            0.05,
            &rhs_i8_buf,
            0.05,
            crate::ops::cuda::BinaryOp::Mul,
        )
        .expect("CUDA resident i8 binary perf path");
        crate::ops::cuda::synchronize().expect("sync i8 binary");
        let i8_us = start.elapsed().as_secs_f64() * 1.0e6;
        let i8_vals = crate::ops::cuda::download_f32(&i8_out).expect("download i8 binary");

        let _i8_typed_warmup = crate::ops::cuda::binary_i8_typed_output_buffer_no_host(
            &lhs_i8_buf,
            0.05,
            &rhs_i8_buf,
            0.05,
            crate::ops::cuda::BinaryOp::Mul,
        )
        .expect("warm up CUDA resident typed-output i8 binary");
        crate::ops::cuda::synchronize().expect("sync typed-output i8 binary warmup");
        let start = std::time::Instant::now();
        let (i8_typed_out, i8_typed_scale) =
            crate::ops::cuda::binary_i8_typed_output_buffer_no_host(
                &lhs_i8_buf,
                0.05,
                &rhs_i8_buf,
                0.05,
                crate::ops::cuda::BinaryOp::Mul,
            )
            .expect("CUDA resident typed-output i8 binary perf path");
        crate::ops::cuda::synchronize().expect("sync typed-output i8 binary");
        let i8_typed_us = start.elapsed().as_secs_f64() * 1.0e6;
        let i8_typed_vals = crate::ops::cuda::download_i8_storage(&i8_typed_out)
            .expect("download typed-output i8 binary")
            .into_iter()
            .map(|value| value as f32 * i8_typed_scale)
            .collect::<Vec<_>>();

        let max_err = |vals: &[f32]| {
            vals.iter()
                .zip(f32_ref.iter())
                .map(|(&got, &expect)| (got - expect).abs())
                .fold(0.0f32, f32::max)
        };

        println!(
            "cuda lowp binary mul len={len}: f32={f32_us:.1}us, f16={f16_us:.1}us max_err={:.5}, bf16={bf16_us:.1}us max_err={:.5}, i8_f32_out={i8_us:.1}us max_err={:.5}, i8_typed_out={i8_typed_us:.1}us kernel_requant_err={:.5} total_err={:.5}",
            max_err(&f16_vals),
            max_err(&bf16_vals),
            max_err(&i8_vals),
            i8_typed_vals
                .iter()
                .zip(i8_vals.iter())
                .map(|(&got, &expect)| (got - expect).abs())
                .fold(0.0f32, f32::max),
            max_err(&i8_typed_vals)
        );

        let start = std::time::Instant::now();
        let (_, f32_sum) = crate::ops::cuda::sum_f32(&lhs_f32).expect("CUDA f32 sum perf path");
        crate::ops::cuda::synchronize().expect("sync f32 sum");
        let f32_sum_us = start.elapsed().as_secs_f64() * 1.0e6;
        let f32_sum = f32_sum[0];

        let start = std::time::Instant::now();
        let (_, f16_sum) =
            crate::ops::cuda::sum_f16_buffer(&lhs_f16_buf).expect("CUDA f16 sum perf path");
        crate::ops::cuda::synchronize().expect("sync f16 sum");
        let f16_sum_us = start.elapsed().as_secs_f64() * 1.0e6;
        let f16_sum = f16_sum[0];

        let start = std::time::Instant::now();
        let (_, bf16_sum) =
            crate::ops::cuda::sum_bf16_buffer(&lhs_bf16_buf).expect("CUDA bf16 sum perf path");
        crate::ops::cuda::synchronize().expect("sync bf16 sum");
        let bf16_sum_us = start.elapsed().as_secs_f64() * 1.0e6;
        let bf16_sum = bf16_sum[0];

        let start = std::time::Instant::now();
        let (_, i8_sum) =
            crate::ops::cuda::sum_i8_buffer(&lhs_i8_buf, 0.05).expect("CUDA i8 sum perf path");
        crate::ops::cuda::synchronize().expect("sync i8 sum");
        let i8_sum_us = start.elapsed().as_secs_f64() * 1.0e6;
        let i8_sum = i8_sum[0];
        let i8_sum_quant_ref = lhs_i8.iter().map(|&v| v as i64).sum::<i64>() as f32 * 0.05;

        println!(
            "cuda lowp sum len={len}: f32={f32_sum_us:.1}us, f16={f16_sum_us:.1}us abs_err={:.5}, bf16={bf16_sum_us:.1}us abs_err={:.5}, i8={i8_sum_us:.1}us kernel_err={:.5} quant_err={:.5}",
            (f16_sum - f32_sum).abs(),
            (bf16_sum - f32_sum).abs(),
            (i8_sum - i8_sum_quant_ref).abs(),
            (i8_sum - f32_sum).abs()
        );

        let grad = (0..len)
            .map(|i| ((i * 7) as f32 % 127.0) / 97.0 - 0.5)
            .collect::<Vec<_>>();
        let grad_buf = crate::ops::cuda::upload_f32(&grad).expect("upload grad f32");

        let start = std::time::Instant::now();
        let (f32_lhs_grad, f32_rhs_grad) = crate::ops::cuda::binary_backward_f32_buffers(
            &lhs_f32,
            &rhs_f32,
            &grad_buf,
            crate::ops::cuda::BinaryOp::Mul,
        )
        .expect("CUDA f32 binary backward perf path");
        crate::ops::cuda::synchronize().expect("sync f32 backward");
        let f32_backward_us = start.elapsed().as_secs_f64() * 1.0e6;
        let f32_lhs_grad =
            crate::ops::cuda::download_f32(&f32_lhs_grad).expect("download f32 lhs grad");
        let f32_rhs_grad =
            crate::ops::cuda::download_f32(&f32_rhs_grad).expect("download f32 rhs grad");

        let start = std::time::Instant::now();
        let f16_lhs_grad = crate::ops::cuda::mul_grad_f16_buffer_no_host(&grad_buf, &rhs_f16_buf)
            .expect("CUDA f16 lhs grad perf path");
        let f16_rhs_grad = crate::ops::cuda::mul_grad_f16_buffer_no_host(&grad_buf, &lhs_f16_buf)
            .expect("CUDA f16 rhs grad perf path");
        crate::ops::cuda::synchronize().expect("sync f16 backward");
        let f16_backward_us = start.elapsed().as_secs_f64() * 1.0e6;
        let f16_lhs_grad =
            crate::ops::cuda::download_f32(&f16_lhs_grad).expect("download f16 lhs grad");
        let f16_rhs_grad =
            crate::ops::cuda::download_f32(&f16_rhs_grad).expect("download f16 rhs grad");

        let start = std::time::Instant::now();
        let bf16_lhs_grad =
            crate::ops::cuda::mul_grad_bf16_buffer_no_host(&grad_buf, &rhs_bf16_buf)
                .expect("CUDA bf16 lhs grad perf path");
        let bf16_rhs_grad =
            crate::ops::cuda::mul_grad_bf16_buffer_no_host(&grad_buf, &lhs_bf16_buf)
                .expect("CUDA bf16 rhs grad perf path");
        crate::ops::cuda::synchronize().expect("sync bf16 backward");
        let bf16_backward_us = start.elapsed().as_secs_f64() * 1.0e6;
        let bf16_lhs_grad =
            crate::ops::cuda::download_f32(&bf16_lhs_grad).expect("download bf16 lhs grad");
        let bf16_rhs_grad =
            crate::ops::cuda::download_f32(&bf16_rhs_grad).expect("download bf16 rhs grad");

        let start = std::time::Instant::now();
        let i8_lhs_grad =
            crate::ops::cuda::mul_grad_i8_buffer_no_host(&grad_buf, &rhs_i8_buf, 0.05)
                .expect("CUDA i8 lhs grad perf path");
        let i8_rhs_grad =
            crate::ops::cuda::mul_grad_i8_buffer_no_host(&grad_buf, &lhs_i8_buf, 0.05)
                .expect("CUDA i8 rhs grad perf path");
        crate::ops::cuda::synchronize().expect("sync i8 backward");
        let i8_backward_us = start.elapsed().as_secs_f64() * 1.0e6;
        let i8_lhs_grad =
            crate::ops::cuda::download_f32(&i8_lhs_grad).expect("download i8 lhs grad");
        let i8_rhs_grad =
            crate::ops::cuda::download_f32(&i8_rhs_grad).expect("download i8 rhs grad");

        let max_err_against = |vals: &[f32], reference: &[f32]| {
            vals.iter()
                .zip(reference.iter())
                .map(|(&got, &expect)| (got - expect).abs())
                .fold(0.0f32, f32::max)
        };
        let max_pair_err = |lhs_vals: &[f32], rhs_vals: &[f32]| {
            max_err_against(lhs_vals, &f32_lhs_grad).max(max_err_against(rhs_vals, &f32_rhs_grad))
        };

        println!(
            "cuda lowp binary mul backward len={len}: f32={f32_backward_us:.1}us, f16={f16_backward_us:.1}us max_err={:.5}, bf16={bf16_backward_us:.1}us max_err={:.5}, i8={i8_backward_us:.1}us max_err={:.5}",
            max_pair_err(&f16_lhs_grad, &f16_rhs_grad),
            max_pair_err(&bf16_lhs_grad, &bf16_rhs_grad),
            max_pair_err(&i8_lhs_grad, &i8_rhs_grad)
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_empty_binary_and_sum_stay_stable_in_strict_mode() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        crate::ops::cuda::set_enabled(true);
        crate::autograd::set_strict_device_execution(true);

        let lhs = make_tensor(&[0, 4], vec![], DType::BF16).to_cuda();
        let rhs = make_tensor(&[0, 4], vec![], DType::BF16).to_cuda();
        let add_out = no_grad(|| lhs.clone() + rhs.clone());
        assert!(add_out.is_cuda());
        assert_eq!(add_out.dtype(), DType::BF16);
        assert_eq!(add_out.shape_vec(), vec![0, 4]);
        assert_eq!(add_out.len(), 0);

        let sum_out = no_grad(|| sum(&add_out));
        assert!(sum_out.is_cuda());
        assert_eq!(sum_out.data_ref().first().copied(), Some(0.0));

        let lhs_train = make_training_tensor(&[0, 4], vec![]).to_cuda();
        let rhs_train = make_training_tensor(&[0, 4], vec![]).to_cuda();
        let train_out = lhs_train.clone() + rhs_train.clone();
        assert!(train_out.is_cuda());
        assert_eq!(train_out.len(), 0);
        let loss = sum(&train_out);
        assert!(loss.is_cuda());
        loss.backward();

        let lhs_grad = lhs_train
            .grad()
            .expect("empty CUDA lhs grad should be recorded");
        let rhs_grad = rhs_train
            .grad()
            .expect("empty CUDA rhs grad should be recorded");
        assert_eq!(lhs_grad.len(), 0);
        assert_eq!(rhs_grad.len(), 0);

        crate::autograd::set_strict_device_execution(false);
        crate::ops::cuda::set_enabled(false);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_sum_backward_matches_cpu_reference_in_strict_mode() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        let shape = [128, 128];
        let values = (0..(shape[0] * shape[1]))
            .map(|i| (i as f32 % 67.0) / 23.0 - 1.25)
            .collect::<Vec<_>>();

        crate::ops::cuda::set_enabled(false);
        crate::autograd::set_strict_device_execution(false);
        let cpu_input = make_training_tensor(&shape, values.clone());
        let cpu_out = sum(&cpu_input);
        cpu_out.backward();
        let cpu_grad = cpu_input
            .grad()
            .expect("CPU sum backward should populate input grad");
        let cpu_sum = cpu_out.data_ref().first().copied().unwrap();

        crate::ops::cuda::set_enabled(true);
        crate::autograd::set_strict_device_execution(true);
        let cuda_input = make_training_tensor(&shape, values).to_cuda();
        let cuda_out = sum(&cuda_input);
        assert!(cuda_out.is_cuda());
        cuda_out.backward();
        let cuda_grad = cuda_input
            .grad()
            .expect("CUDA sum backward should populate input grad");
        assert!(cuda_input.cloned_cuda_f32_grad().is_some());
        let cuda_sum = cuda_out.data_ref().first().copied().unwrap();

        crate::autograd::set_strict_device_execution(false);
        crate::ops::cuda::set_enabled(false);

        assert!(
            (cuda_sum - cpu_sum).abs() <= 1e-2,
            "CUDA sum got {cuda_sum}, CPU expected {cpu_sum}"
        );
        for (idx, (got, expect)) in cuda_grad.iter().zip(cpu_grad.iter()).enumerate() {
            assert!(
                (got - expect).abs() <= 1e-6,
                "sum grad mismatch at {idx}: got {got}, expect {expect}"
            );
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_binary_backward_matches_cpu_reference_in_strict_mode() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        let shape = [64, 256];
        let lhs_values = (0..(shape[0] * shape[1]))
            .map(|i| (i as f32 % 37.0) / 17.0 - 0.9)
            .collect::<Vec<_>>();
        let rhs_values = (0..(shape[0] * shape[1]))
            .map(|i| ((i * 11) as f32 % 43.0) / 19.0 - 0.7)
            .collect::<Vec<_>>();

        for op in [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul] {
            crate::ops::cuda::set_enabled(false);
            crate::autograd::set_strict_device_execution(false);
            let cpu_lhs = make_training_tensor(&shape, lhs_values.clone());
            let cpu_rhs = make_training_tensor(&shape, rhs_values.clone());
            let cpu_out = match op {
                BinaryOp::Add => sum(&(cpu_lhs.clone() + cpu_rhs.clone())),
                BinaryOp::Sub => sum(&(cpu_lhs.clone() - cpu_rhs.clone())),
                BinaryOp::Mul => sum(&(cpu_lhs.clone() * cpu_rhs.clone())),
            };
            cpu_out.backward();
            let cpu_lhs_grad = cpu_lhs
                .grad()
                .expect("CPU binary backward should populate lhs grad");
            let cpu_rhs_grad = cpu_rhs
                .grad()
                .expect("CPU binary backward should populate rhs grad");

            crate::ops::cuda::set_enabled(true);
            crate::autograd::set_strict_device_execution(true);
            let cuda_lhs = make_training_tensor(&shape, lhs_values.clone()).to_cuda();
            let cuda_rhs = make_training_tensor(&shape, rhs_values.clone()).to_cuda();
            let binary_out = match op {
                BinaryOp::Add => cuda_lhs.clone() + cuda_rhs.clone(),
                BinaryOp::Sub => cuda_lhs.clone() - cuda_rhs.clone(),
                BinaryOp::Mul => cuda_lhs.clone() * cuda_rhs.clone(),
            };
            assert!(binary_out.is_cuda());
            assert!(!binary_out.has_host_f32_data());
            let cuda_out = match op {
                BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul => sum(&binary_out),
            };
            assert!(cuda_out.is_cuda());
            cuda_out.backward();
            assert!(!cuda_lhs.has_host_grad());
            assert!(!cuda_rhs.has_host_grad());
            assert!(cuda_lhs.cloned_cuda_f32_grad().is_some());
            assert!(cuda_rhs.cloned_cuda_f32_grad().is_some());
            let cuda_lhs_grad = cuda_lhs
                .grad()
                .expect("CUDA binary backward should populate lhs grad");
            let cuda_rhs_grad = cuda_rhs
                .grad()
                .expect("CUDA binary backward should populate rhs grad");

            for (idx, (got, expect)) in cuda_lhs_grad.iter().zip(cpu_lhs_grad.iter()).enumerate() {
                assert!(
                    (got - expect).abs() <= 1e-5,
                    "{op:?} lhs grad mismatch at {idx}: got {got}, expect {expect}"
                );
            }
            for (idx, (got, expect)) in cuda_rhs_grad.iter().zip(cpu_rhs_grad.iter()).enumerate() {
                assert!(
                    (got - expect).abs() <= 1e-5,
                    "{op:?} rhs grad mismatch at {idx}: got {got}, expect {expect}"
                );
            }
        }

        crate::autograd::set_strict_device_execution(false);
        crate::ops::cuda::set_enabled(false);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn backward_traverses_node_with_cuda_only_grad() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        crate::ops::cuda::set_enabled(true);
        crate::autograd::set_strict_device_execution(true);

        let shape = [64, 256];
        let values = (0..(shape[0] * shape[1]))
            .map(|i| (i as f32 % 29.0) / 11.0 - 0.75)
            .collect::<Vec<_>>();
        let input = make_training_tensor(&shape, values).to_cuda();
        let out = input.clone() + input.clone();
        assert!(out.is_cuda());
        assert!(!out.has_host_f32_data());

        let grad = vec![1.0f32; out.len()];
        let grad_buffer =
            crate::ops::cuda::upload_f32(&grad).expect("test should upload CUDA-only grad");
        out.add_cuda_grad_buffer_only(grad_buffer);
        out.backward();
        assert!(
            !out.has_host_f32_data(),
            "strict CUDA-only backward should not materialize output data"
        );

        let input_grad = input
            .grad()
            .expect("CUDA-only grad should still propagate to input");
        assert!(input.cloned_cuda_f32_grad().is_some());
        for (idx, got) in input_grad.iter().enumerate() {
            assert!(
                (*got - 2.0).abs() <= 1e-6,
                "CUDA-only backward grad mismatch at {idx}: got {got}"
            );
        }

        crate::autograd::set_strict_device_execution(false);
        crate::ops::cuda::set_enabled(false);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn backward_seeds_no_host_cuda_output_without_host_grad() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        crate::ops::cuda::set_enabled(true);
        crate::autograd::set_strict_device_execution(true);

        let shape = [64, 256];
        let values = (0..(shape[0] * shape[1]))
            .map(|i| (i as f32 % 23.0) / 9.0 - 0.6)
            .collect::<Vec<_>>();
        let input = make_training_tensor(&shape, values).to_cuda();
        let out = input.clone() + input.clone();
        assert!(!out.has_host_f32_data());

        out.backward();
        assert!(
            !out.has_host_f32_data(),
            "strict CUDA backward seed should stay CUDA-only for no-host outputs"
        );

        let input_grad = input.grad().expect("input grad");
        assert!(input.cloned_cuda_f32_grad().is_some());
        for (idx, got) in input_grad.iter().enumerate() {
            assert!(
                (*got - 2.0).abs() <= 1e-6,
                "seeded CUDA backward grad mismatch at {idx}: got {got}"
            );
        }

        crate::autograd::set_strict_device_execution(false);
        crate::ops::cuda::set_enabled(false);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn backward_materializes_cuda_only_grad_for_host_backed_cuda_node() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        crate::ops::cuda::set_enabled(true);
        crate::autograd::set_strict_device_execution(true);

        let shape = [128];
        let values = (0..shape[0])
            .map(|i| i as f32 / 17.0 - 0.5)
            .collect::<Vec<_>>();
        let input = make_training_tensor(&shape, values).to_cuda();
        let out = sum(&input);
        assert!(out.is_cuda());
        assert!(
            out.has_host_f32_data(),
            "sum currently keeps a host scalar result"
        );

        let grad_buffer =
            crate::ops::cuda::upload_f32(&[3.0]).expect("test should upload CUDA-only grad");
        out.add_cuda_grad_buffer_only(grad_buffer);
        out.backward();

        let input_grad = input
            .grad()
            .expect("CUDA-only sum grad should still propagate to input");
        assert!(input.cloned_cuda_f32_grad().is_some());
        for (idx, got) in input_grad.iter().enumerate() {
            assert!(
                (*got - 3.0).abs() <= 1e-6,
                "CUDA-only sum backward grad mismatch at {idx}: got {got}"
            );
        }

        crate::autograd::set_strict_device_execution(false);
        crate::ops::cuda::set_enabled(false);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_only_grad_accumulates_after_host_grad() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        crate::ops::cuda::set_enabled(true);
        crate::autograd::set_strict_device_execution(true);

        let input = make_training_tensor(&[3], vec![1.0, 2.0, 3.0]).to_cuda();
        let out = input.clone() + input.clone();
        out.add_grad(ArrayD::from_elem(IxDyn(&[3]), 1.0));
        let cuda_grad = crate::ops::cuda::upload_f32(&[2.0, 2.0, 2.0]).expect("upload grad");
        out.add_cuda_grad_buffer_only(cuda_grad);

        out.backward();

        let grad = input.grad().expect("input grad");
        assert_eq!(
            grad.iter().copied().collect::<Vec<_>>(),
            vec![6.0, 6.0, 6.0]
        );
        assert!(input.cloned_cuda_f32_grad().is_some());
        crate::autograd::set_strict_device_execution(false);
        crate::ops::cuda::set_enabled(false);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_broadcast_binary_backward_matches_cpu_reference_in_strict_mode() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        let lhs_shape = [32, 128];
        let rhs_shape = [128];
        let lhs_values = (0..(lhs_shape[0] * lhs_shape[1]))
            .map(|i| (i as f32 % 31.0) / 13.0 - 0.8)
            .collect::<Vec<_>>();
        let rhs_values = (0..rhs_shape[0])
            .map(|i| (i as f32 % 17.0) / 7.0 - 1.1)
            .collect::<Vec<_>>();

        for op in [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul] {
            crate::ops::cuda::set_enabled(false);
            crate::autograd::set_strict_device_execution(false);
            let cpu_lhs = make_training_tensor(&lhs_shape, lhs_values.clone());
            let cpu_rhs = make_training_tensor(&rhs_shape, rhs_values.clone());
            let cpu_out = match op {
                BinaryOp::Add => sum(&(cpu_lhs.clone() + cpu_rhs.clone())),
                BinaryOp::Sub => sum(&(cpu_lhs.clone() - cpu_rhs.clone())),
                BinaryOp::Mul => sum(&(cpu_lhs.clone() * cpu_rhs.clone())),
            };
            cpu_out.backward();
            let cpu_lhs_grad = cpu_lhs
                .grad()
                .expect("CPU broadcast binary backward should populate lhs grad");
            let cpu_rhs_grad = cpu_rhs
                .grad()
                .expect("CPU broadcast binary backward should populate rhs grad");

            crate::ops::cuda::set_enabled(true);
            crate::autograd::set_strict_device_execution(true);
            let cuda_lhs = make_training_tensor(&lhs_shape, lhs_values.clone()).to_cuda();
            let cuda_rhs = make_training_tensor(&rhs_shape, rhs_values.clone()).to_cuda();
            let cuda_out = match op {
                BinaryOp::Add => sum(&(cuda_lhs.clone() + cuda_rhs.clone())),
                BinaryOp::Sub => sum(&(cuda_lhs.clone() - cuda_rhs.clone())),
                BinaryOp::Mul => sum(&(cuda_lhs.clone() * cuda_rhs.clone())),
            };
            assert!(cuda_out.is_cuda());
            cuda_out.backward();
            assert!(!cuda_lhs.has_host_grad());
            assert!(!cuda_rhs.has_host_grad());
            assert!(cuda_lhs.cloned_cuda_f32_grad().is_some());
            assert!(cuda_rhs.cloned_cuda_f32_grad().is_some());
            let cuda_lhs_grad = cuda_lhs
                .grad()
                .expect("CUDA broadcast binary backward should populate lhs grad");
            let cuda_rhs_grad = cuda_rhs
                .grad()
                .expect("CUDA broadcast binary backward should populate rhs grad");

            for (idx, (got, expect)) in cuda_lhs_grad.iter().zip(cpu_lhs_grad.iter()).enumerate() {
                assert!(
                    (got - expect).abs() <= 1e-5,
                    "{op:?} broadcast lhs grad mismatch at {idx}: got {got}, expect {expect}"
                );
            }
            for (idx, (got, expect)) in cuda_rhs_grad.iter().zip(cpu_rhs_grad.iter()).enumerate() {
                assert!(
                    (got - expect).abs() <= 1e-4,
                    "{op:?} broadcast rhs grad mismatch at {idx}: got {got}, expect {expect}"
                );
            }
        }

        crate::autograd::set_strict_device_execution(false);
        crate::ops::cuda::set_enabled(false);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_async_general_broadcast_chain_matches_cpu_after_metadata_and_buffer_reuse() {
        if !crate::ops::cuda::is_available() {
            return;
        }

        {
            let a = make_tensor(
                &[2, 1, 1, 3],
                vec![1.0, -2.0, 0.5, 3.0, -4.0, 2.0],
                DType::F32,
            );
            let b = make_tensor(
                &[1, 4, 2, 3],
                (0..24).map(|i| (i as f32 % 11.0) * 0.2 - 0.75).collect(),
                DType::F32,
            );
            let c = make_tensor(
                &[2, 4, 1, 1],
                (0..8).map(|i| 0.5 + i as f32 * 0.125).collect(),
                DType::F32,
            );
            let d = make_tensor(
                &[1, 1, 2, 3],
                vec![0.25, -0.5, 1.0, -1.5, 0.75, 0.125],
                DType::F32,
            );
            let cpu_out = no_grad(|| ((a.clone() + b.clone()) * c.clone()) - d.clone());

            crate::ops::cuda::set_enabled(true);
            crate::autograd::set_strict_device_execution(true);
            let cuda_out = no_grad(|| ((a.to_cuda() + b.to_cuda()) * c.to_cuda()) - d.to_cuda());
            assert!(cuda_out.is_cuda());
            assert!(!cuda_out.has_host_f32_data());
            crate::autograd::set_strict_device_execution(false);
            crate::ops::cuda::set_enabled(false);

            for (idx, (got, expected)) in cuda_out
                .data_ref()
                .iter()
                .zip(cpu_out.data_ref().iter())
                .enumerate()
            {
                assert!(
                    (got - expected).abs() <= 1e-6,
                    "async general broadcast chain mismatch at {idx}: got {got}, expected {expected}"
                );
            }
        }
        crate::ops::cuda::release_cached_memory().expect("release CUDA cached memory");
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[should_panic(expected = "same device")]
    fn add_panics_on_mixed_devices() {
        if !crate::ops::cuda::is_available() {
            panic!("same device");
        }

        let lhs = make_tensor(&[2], vec![1.0, 2.0], DType::F32).to_cuda();
        let rhs = make_tensor(&[2], vec![3.0, 4.0], DType::F32);
        let _ = no_grad(|| lhs + rhs);
    }
}
