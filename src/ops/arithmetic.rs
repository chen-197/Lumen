// src/ops/arithmetic.rs
use crate::autograd::{
    StoragePreference, Tensor, TensorData, TensorStorageOwned, TensorStorageView, is_no_grad,
};
use crate::device::{
    Device, assert_native_device_support, assert_same_device, is_strict_device_execution,
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

// Flat kernels emit row-major output and use logical row/column indices.
// A contiguous transposed view has a different physical element order.
fn apply_binary_views(
    lhs: ArrayViewD<'_, f32>,
    rhs: ArrayViewD<'_, f32>,
    op: BinaryOp,
) -> ArrayD<f32> {
    if lhs.shape() == rhs.shape()
        && let (Some(lhs_slice), Some(rhs_slice)) = (lhs.as_slice(), rhs.as_slice())
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
        && let (Some(&last_dim), Some(lhs_slice), Some(rhs_slice)) =
            (lhs.shape().last(), lhs.as_slice(), rhs.as_slice())
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
                let (Some(lhs_slice), Some(rhs_slice)) = (lhs_view.as_slice(), rhs_view.as_slice())
                else {
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
                let (Some(lhs_slice), Some(rhs_slice)) = (lhs_view.as_slice(), rhs_view.as_slice())
                else {
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
            let (Some(lhs_slice), Some(rhs_slice)) = (lhs_data.as_slice(), rhs_data.as_slice())
            else {
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
                let (Some(lhs_slice), Some(rhs_slice)) = (lhs_view.as_slice(), rhs_view.as_slice())
                else {
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
                let (Some(lhs_slice), Some(rhs_slice)) = (lhs_view.as_slice(), rhs_view.as_slice())
                else {
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
            let (Some(lhs_slice), Some(rhs_slice)) = (lhs_data.as_slice(), rhs_data.as_slice())
            else {
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
                let (Some(lhs_slice), Some(rhs_slice)) = (lhs_view.as_slice(), rhs_view.as_slice())
                else {
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
                let (Some(lhs_slice), Some(rhs_slice)) = (lhs_view.as_slice(), rhs_view.as_slice())
                else {
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
            let (Some(lhs_slice), Some(rhs_slice)) = (lhs_data.as_slice(), rhs_data.as_slice())
            else {
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
                let (Some(lhs_slice), Some(rhs_slice)) = (lhs_view.as_slice(), rhs_view.as_slice())
                else {
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
                let (Some(lhs_slice), Some(rhs_slice)) = (lhs_view.as_slice(), rhs_view.as_slice())
                else {
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
            let (Some(lhs_slice), Some(rhs_slice)) = (lhs_data.as_slice(), rhs_data.as_slice())
            else {
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
                f32_data.as_slice()?,
                lowp_data.as_slice()?,
                false,
                op,
                &mut out,
            );
        }
        (TensorStorageOwned::F16(lowp_data), TensorStorageOwned::F32(f32_data)) => {
            apply_mixed_add_sub_f16(
                f32_data.as_slice()?,
                lowp_data.as_slice()?,
                true,
                op,
                &mut out,
            );
        }
        (TensorStorageOwned::F32(f32_data), TensorStorageOwned::BF16(lowp_data)) => {
            apply_mixed_add_sub_bf16(
                f32_data.as_slice()?,
                lowp_data.as_slice()?,
                false,
                op,
                &mut out,
            );
        }
        (TensorStorageOwned::BF16(lowp_data), TensorStorageOwned::F32(f32_data)) => {
            apply_mixed_add_sub_bf16(
                f32_data.as_slice()?,
                lowp_data.as_slice()?,
                true,
                op,
                &mut out,
            );
        }
        (TensorStorageOwned::F32(f32_data), TensorStorageOwned::I8(lowp_data, scale)) => {
            apply_mixed_add_sub_i8(
                f32_data.as_slice()?,
                lowp_data.as_slice()?,
                scale,
                false,
                op,
                &mut out,
            );
        }
        (TensorStorageOwned::I8(lowp_data, scale), TensorStorageOwned::F32(f32_data)) => {
            apply_mixed_add_sub_i8(
                f32_data.as_slice()?,
                lowp_data.as_slice()?,
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
            let f32_slice = f32_data.as_slice()?;
            let lowp_slice = lowp_data.as_slice()?;
            mul_f32_f16_slice_to_f32(f32_slice, lowp_slice, &mut out);
        }
        (TensorStorageOwned::F32(f32_data), TensorStorageOwned::BF16(lowp_data))
        | (TensorStorageOwned::BF16(lowp_data), TensorStorageOwned::F32(f32_data)) => {
            let f32_slice = f32_data.as_slice()?;
            let lowp_slice = lowp_data.as_slice()?;
            mul_f32_bf16_slice_to_f32(f32_slice, lowp_slice, &mut out);
        }
        (TensorStorageOwned::F32(f32_data), TensorStorageOwned::I8(lowp_data, scale))
        | (TensorStorageOwned::I8(lowp_data, scale), TensorStorageOwned::F32(f32_data)) => {
            let f32_slice = f32_data.as_slice()?;
            let lowp_slice = lowp_data.as_slice()?;
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
            let f32_slice = f32_data.as_slice()?;
            let lowp_slice = lowp_data.as_slice()?;
            apply_mixed_row_broadcast_f16(f32_slice, lowp_slice, side, rows, row_len, &mut out);
        }
        (TensorStorageOwned::F16(lowp_data), TensorStorageOwned::F32(f32_data)) => {
            let f32_slice = f32_data.as_slice()?;
            let lowp_slice = lowp_data.as_slice()?;
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
            let f32_slice = f32_data.as_slice()?;
            let lowp_slice = lowp_data.as_slice()?;
            apply_mixed_row_broadcast_bf16(f32_slice, lowp_slice, side, rows, row_len, &mut out);
        }
        (TensorStorageOwned::BF16(lowp_data), TensorStorageOwned::F32(f32_data)) => {
            let f32_slice = f32_data.as_slice()?;
            let lowp_slice = lowp_data.as_slice()?;
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
            let f32_slice = f32_data.as_slice()?;
            let lowp_slice = lowp_data.as_slice()?;
            apply_mixed_row_broadcast_i8(
                f32_slice, lowp_slice, scale, side, rows, row_len, &mut out,
            );
        }
        (TensorStorageOwned::I8(lowp_data, scale), TensorStorageOwned::F32(f32_data)) => {
            let f32_slice = f32_data.as_slice()?;
            let lowp_slice = lowp_data.as_slice()?;
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
                    f32_slice: f32_data.as_slice()?,
                    lowp_slice: lowp_data.as_slice()?,
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
                    f32_slice: f32_data.as_slice()?,
                    lowp_slice: lowp_data.as_slice()?,
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
                    f32_slice: f32_data.as_slice()?,
                    lowp_slice: lowp_data.as_slice()?,
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
                    f32_slice: f32_data.as_slice()?,
                    lowp_slice: lowp_data.as_slice()?,
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
                    f32_slice: f32_data.as_slice()?,
                    lowp_slice: lowp_data.as_slice()?,
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
                    f32_slice: f32_data.as_slice()?,
                    lowp_slice: lowp_data.as_slice()?,
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
            let f32_slice = f32_data.as_slice()?;
            let lowp_slice = lowp_data.as_slice()?;
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
            let f32_slice = f32_data.as_slice()?;
            let lowp_slice = lowp_data.as_slice()?;
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
            let f32_slice = f32_data.as_slice()?;
            let lowp_slice = lowp_data.as_slice()?;
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
            let f32_slice = f32_data.as_slice()?;
            let lowp_slice = lowp_data.as_slice()?;
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
            let f32_slice = f32_data.as_slice()?;
            let lowp_slice = lowp_data.as_slice()?;
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
            let f32_slice = f32_data.as_slice()?;
            let lowp_slice = lowp_data.as_slice()?;
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
            let f32_slice = f32_data.as_slice()?;
            let lowp_slice = lowp_data.as_slice()?;
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
            let f32_slice = f32_data.as_slice()?;
            let lowp_slice = lowp_data.as_slice()?;
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
            let f32_slice = f32_data.as_slice()?;
            let lowp_slice = lowp_data.as_slice()?;
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
            let f32_slice = f32_data.as_slice()?;
            let lowp_slice = lowp_data.as_slice()?;
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
            let f32_slice = f32_data.as_slice()?;
            let lowp_slice = lowp_data.as_slice()?;
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
            let f32_slice = f32_data.as_slice()?;
            let lowp_slice = lowp_data.as_slice()?;
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
            let f32_slice = f32_data.as_slice()?;
            let lowp_slice = lowp_data.as_slice()?;
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
            let f32_slice = f32_data.as_slice()?;
            let lowp_slice = lowp_data.as_slice()?;
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
            let f32_slice = f32_data.as_slice()?;
            let lowp_slice = lowp_data.as_slice()?;
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
            let f32_slice = f32_data.as_slice()?;
            let lowp_slice = lowp_data.as_slice()?;
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
            let f32_slice = f32_data.as_slice()?;
            let lowp_slice = lowp_data.as_slice()?;
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
            let f32_slice = f32_data.as_slice()?;
            let lowp_slice = lowp_data.as_slice()?;
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
            let f32_slice = f32_data.as_slice()?;
            let lowp_slice = lowp_data.as_slice()?;
            if f32_slice.len() == 1 {
                mul_f16_slice_by_scalar_to_f32(lowp_slice, f32_slice[0], &mut out);
            } else {
                mul_f32_slice_by_scalar_to_f32(f32_slice, lowp_slice[0].to_f32(), &mut out);
            }
        }
        (TensorStorageOwned::F16(lowp_data), TensorStorageOwned::F32(f32_data)) => {
            let f32_slice = f32_data.as_slice()?;
            let lowp_slice = lowp_data.as_slice()?;
            if lowp_slice.len() == 1 {
                mul_f32_slice_by_scalar_to_f32(f32_slice, lowp_slice[0].to_f32(), &mut out);
            } else {
                mul_f16_slice_by_scalar_to_f32(lowp_slice, f32_slice[0], &mut out);
            }
        }
        (TensorStorageOwned::F32(f32_data), TensorStorageOwned::BF16(lowp_data)) => {
            let f32_slice = f32_data.as_slice()?;
            let lowp_slice = lowp_data.as_slice()?;
            if f32_slice.len() == 1 {
                mul_bf16_slice_by_scalar_to_f32(lowp_slice, f32_slice[0], &mut out);
            } else {
                mul_f32_slice_by_scalar_to_f32(f32_slice, lowp_slice[0].to_f32(), &mut out);
            }
        }
        (TensorStorageOwned::BF16(lowp_data), TensorStorageOwned::F32(f32_data)) => {
            let f32_slice = f32_data.as_slice()?;
            let lowp_slice = lowp_data.as_slice()?;
            if lowp_slice.len() == 1 {
                mul_f32_slice_by_scalar_to_f32(f32_slice, lowp_slice[0].to_f32(), &mut out);
            } else {
                mul_bf16_slice_by_scalar_to_f32(lowp_slice, f32_slice[0], &mut out);
            }
        }
        (TensorStorageOwned::F32(f32_data), TensorStorageOwned::I8(lowp_data, scale)) => {
            let f32_slice = f32_data.as_slice()?;
            let lowp_slice = lowp_data.as_slice()?;
            if f32_slice.len() == 1 {
                mul_i8_slice_by_scalar_to_f32(lowp_slice, scale, f32_slice[0], &mut out);
            } else {
                mul_f32_slice_by_scalar_to_f32(f32_slice, (lowp_slice[0] as f32) * scale, &mut out);
            }
        }
        (TensorStorageOwned::I8(lowp_data, scale), TensorStorageOwned::F32(f32_data)) => {
            let f32_slice = f32_data.as_slice()?;
            let lowp_slice = lowp_data.as_slice()?;
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
        || lhs.dtype() == DType::F32
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
            let lhs_slice = lhs_data.as_slice()?;
            let rhs_slice = rhs_data.as_slice()?;
            let lhs_bits: Vec<u16> = lhs_slice.iter().map(|v| v.to_bits()).collect();
            let rhs_bits: Vec<u16> = rhs_slice.iter().map(|v| v.to_bits()).collect();
            cuda::binary_f16_host_no_host(&lhs_bits, &rhs_bits, cuda_op).ok()
        }
        (TensorStorageOwned::BF16(lhs_data), TensorStorageOwned::BF16(rhs_data)) => {
            let lhs_slice = lhs_data.as_slice()?;
            let rhs_slice = rhs_data.as_slice()?;
            let lhs_bits: Vec<u16> = lhs_slice.iter().map(|v| v.to_bits()).collect();
            let rhs_bits: Vec<u16> = rhs_slice.iter().map(|v| v.to_bits()).collect();
            cuda::binary_bf16_host_no_host(&lhs_bits, &rhs_bits, cuda_op).ok()
        }
        (
            TensorStorageOwned::I8(lhs_data, lhs_scale),
            TensorStorageOwned::I8(rhs_data, rhs_scale),
        ) => {
            let lhs_slice = lhs_data.as_slice()?;
            let rhs_slice = rhs_data.as_slice()?;
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
    if lhs.device() != Device::Cuda
        || rhs.device() != Device::Cuda
        || lhs.dtype() != rhs.dtype()
        || lhs.dtype() == DType::F32
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
            let lhs_slice = lhs_data.as_slice()?;
            let rhs_slice = rhs_data.as_slice()?;
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
            let lhs_slice = lhs_data.as_slice()?;
            let rhs_slice = rhs_data.as_slice()?;
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
            let lhs_slice = lhs_data.as_slice()?;
            let rhs_slice = rhs_data.as_slice()?;
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
            .as_slice()
            .and_then(|slice| sum_i8_arch(slice, scale))
            .or_else(|| Some(data.iter().map(|&v| (v as f32) * scale).sum()));
    }

    input.with_storage_view_preferring(StoragePreference::Native, |view| match view {
        TensorStorageView::F16(view) => view
            .as_slice()
            .and_then(sum_f16_arch)
            .or_else(|| Some(view.iter().map(|v| v.to_f32()).sum())),
        TensorStorageView::BF16(view) => view
            .as_slice()
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
                if output_device == crate::device::Device::Cuda
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

    if let (Some(&grad_last), Some(&target_last), Some(grad_slice)) =
        (grad.shape().last(), target_shape.last(), grad.as_slice())
    {
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
    let grad_slice = grad.as_slice();

    match operand.native_storage_owned() {
        TensorStorageOwned::F32(data) => {
            if let (Some(g), Some(x)) = (grad_slice, data.as_slice()) {
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
            if let (Some(g), Some(x)) = (grad_slice, data.as_slice()) {
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
            if let (Some(g), Some(x)) = (grad_slice, data.as_slice()) {
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
            if let (Some(g), Some(x)) = (grad_slice, data.as_slice()) {
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
    let grad_slice = grad.as_slice()?;

    match lhs.dtype() {
        DType::F16 => lhs.with_storage_view_preferring(StoragePreference::Native, |lhs_view| {
            rhs.with_storage_view_preferring(StoragePreference::Native, |rhs_view| {
                let (TensorStorageView::F16(lhs_view), TensorStorageView::F16(rhs_view)) =
                    (lhs_view, rhs_view)
                else {
                    return None;
                };
                let (Some(lhs_slice), Some(rhs_slice)) = (lhs_view.as_slice(), rhs_view.as_slice())
                else {
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
                let (Some(lhs_slice), Some(rhs_slice)) = (lhs_view.as_slice(), rhs_view.as_slice())
                else {
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
            let (Some(lhs_slice), Some(rhs_slice)) = (lhs_data.as_slice(), rhs_data.as_slice())
            else {
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
    let grad_slice = grad.as_slice()?;
    let mut lhs_grad = vec![0.0f32; lhs_shape.iter().product()];
    let mut rhs_grad = vec![0.0f32; rhs_shape.iter().product()];
    let mut tmp = vec![0.0f32; row_len];

    match (lhs.native_storage_owned(), rhs.native_storage_owned()) {
        (TensorStorageOwned::F32(lhs_data), TensorStorageOwned::F16(rhs_data)) => {
            let lhs_slice = lhs_data.as_slice()?;
            let rhs_slice = rhs_data.as_slice()?;
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
            let lhs_slice = lhs_data.as_slice()?;
            let rhs_slice = rhs_data.as_slice()?;
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
            let lhs_slice = lhs_data.as_slice()?;
            let rhs_slice = rhs_data.as_slice()?;
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
            let lhs_slice = lhs_data.as_slice()?;
            let rhs_slice = rhs_data.as_slice()?;
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
            let lhs_slice = lhs_data.as_slice()?;
            let rhs_slice = rhs_data.as_slice()?;
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
            let lhs_slice = lhs_data.as_slice()?;
            let rhs_slice = rhs_data.as_slice()?;
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
    let grad_slice = grad.as_slice()?;
    let mut lhs_grad = vec![0.0f32; lhs.len()];
    let mut rhs_grad = vec![0.0f32; rhs.len()];

    match (lhs.native_storage_owned(), rhs.native_storage_owned()) {
        (TensorStorageOwned::F32(lhs_data), TensorStorageOwned::F16(rhs_data)) => {
            let lhs_slice = lhs_data.as_slice()?;
            let rhs_slice = rhs_data.as_slice()?;
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
            let lhs_slice = lhs_data.as_slice()?;
            let rhs_slice = rhs_data.as_slice()?;
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
            let lhs_slice = lhs_data.as_slice()?;
            let rhs_slice = rhs_data.as_slice()?;
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
            let lhs_slice = lhs_data.as_slice()?;
            let rhs_slice = rhs_data.as_slice()?;
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
            let lhs_slice = lhs_data.as_slice()?;
            let rhs_slice = rhs_data.as_slice()?;
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
            let lhs_slice = lhs_data.as_slice()?;
            let rhs_slice = rhs_data.as_slice()?;
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
    let grad_slice = grad.as_slice()?;
    let mut lhs_grad = vec![0.0f32; lhs.len()];
    let mut rhs_grad = vec![0.0f32; rhs.len()];

    match (lhs.native_storage_owned(), rhs.native_storage_owned()) {
        (TensorStorageOwned::F32(lhs_data), TensorStorageOwned::F16(rhs_data)) => {
            let lhs_slice = lhs_data.as_slice()?;
            let rhs_slice = rhs_data.as_slice()?;
            if lhs_slice.len() == 1 {
                lhs_grad[0] = dot_f32_f16_slice(grad_slice, rhs_slice);
                mul_f32_slice_by_scalar_to_f32(grad_slice, lhs_slice[0], &mut rhs_grad);
            } else {
                mul_f32_slice_by_scalar_to_f32(grad_slice, rhs_slice[0].to_f32(), &mut lhs_grad);
                rhs_grad[0] = dot_f32_f32_slice(grad_slice, lhs_slice);
            }
        }
        (TensorStorageOwned::F16(lhs_data), TensorStorageOwned::F32(rhs_data)) => {
            let lhs_slice = lhs_data.as_slice()?;
            let rhs_slice = rhs_data.as_slice()?;
            if lhs_slice.len() == 1 {
                lhs_grad[0] = dot_f32_f32_slice(grad_slice, rhs_slice);
                mul_f32_slice_by_scalar_to_f32(grad_slice, lhs_slice[0].to_f32(), &mut rhs_grad);
            } else {
                mul_f32_slice_by_scalar_to_f32(grad_slice, rhs_slice[0], &mut lhs_grad);
                rhs_grad[0] = dot_f32_f16_slice(grad_slice, lhs_slice);
            }
        }
        (TensorStorageOwned::F32(lhs_data), TensorStorageOwned::BF16(rhs_data)) => {
            let lhs_slice = lhs_data.as_slice()?;
            let rhs_slice = rhs_data.as_slice()?;
            if lhs_slice.len() == 1 {
                lhs_grad[0] = dot_f32_bf16_slice(grad_slice, rhs_slice);
                mul_f32_slice_by_scalar_to_f32(grad_slice, lhs_slice[0], &mut rhs_grad);
            } else {
                mul_f32_slice_by_scalar_to_f32(grad_slice, rhs_slice[0].to_f32(), &mut lhs_grad);
                rhs_grad[0] = dot_f32_f32_slice(grad_slice, lhs_slice);
            }
        }
        (TensorStorageOwned::BF16(lhs_data), TensorStorageOwned::F32(rhs_data)) => {
            let lhs_slice = lhs_data.as_slice()?;
            let rhs_slice = rhs_data.as_slice()?;
            if lhs_slice.len() == 1 {
                lhs_grad[0] = dot_f32_f32_slice(grad_slice, rhs_slice);
                mul_f32_slice_by_scalar_to_f32(grad_slice, lhs_slice[0].to_f32(), &mut rhs_grad);
            } else {
                mul_f32_slice_by_scalar_to_f32(grad_slice, rhs_slice[0], &mut lhs_grad);
                rhs_grad[0] = dot_f32_bf16_slice(grad_slice, lhs_slice);
            }
        }
        (TensorStorageOwned::F32(lhs_data), TensorStorageOwned::I8(rhs_data, rhs_scale)) => {
            let lhs_slice = lhs_data.as_slice()?;
            let rhs_slice = rhs_data.as_slice()?;
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
            let lhs_slice = lhs_data.as_slice()?;
            let rhs_slice = rhs_data.as_slice()?;
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
            lhs.add_backward_grad(reduce_gradient(grad.view(), &l_shape));
            rhs.add_backward_grad(reduce_gradient(grad.view(), &r_shape));
        }
        BinaryOp::Sub => {
            lhs.add_backward_grad(reduce_gradient(grad.view(), &l_shape));
            let grad_neg = Zip::from(&grad).par_map_collect(|&x| -x);
            rhs.add_backward_grad(reduce_gradient(grad_neg.view(), &r_shape));
        }
        BinaryOp::Mul => {
            let (g_lhs, g_rhs, lhs_shape, rhs_shape) = {
                if let Some(native_grads) = try_same_shape_mul_grads_native(lhs, rhs, grad.view())
                    .or_else(|| try_row_broadcast_mul_grads_native(lhs, rhs, grad.view()))
                    .or_else(|| try_mixed_row_broadcast_mul_grads_native(lhs, rhs, grad.view()))
                    .or_else(|| try_mixed_row_scalar_mul_grads_native(lhs, rhs, grad.view()))
                    .or_else(|| try_mixed_scalar_mul_grads_native(lhs, rhs, grad.view()))
                {
                    lhs.add_backward_grad(reduce_gradient(native_grads.0.view(), &native_grads.2));
                    rhs.add_backward_grad(reduce_gradient(native_grads.1.view(), &native_grads.3));
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
            lhs.add_backward_grad(reduce_gradient(g_lhs.view(), &lhs_shape));
            rhs.add_backward_grad(reduce_gradient(g_rhs.view(), &rhs_shape));
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
            let output_self = Rc::new(RefCell::new(std::rc::Weak::<RefCell<TensorData>>::new()));
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
                        .upgrade()
                        .map(Tensor)
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
                                lhs_clone.add_backward_cuda_grad_buffer_only(lhs_buffer);
                                rhs_clone.add_backward_cuda_grad_buffer_only(rhs_buffer);
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
                            lhs_clone
                                .add_backward_grad_with_cuda_buffer(grad_lhs, Some(lhs_buffer));
                            rhs_clone
                                .add_backward_grad_with_cuda_buffer(grad_rhs, Some(rhs_buffer));
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
            *output_self.borrow_mut() = Rc::downgrade(&tensor.0);
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
                                    input_clone.add_backward_cuda_grad_buffer_only(grad_buffer);
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
                                input_clone.add_backward_grad_with_cuda_buffer(
                                    grad_input,
                                    Some(grad_buffer),
                                );
                            }
                            Err(err) => {
                                assert!(
                                    !is_strict_device_execution(),
                                    "sum CUDA backward failed while strict device execution is enabled: {err}"
                                );
                                let grad_input = ndarray::ArrayD::from_elem(input_shape.clone(), g);
                                input_clone.add_backward_grad(grad_input);
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
            input_clone.add_backward_grad(grad_input);
        })),
        requires_grad: true,
        device: output_device,
    })))
}

#[cfg(test)]
#[path = "../../tests/unit/ops/arithmetic/tests.rs"]
mod tests;
