use super::*;

use crate::autograd::no_grad;
use crate::device::{Device, set_strict_device_execution_scoped};
use ndarray::{ArrayD, IxDyn, ShapeBuilder};

fn check_empty_products(device: Device) {
    let _strict = set_strict_device_execution_scoped(true);
    let dtypes = [DType::F32, DType::F16, DType::BF16, DType::I8];
    let ordinary = [
        (vec![0], vec![3, 0]),
        (vec![2, 0], vec![3, 0]),
        (vec![0, 5], vec![3, 5]),
        (vec![2, 5], vec![0, 5]),
        (vec![0, 0], vec![3, 0]),
        (vec![2, 0], vec![0, 0]),
        (vec![0, 5], vec![0, 5]),
        (vec![2, 0, 3, 5], vec![4, 5]),
        (vec![2, 3, 0], vec![4, 0]),
        (vec![2, 3, 5], vec![0, 5]),
    ];
    let batched = [
        (vec![0, 2, 3, 5], vec![0, 2, 5, 4]),
        (vec![2, 0, 3, 5], vec![2, 0, 5, 4]),
        (vec![2, 2, 0, 5], vec![2, 2, 5, 4]),
        (vec![2, 2, 3, 0], vec![2, 2, 0, 4]),
        (vec![2, 2, 3, 5], vec![2, 2, 5, 0]),
        (vec![0, 2, 3, 0], vec![0, 2, 0, 4]),
        (vec![2, 2, 0, 0], vec![2, 2, 0, 4]),
        (vec![2, 2, 3, 0], vec![2, 2, 0, 0]),
    ];
    let mut cases = 0;
    for is_batched in [false, true] {
        let shapes = if is_batched {
            &batched[..]
        } else {
            &ordinary[..]
        };
        for (a_shape, b_shape) in shapes {
            let out_shape = if is_batched {
                vec![a_shape[0], a_shape[1], a_shape[2], b_shape[3]]
            } else {
                let mut shape = a_shape.clone();
                *shape.last_mut().unwrap() = b_shape[0];
                shape
            };
            for a_dtype in dtypes {
                for b_dtype in dtypes {
                    for column_major in [false, true] {
                        for (a_grad, b_grad) in
                            [(false, false), (true, true), (true, false), (false, true)]
                        {
                            let make = |shape: &[usize], dtype, requires_grad| {
                                let layout = IxDyn(shape).set_f(column_major);
                                let data = ArrayD::from_elem(layout, 0.75);
                                let t = if requires_grad {
                                    Tensor::new_with_dtype(data, dtype)
                                } else {
                                    Tensor::from_f32_data_no_grad_with_device_dtype(
                                        data,
                                        dtype,
                                        Device::Cpu,
                                    )
                                };
                                t.to_device_inplace(device);
                                t
                            };
                            let a = make(a_shape, a_dtype, a_grad);
                            let b = make(b_shape, b_dtype, b_grad);
                            let host_before = (a.has_host_f32_data(), b.has_host_f32_data());
                            let call = || {
                                if is_batched {
                                    batch_matmul(&a, &b)
                                } else {
                                    matmul_transposed_rhs(&a, &b)
                                }
                            };
                            let assert_output = |out: &Tensor, training| {
                                assert_eq!(out.shape_vec(), out_shape);
                                assert_eq!(out.device(), device);
                                assert_eq!(out.requires_grad(), training);
                                let dtype = if training || a_dtype != b_dtype {
                                    DType::F32
                                } else {
                                    a_dtype
                                };
                                assert_eq!(out.dtype(), dtype);
                                assert!(out.data().iter().all(|&v| v == 0.));
                                if device == Device::Cuda && out.is_empty() {
                                    assert!(out.cloned_cuda_f32_buffer().is_none());
                                }
                            };
                            let out = call();
                            assert_output(&out, a_grad || b_grad);
                            if a_grad || b_grad {
                                out.backward_with_grad(ArrayD::from_elem(IxDyn(&out_shape), 2.));
                                for (parent, required) in [(&a, a_grad), (&b, b_grad)] {
                                    if required {
                                        let grad = parent.grad().unwrap();
                                        assert_eq!(grad.shape(), parent.shape_vec());
                                        assert!(grad.iter().all(|&v| v == 0.));
                                        drop(grad);
                                        parent.add_grad(ArrayD::from_elem(
                                            IxDyn(&parent.shape_vec()),
                                            1.25,
                                        ));
                                    } else {
                                        assert!(parent.grad().is_none());
                                    }
                                }
                                // A zero VJP must add to, rather than erase, an earlier contribution.
                                for _ in 0..2 {
                                    out.backward_with_grad(ArrayD::from_elem(
                                        IxDyn(&out_shape),
                                        -3.,
                                    ));
                                }
                                for (parent, required) in [(&a, a_grad), (&b, b_grad)] {
                                    if required {
                                        assert!(parent.grad().unwrap().iter().all(|&v| v == 1.25));
                                    } else {
                                        assert!(parent.grad().is_none());
                                    }
                                }
                            }
                            no_grad(|| assert_output(&call(), false));
                            assert_eq!((a.has_host_f32_data(), b.has_host_f32_data()), host_before);
                            cases += 1;
                        }
                    }
                }
            }
        }
    }
    assert_eq!(cases, 2304);
    println!("EMPTY_MATMUL_CHECK device={device:?} cases={cases} passed");
}

#[test]
fn empty_products_cpu_forward_backward_and_accumulation() {
    check_empty_products(Device::Cpu);
}

#[cfg(feature = "cuda")]
#[test]
fn empty_products_cuda_forward_backward_and_accumulation() {
    if !cuda::is_available() {
        println!("SKIP empty CUDA products: no CUDA device");
        return;
    }
    cuda::set_enabled(true);
    check_empty_products(Device::Cuda);
}

#[test]
#[should_panic(expected = "MatMul LHS must have at least one dimension")]
fn scalar_lhs_has_explicit_shape_error() {
    let a = Tensor::from_array_no_grad(ArrayD::from_elem(IxDyn(&[]), 1.));
    let b = Tensor::from_array_no_grad(ArrayD::zeros(IxDyn(&[2, 1])));
    matmul_transposed_rhs(&a, &b);
}

#[test]
#[should_panic(expected = "MatMul shape mismatch")]
fn empty_operands_still_validate_reduction_dimensions() {
    let a = Tensor::from_array_no_grad(ArrayD::zeros(IxDyn(&[2, 0])));
    let b = Tensor::from_array_no_grad(ArrayD::zeros(IxDyn(&[3, 1])));
    matmul_transposed_rhs(&a, &b);
}
