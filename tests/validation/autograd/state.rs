#[cfg(feature = "cuda")]
use crate::accuracy_gate::check;
use crate::accuracy_gate::{run, tensor, values};
#[cfg(feature = "cuda")]
use crate::autograd::Tensor;
use crate::autograd::{Device, TensorRawData};
use crate::precision::DType;
use ndarray::{ArrayD, IxDyn};

#[test]
fn regression_optimizer_take_raw_data_clears_old_gradients() {
    use crate::ops::cuda;
    for device in [Device::Cpu, Device::Cuda] {
        if device == Device::Cuda && !cuda::is_available() {
            continue;
        }
        run(device, |device| {
            for dtype in [DType::F32, DType::F16, DType::BF16, DType::I8] {
                for resident_only in [false, true] {
                    let x = tensor(&[2, 2], &[1., -2., 0.5, 3.], dtype, 0, device);
                    let expected = values(&x);
                    if device == Device::Cuda && resident_only {
                        x.backward_with_cuda_grad(cuda::upload_f32(&[1.; 4]).unwrap());
                    } else {
                        x.add_grad(ArrayD::ones(IxDyn(&[2, 2])));
                    }
                    let (shape, raw) = x.take_raw_data();
                    assert_eq!(shape, vec![2, 2]);
                    assert_eq!(raw, expected);
                    assert!(x.is_empty());
                    assert_eq!(x.device(), device);
                    assert!(
                        !x.has_pending_grad(),
                        "take_raw_data retained a gradient with the old shape"
                    );
                    assert!(x.grad_ref().is_none());
                    assert!(x.grad().is_none());
                    assert!(x.cloned_cuda_f32_grad().is_none());
                    assert!(x.data_ref().is_empty());
                    assert_eq!(x.take_raw_data(), (vec![0], vec![]));
                }
            }
        });
    }
}

#[cfg(feature = "cuda")]
fn array(values: &[f32]) -> ArrayD<f32> {
    ArrayD::from_shape_vec(IxDyn(&[2, 2]), values.to_vec()).unwrap()
}

fn native_mutation_invalidates_compute_cache(device: Device) {
    use crate::autograd::{StoragePreference, TensorStorageView, TensorStorageViewMut};
    for dtype in [DType::F16, DType::BF16, DType::I8] {
        let x = tensor(&[2, 2], &[0.25, -0.5, 0.75, -1.], dtype, 0, device);
        let before =
            x.with_storage_view_preferring(StoragePreference::F32Compute, |view| match view {
                TensorStorageView::F32(data) => data.iter().copied().collect::<Vec<_>>(),
                _ => unreachable!(),
            });
        x.with_native_storage_view_mut(|view| match view {
            TensorStorageViewMut::F16(mut data) => data.mapv_inplace(|x| -x),
            TensorStorageViewMut::BF16(mut data) => data.mapv_inplace(|x| -x),
            TensorStorageViewMut::I8(mut data, _) => data.mapv_inplace(|x| -x),
            _ => panic!("native view changed dtype"),
        });
        assert!(!x.has_host_f32_data());
        let after =
            x.with_storage_view_preferring(StoragePreference::F32Compute, |view| match view {
                TensorStorageView::F32(data) => data.iter().copied().collect::<Vec<_>>(),
                _ => unreachable!(),
            });
        assert_eq!(x.dtype(), dtype);
        assert_eq!(after, before.iter().map(|x| -*x).collect::<Vec<_>>());
    }
}

#[test]
fn cpu_native_write_invalidates_f32_compute_cache() {
    run(Device::Cpu, native_mutation_invalidates_compute_cache);
}

#[test]
fn cpu_optimizer_preserves_native_dtype_after_reading_compute_cache() {
    use crate::autograd::StoragePreference;
    use crate::optim::{Adam, Optimizer, SGD};
    run(Device::Cpu, |device| {
        for dtype in [DType::F16, DType::BF16, DType::I8] {
            for layout in 0..3 {
                for kind in ["sgd", "momentum", "adam"] {
                    let cached = tensor(&[2, 2], &[1., -2., 0.5, 3.], dtype, layout, device);
                    let plain = tensor(&[2, 2], &[1., -2., 0.5, 3.], dtype, layout, device);
                    cached.with_storage_view_preferring(StoragePreference::F32Compute, |_| {});
                    assert!(cached.has_host_f32_data());
                    let mut opts: Vec<Box<dyn Optimizer>> = [&cached, &plain]
                        .into_iter()
                        .map(|param| {
                            if kind == "adam" {
                                Box::new(Adam::new_with_dtype(vec![param.clone()], 0.1, DType::F32))
                                    as Box<dyn Optimizer>
                            } else {
                                Box::new(
                                    SGD::new_with_dtype(vec![param.clone()], 0.1, DType::F32)
                                        .with_momentum(if kind == "momentum" { 0.9 } else { 0. }),
                                ) as Box<dyn Optimizer>
                            }
                        })
                        .collect();
                    for _ in 0..3 {
                        for (param, opt) in [&cached, &plain].into_iter().zip(&mut opts) {
                            param.zero_grad();
                            param.add_grad(crate::accuracy_gate::ordered(
                                &[2, 2],
                                &[0.5, -1., 2., 0.25],
                                layout,
                            ));
                            opt.step();
                            assert_eq!(param.dtype(), dtype);
                        }
                        assert_eq!(
                            values(&cached),
                            values(&plain),
                            "cache altered {kind}/{dtype:?}"
                        );
                        cached.with_storage_view_preferring(StoragePreference::F32Compute, |_| {});
                    }
                }
            }
        }
    });
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_native_write_invalidates_f32_compute_cache() {
    run(Device::Cuda, native_mutation_invalidates_compute_cache);
}

#[test]
fn cpu_same_device_move_accepts_held_read_view() {
    run(Device::Cpu, |device| {
        let x = tensor(&[2, 2], &[1., 2., 3., 4.], DType::F32, 0, device);
        let view = x.data_ref();
        x.to_cpu_inplace();
        assert_eq!(view.iter().copied().collect::<Vec<_>>(), [1., 2., 3., 4.]);
    });
}

#[test]
fn invalid_raw_import_preserves_values_and_gradients() {
    run(Device::Cpu, |device| {
        let x = tensor(&[2, 2], &[1., 2., 3., 4.], DType::F32, 0, device);
        x.add_grad(ArrayD::ones(IxDyn(&[2, 2])));
        for scale in [0., -1., f32::NAN, f32::INFINITY] {
            assert!(
                x.import_raw(
                    vec![2, 2],
                    DType::I8,
                    TensorRawData::I8 {
                        values: vec![1; 4],
                        scale
                    }
                )
                .is_err()
            );
            assert_eq!(x.dtype(), DType::F32);
            assert_eq!(values(&x), [1., 2., 3., 4.]);
            assert_eq!(x.grad().unwrap(), ArrayD::<f32>::ones(IxDyn(&[2, 2])));
        }
    });
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_cpu_transfer_preserves_pending_gradient_for_cpu_optimizer() {
    use crate::optim::{Optimizer, SGD};
    run(Device::Cuda, |device| {
        let x = tensor(&[2, 2], &[1., 2., 3., 4.], DType::F32, 0, device);
        (&x * &x).sum().backward();
        assert!(!x.has_host_grad());
        x.to_cpu_inplace();
        assert!(x.cloned_cuda_f32_grad().is_none());
        SGD::new_with_dtype(vec![x.clone()], 0.1, DType::F32).step();
        check(
            "transfer pending gradient",
            &values(&x),
            &[0.8, 1.6, 2.4, 3.2],
            2e-6,
            2e-6,
        );
    });
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_snapshot_is_independent_after_next_inplace_optimizer_update() {
    use crate::optim::{Optimizer, SGD};
    run(Device::Cuda, |_| {
        for dtype in [DType::F32, DType::F16, DType::BF16, DType::I8] {
            let x = updated(dtype);
            let copy = x.detach();
            assert!(!copy.requires_grad());
            let before = values(&copy);
            x.add_grad(array(&[1.; 4]));
            SGD::new_with_dtype(vec![x.clone()], 0.1, DType::F32).step();
            assert_ne!(values(&x), before);
            assert_eq!(values(&copy), before);
        }
    });
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_copy_accepts_held_host_view_after_host_mutation() {
    run(Device::Cuda, |device| {
        let x = tensor(&[2, 2], &[1., 2., 3., 4.], DType::F32, 0, device);
        x.data_mut()[[0, 0]] = 7.;
        let view = x.data_ref();
        assert!(x.0.borrow().cuda_f32_data.is_none());
        for copy in [x.detach(), x.to_cuda(), x.to_cpu()] {
            assert_eq!(values(&copy), [7., 2., 3., 4.]);
        }
        assert_eq!(view[[0, 0]], 7.);
        assert!(x.0.borrow().cuda_f32_data.is_none());
    });
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_f32_binary_dispatch_does_not_snapshot_host_inputs() {
    use crate::autograd::no_grad;
    use crate::ops::cuda;
    run(Device::Cuda, |device| {
        for shape in [vec![2, 3], vec![3], vec![]] {
            let left = Tensor::from_cuda_f32_buffer_no_host(
                &[2, 3],
                cuda::upload_f32(&[1.; 6]).unwrap(),
                device,
            );
            let right = Tensor::from_cuda_f32_buffer_no_host(
                &shape,
                cuda::upload_f32(&vec![2.; shape.iter().product()]).unwrap(),
                device,
            );
            for op in 0..3 {
                let before = cuda::test_host_downloads();
                let output = no_grad(|| match op {
                    0 => &left + &right,
                    1 => &left - &right,
                    _ => &left * &right,
                });
                assert_eq!(
                    cuda::test_host_downloads(),
                    before,
                    "dispatch downloaded unused F32 input snapshots"
                );
                assert!(!left.has_host_f32_data() && !right.has_host_f32_data());
                assert_eq!(
                    values(&output),
                    vec![
                        match op {
                            0 => 3.,
                            1 => -1.,
                            _ => 2.,
                        };
                        6
                    ]
                );
            }
        }
    });
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_matmul_dispatch_does_not_snapshot_unsupported_host_pairs() {
    use crate::autograd::no_grad;
    use crate::ops::{cuda, matmul};
    run(Device::Cuda, |device| {
        for rows in [1, 2] {
            let input = [0.5, -1., 2., 0.25, 0.75, -0.5][..rows * 3].to_vec();
            for dtype in [DType::F32, DType::F16, DType::BF16, DType::I8] {
                let x = Tensor::from_cuda_f32_buffer_no_host(
                    &[rows, 3],
                    cuda::upload_f32(&input).unwrap(),
                    device,
                );
                let w = tensor(
                    &[2, 3],
                    &[0.25, -0.5, 0.75, -1., 0.125, 0.5],
                    dtype,
                    0,
                    device,
                );
                // Float compute buffers can coexist with native low-precision weights.
                // Preparing the weight is separate from the dispatch under test.
                w.cloned_cuda_f32_buffer().unwrap();
                let represented = values(&w);
                let expected = (0..rows)
                    .flat_map(|row| {
                        let input = &input;
                        let represented = &represented;
                        (0..2).map(move |column| {
                            (0..3)
                                .map(|k| {
                                    input[row * 3 + k] as f64 * represented[column * 3 + k] as f64
                                })
                                .sum::<f64>()
                        })
                    })
                    .collect::<Vec<_>>();
                let before = cuda::test_host_downloads();
                let output = no_grad(|| matmul::matmul(&x, &w));
                assert_eq!(
                    cuda::test_host_downloads(),
                    before,
                    "unsupported native probe downloaded inputs for {dtype:?}"
                );
                assert!(!x.has_host_f32_data());
                check(
                    "resident mixed matmul",
                    &values(&output),
                    &expected,
                    2e-6,
                    2e-6,
                );
            }
        }
    });
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_native_mutation_keeps_dtype_scale_and_current_values() {
    use crate::autograd::TensorStorageViewMut;
    run(Device::Cuda, |_| {
        for dtype in [DType::F16, DType::BF16, DType::I8] {
            let x = updated(dtype);
            let before = values(&x);
            let scale = x.quantization_scale();
            x.with_native_storage_view_mut(|view| match view {
                TensorStorageViewMut::F16(mut data) => data.mapv_inplace(|v| -v),
                TensorStorageViewMut::BF16(mut data) => data.mapv_inplace(|v| -v),
                TensorStorageViewMut::I8(mut data, s) => {
                    assert_eq!(Some(s), scale);
                    data.mapv_inplace(|v| -v);
                }
                _ => panic!("native dtype was changed by a mutable view"),
            });
            assert_eq!(x.dtype(), dtype);
            assert_eq!(x.quantization_scale(), scale);
            assert_eq!(values(&x), before.iter().map(|v| -*v).collect::<Vec<_>>());
        }
    });
}

fn raw_values(raw: TensorRawData) -> Vec<f32> {
    match raw {
        TensorRawData::F32(v) => v,
        TensorRawData::F16(v) => v
            .into_iter()
            .map(|x| half::f16::from_bits(x).to_f32())
            .collect(),
        TensorRawData::BF16(v) => v
            .into_iter()
            .map(|x| half::bf16::from_bits(x).to_f32())
            .collect(),
        TensorRawData::I8 { values, scale } => {
            values.into_iter().map(|x| x as f32 * scale).collect()
        }
    }
}

#[cfg(feature = "cuda")]
fn updated(dtype: DType) -> Tensor {
    use crate::optim::{Optimizer, SGD};
    let x = tensor(&[2, 2], &[1., -2., 0.5, 3.], dtype, 0, Device::Cuda);
    x.add_grad(array(&[0.5, -1., 2., 0.25]));
    SGD::new_with_dtype(vec![x.clone()], 0.1, DType::F32).step();
    x
}

#[cfg(feature = "cuda")]
fn snapshots<const OP: usize>(device: Device) {
    for dtype in [DType::F32, DType::F16, DType::BF16, DType::I8] {
        let x = updated(dtype);
        let expected = values(&x);
        let actual = match OP {
            0 => raw_values(x.export_raw().2),
            1 => values(&x.to_cpu()),
            2 => values(&x.detach()),
            3 => {
                x.to_cpu_inplace();
                values(&x)
            }
            _ => unreachable!(),
        };
        check(
            "CUDA updated snapshot",
            &actual,
            &expected.iter().map(|&v| v as f64).collect::<Vec<_>>(),
            1e-7,
            1e-7,
        );
        assert_eq!(x.dtype(), dtype);
        assert_eq!(x.device(), if OP == 3 { Device::Cpu } else { device });
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_export_reads_updated_resident_values() {
    run(Device::Cuda, snapshots::<0>);
}
#[cfg(feature = "cuda")]
#[test]
fn cuda_to_cpu_copy_reads_updated_resident_values() {
    run(Device::Cuda, snapshots::<1>);
}
#[cfg(feature = "cuda")]
#[test]
fn cuda_detach_reads_updated_resident_values() {
    run(Device::Cuda, snapshots::<2>);
}
#[cfg(feature = "cuda")]
#[test]
fn cuda_inplace_cpu_transfer_reads_updated_resident_values() {
    run(Device::Cuda, snapshots::<3>);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_import_clears_old_gradient_before_optimizer_step() {
    use crate::optim::{Optimizer, SGD};
    run(Device::Cuda, |device| {
        for dtype in [DType::F32, DType::F16, DType::BF16, DType::I8] {
            let x = tensor(&[2, 2], &[1.; 4], dtype, 0, device);
            x.add_grad(array(&[5.; 4]));
            let replacement = tensor(&[2, 2], &[2.; 4], dtype, 0, Device::Cpu);
            let (shape, dt, raw) = replacement.export_raw();
            x.import_raw(shape, dt, raw).unwrap();
            assert!(
                !x.has_pending_grad(),
                "old gradient survived import for {dtype:?}"
            );
            SGD::new_with_dtype(vec![x.clone()], 0.1, DType::F32).step();
            assert_eq!(values(&x), values(&replacement));
        }
    });
}

#[test]
fn cpu_snapshot_covers_native_dtype_and_strided_layout() {
    run(Device::Cpu, |device| {
        for dtype in [DType::F32, DType::F16, DType::BF16, DType::I8] {
            for layout in 0..3 {
                let x = tensor(&[2, 2], &[0.25, -0.5, 0.75, -1.], dtype, layout, device);
                assert_eq!(raw_values(x.export_raw().2), values(&x));
                assert_eq!(values(&x.detach()), values(&x));
            }
        }
    });
}
