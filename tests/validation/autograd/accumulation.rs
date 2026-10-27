use crate::accuracy_gate::{run, tensor};
use crate::autograd::{Device, Tensor};
use crate::precision::DType;
use ndarray::{ArrayD, IxDyn};
use std::cell::Cell;

thread_local! {
    static FAIL_OPERATION: Cell<Option<&'static str>> = const { Cell::new(None) };
}

pub(crate) fn check_cuda_failure(operation: &str) -> Result<(), String> {
    FAIL_OPERATION.with(|failure| {
        if failure.get() == Some(operation) {
            failure.set(None);
            return Err(format!("injected CUDA gradient failure: {operation}"));
        }
        Ok(())
    })
}

fn array(values: &[f32]) -> ArrayD<f32> {
    ArrayD::from_shape_vec(IxDyn(&[2, 2]), values.to_vec()).unwrap()
}

#[test]
fn regression_grad_ref_reports_pending_gradient() {
    use crate::ops::cuda;
    for device in [Device::Cpu, Device::Cuda] {
        if device == Device::Cuda && !cuda::is_available() {
            continue;
        }
        run(device, |device| {
            let x = tensor(&[2, 2], &[0.; 4], DType::F32, 0, device);
            assert!(x.grad_ref().is_none());
            if device == Device::Cuda {
                x.backward_with_cuda_grad(cuda::upload_f32(&[1., -2., 3., -4.]).unwrap());
                assert!(!x.has_host_grad());
            } else {
                x.backward_with_grad(array(&[1., -2., 3., -4.]));
            }
            let first = x.grad_ref();
            assert_eq!(
                first
                    .as_ref()
                    .expect("grad_ref missed resident gradient")
                    .to_owned(),
                array(&[1., -2., 3., -4.])
            );
            let second = x.grad_ref();
            assert_eq!(first.as_ref(), second.as_ref());
        });
    }
}

#[cfg(feature = "cuda")]
#[test]
fn regression_cuda_reused_upstream_seed_accumulates_linearly() {
    use crate::ops::cuda;
    run(Device::Cuda, |device| {
        for dtype in [DType::F32, DType::F16, DType::BF16, DType::I8] {
            for through_reshape in [false, true] {
                let x = tensor(&[2, 2], &[0.; 4], dtype, 0, device);
                let out = if through_reshape {
                    crate::ops::shape::reshape(&x, vec![4])
                } else {
                    x.clone()
                };
                let seed = cuda::upload_f32(&[1., -2., 3., -4.]).unwrap();
                let downloads = cuda::test_host_downloads();
                for _ in 0..3 {
                    out.backward_with_cuda_grad(seed.clone());
                }
                assert_eq!(cuda::test_host_downloads(), downloads);
                assert!(!x.has_host_grad());
                assert_eq!(
                    cuda::download_f32(&x.cloned_cuda_f32_grad().unwrap()).unwrap(),
                    [3., -6., 9., -12.]
                );
                assert_eq!(cuda::download_f32(&seed).unwrap(), [1., -2., 3., -4.]);
                if through_reshape {
                    assert_eq!(
                        cuda::download_f32(&out.cloned_cuda_f32_grad().unwrap()).unwrap(),
                        [1., -2., 3., -4.]
                    );
                }
                let fresh = tensor(&[2, 2], &[0.; 4], dtype, 0, device);
                fresh.backward_with_cuda_grad(seed.clone());
                cuda::upload_f32_offset(&seed, 0, &[9.; 4]).unwrap();
                assert_eq!(
                    cuda::download_f32(&fresh.cloned_cuda_f32_grad().unwrap()).unwrap(),
                    [1., -2., 3., -4.]
                );
            }
        }
    });
}

#[cfg(feature = "cuda")]
#[test]
fn regression_cuda_shared_accumulators_remain_independent() {
    use crate::ops::cuda;
    run(Device::Cuda, |device| {
        for dtype in [DType::F32, DType::F16, DType::BF16, DType::I8] {
            for paired in [false, true] {
                let left = tensor(&[2, 2], &[0.; 4], dtype, 0, device);
                let right = tensor(&[2, 2], &[0.; 4], dtype, 0, device);
                let shared = cuda::upload_f32(&[1., -2., 3., -4.]).unwrap();
                left.add_cuda_grad_buffer_only(shared.clone());
                right.add_cuda_grad_buffer_only(shared);
                let snapshot = right.grad_arc().unwrap();
                let incoming = [5., -6., 7., -8.];
                let next = cuda::upload_f32(&incoming).unwrap();
                if paired {
                    left.add_grad_with_cuda_buffer(array(&incoming), Some(next));
                } else {
                    left.add_cuda_grad_buffer_only(next);
                }
                assert_eq!(
                    cuda::download_f32(&right.cloned_cuda_f32_grad().unwrap()).unwrap(),
                    [1., -2., 3., -4.]
                );
                assert_eq!(snapshot.to_owned(), array(&[1., -2., 3., -4.]));
                assert_eq!(left.grad().unwrap(), array(&[6., -8., 10., -12.]));
                assert_eq!(snapshot.to_owned(), array(&[1., -2., 3., -4.]));

                // Unique buffers keep the allocation-free accumulation path.
                let unique = tensor(&[2, 2], &[0.; 4], dtype, 0, device);
                let owned_seed = cuda::upload_f32(&[1.; 4]).unwrap();
                let handle = owned_seed.handle();
                unique.backward_with_cuda_grad(owned_seed);
                let downloads = cuda::test_host_downloads();
                if paired {
                    unique.add_grad_with_cuda_buffer(
                        array(&incoming),
                        Some(cuda::upload_f32(&incoming).unwrap()),
                    );
                } else {
                    unique.add_cuda_grad_buffer_only(cuda::upload_f32(&incoming).unwrap());
                }
                assert_eq!(cuda::test_host_downloads(), downloads);
                assert_eq!(unique.cloned_cuda_f32_grad().unwrap().handle(), handle);
                assert_eq!(unique.grad().unwrap(), array(&[6., -5., 8., -7.]));
                assert_eq!(
                    cuda::download_f32(&left.cloned_cuda_f32_grad().unwrap()).unwrap(),
                    [6., -8., 10., -12.]
                );
                right.add_cuda_grad_buffer_only(cuda::upload_f32(&[1.; 4]).unwrap());
                assert_eq!(right.grad().unwrap(), array(&[2., -1., 4., -3.]));
                assert_eq!(left.grad().unwrap(), array(&[6., -8., 10., -12.]));
            }
        }
    });
}

#[test]
fn cpu_frozen_input_allows_held_read_view_during_backward() {
    run(Device::Cpu, |device| {
        let x = tensor(&[2, 2], &[1., 2., 3., 4.], DType::F32, 0, device);
        let frozen = Tensor::from_data_with_grad_flag(array(&[2., -3., 4., -5.]), false);
        let out = (&x * &frozen).sum();
        let view = frozen.data_ref();
        out.backward();
        assert_eq!(x.grad().unwrap(), array(&[2., -3., 4., -5.]));
        assert_eq!(view.iter().copied().collect::<Vec<_>>(), [2., -3., 4., -5.]);
        assert!(!frozen.has_pending_grad());
    });
}

#[cfg(feature = "cuda")]
fn mixed_residency<const OLD_DEVICE: bool, const NEW_DEVICE: bool, const PAIRED: bool>() {
    use crate::ops::cuda;
    run(Device::Cuda, |device| {
        let old = [1., -2., 3., -4.];
        let new = [-5., 6., -7., 8.];
        let x = tensor(&[2, 2], &[0.; 4], DType::F32, 0, device);
        if OLD_DEVICE {
            x.add_cuda_grad_buffer_only(cuda::upload_f32(&old).unwrap());
        } else {
            x.add_grad(array(&old));
            // A valid host-only gradient, as installed through grad_mut().
            x.0.borrow_mut().cuda_f32_grad = None;
        }
        if NEW_DEVICE {
            let buffer = cuda::upload_f32(&new).unwrap();
            if PAIRED {
                x.add_grad_with_cuda_buffer(array(&new), Some(buffer));
            } else {
                x.add_cuda_grad_buffer_only(buffer);
            }
        } else {
            x.add_grad(array(&new));
        }
        assert_eq!(x.grad().unwrap(), array(&[-4., 4., -4., 4.]));
        assert_eq!(
            cuda::download_f32(&x.cloned_cuda_f32_grad().unwrap()).unwrap(),
            [-4., 4., -4., 4.]
        );
    });
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_device_only_then_host_gradient_accumulates() {
    mixed_residency::<true, false, false>();
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_device_only_then_paired_gradient_agrees_on_host_and_device() {
    mixed_residency::<true, true, true>();
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_host_only_then_paired_gradient_agrees_on_host_and_device() {
    mixed_residency::<false, true, true>();
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_host_only_then_device_gradient_accumulates() {
    mixed_residency::<false, true, false>();
}

fn frozen_read_views(device: Device) {
    let frozen = Tensor::from_data_with_grad_flag(array(&[2., -3., 4., -5.]), false);
    if device == Device::Cuda {
        frozen.to_device_inplace(device);
    }
    frozen.add_grad(array(&[7.; 4]));
    let data = frozen.data_ref();
    let grad = frozen.grad_ref();
    frozen.add_backward_grad(array(&[1.; 4]));
    if device == Device::Cuda {
        let buffer = crate::ops::cuda::upload_f32(&[1.; 4]).unwrap();
        frozen.add_backward_grad_with_cuda_buffer(array(&[1.; 4]), Some(buffer.clone()));
        frozen.add_backward_cuda_grad_buffer_only(buffer);
    }
    assert_eq!(data.iter().copied().collect::<Vec<_>>(), [2., -3., 4., -5.]);
    assert_eq!(
        grad.as_ref().unwrap().iter().copied().collect::<Vec<_>>(),
        [7.; 4]
    );
}

#[test]
fn cpu_frozen_checked_writers_preserve_held_views_and_manual_gradient() {
    run(Device::Cpu, frozen_read_views);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_frozen_checked_writers_preserve_held_views_and_manual_gradient() {
    run(Device::Cuda, frozen_read_views);
}

fn accumulation_matrix(device: Device) {
    use crate::accuracy_gate::ordered;
    use crate::ops::cuda;
    let residencies = if device == Device::Cuda { 3 } else { 1 };
    let routes = if device == Device::Cuda { 3 } else { 1 };
    for dtype in [DType::F32, DType::F16, DType::BF16, DType::I8] {
        for layout in 0..3 {
            for residency in 0..residencies {
                for route in 0..routes {
                    for checked in [false, true] {
                        let x = tensor(&[2, 2], &[0.; 4], dtype, layout, device);
                        let initial = [1.25, -2.5, 3.75, -4.];
                        x.add_grad(array(&initial));
                        let mut expected = initial.map(f64::from);
                        for step in 0..6 {
                            if device == Device::Cuda {
                                match residency {
                                    0 => drop(x.grad_mut()),
                                    1 => x.0.borrow_mut().grad = None,
                                    _ => {}
                                }
                            }
                            let incoming = (0..4)
                                .map(|i| ((step * 11 + i * 7) % 23) as f32 / 13. - 0.8)
                                .collect::<Vec<_>>();
                            let host = ordered(&[2, 2], &incoming, layout);
                            match (route, checked) {
                                (0, false) => x.add_grad(host),
                                (0, true) => x.add_backward_grad(host),
                                (1, false) => x.add_grad_with_cuda_buffer(
                                    host,
                                    Some(cuda::upload_f32(&incoming).unwrap()),
                                ),
                                (1, true) => x.add_backward_grad_with_cuda_buffer(
                                    host,
                                    Some(cuda::upload_f32(&incoming).unwrap()),
                                ),
                                (2, false) => x.add_cuda_grad_buffer_only(
                                    cuda::upload_f32(&incoming).unwrap(),
                                ),
                                (2, true) => x.add_backward_cuda_grad_buffer_only(
                                    cuda::upload_f32(&incoming).unwrap(),
                                ),
                                _ => unreachable!(),
                            }
                            for (sum, &v) in expected.iter_mut().zip(&incoming) {
                                *sum += v as f64;
                            }
                            let device_values = if device == Device::Cuda {
                                let buffer = x.cloned_cuda_f32_grad().expect("resident gradient");
                                Some(cuda::download_f32(&buffer).unwrap())
                            } else {
                                None
                            };
                            let host_values = x.grad().unwrap().iter().copied().collect::<Vec<_>>();
                            crate::accuracy_gate::check(
                                "mixed gradient host",
                                &host_values,
                                &expected,
                                2e-6,
                                2e-6,
                            );
                            if let Some(values) = &device_values {
                                crate::accuracy_gate::check(
                                    "mixed gradient device",
                                    values,
                                    &expected,
                                    2e-6,
                                    2e-6,
                                );
                            }
                            if std::env::var_os("LUMEN_OPERATOR_AUDIT").is_some() {
                                println!(
                                    "GRADIENT_ACCUMULATION_AUDIT={}",
                                    serde_json::json!({
                                        "device":format!("{device:?}"),"dtype":format!("{dtype:?}"),"layout":layout,
                                        "residency":residency,"route":route,"checked":checked,"step":step,
                                        "initial":initial,"incoming":incoming,"host":host_values,"cuda":device_values
                                    })
                                );
                            }
                        }
                        x.zero_grad();
                        assert!(!x.has_pending_grad());
                    }
                }
            }
        }
    }
}

#[test]
fn cpu_mixed_accumulation_matrix() {
    run(Device::Cpu, accumulation_matrix);
}

fn empty_gradients(device: Device) {
    for shape in [vec![0], vec![2, 0, 3], vec![0, 0]] {
        let x = tensor(&shape, &[], DType::F32, 0, device);
        x.backward();
        x.backward();
        x.add_backward_grad(ArrayD::zeros(IxDyn(&shape)));
        assert_eq!(x.grad().unwrap().shape(), shape);
        assert!(x.has_pending_grad());
        assert!(x.cloned_cuda_f32_grad().is_none());
        assert!(x.grad_mut().as_ref().unwrap().is_empty());
        x.zero_grad();
        assert!(!x.has_pending_grad());
    }
}

#[test]
fn cpu_empty_gradient_accumulation_preserves_shape() {
    run(Device::Cpu, empty_gradients);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_empty_gradient_accumulation_preserves_shape_without_device_allocation() {
    run(Device::Cuda, empty_gradients);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_mixed_accumulation_matrix() {
    run(Device::Cuda, accumulation_matrix);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_mutable_host_gradient_is_used_by_optimizer_and_can_be_cleared() {
    use crate::ops::cuda;
    use crate::optim::{Optimizer, SGD};
    run(Device::Cuda, |device| {
        for resident_only in [false, true] {
            let x = tensor(&[2, 2], &[1., 2., 3., 4.], DType::F32, 0, device);
            if resident_only {
                x.add_cuda_grad_buffer_only(cuda::upload_f32(&[1.; 4]).unwrap());
            } else {
                x.add_grad(array(&[1.; 4]));
            }
            {
                let mut grad = x.grad_mut();
                assert_eq!(
                    grad.as_ref().unwrap().iter().copied().collect::<Vec<_>>(),
                    [1.; 4]
                );
                *grad = Some(array(&[2., -3., 4., -5.]).into_shared());
            }
            assert!(x.0.borrow().cuda_f32_grad.is_none());
            let mut optimizer = SGD::new_with_dtype(vec![x.clone()], 0.1, DType::F32);
            optimizer.step();
            let values = crate::accuracy_gate::values(&x);
            crate::accuracy_gate::check(
                "edited gradient SGD",
                &values,
                &[0.8, 2.3, 2.6, 4.5],
                2e-6,
                2e-6,
            );
            *x.grad_mut() = None;
            assert!(!x.has_pending_grad());
            optimizer.step();
            assert_eq!(crate::accuracy_gate::values(&x), values);
        }
    });
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_accumulation_errors_fail_explicitly_without_substituting_new_gradient() {
    use crate::ops::cuda;
    use std::panic::{AssertUnwindSafe, catch_unwind};
    run(Device::Cuda, |device| {
        for failure in [
            "upload previous gradient",
            "accumulate resident gradient",
            "download accumulated gradient",
            "upload accumulated gradient",
            "upload edited gradient",
        ] {
            let x = tensor(&[2, 2], &[0.; 4], DType::F32, 0, device);
            x.add_grad(array(&[3.; 4]));
            match failure {
                "upload previous gradient" | "upload edited gradient" => drop(x.grad_mut()),
                "download accumulated gradient" => x.0.borrow_mut().grad = None,
                _ => {}
            }
            let new = cuda::upload_f32(&[5.; 4]).unwrap();
            FAIL_OPERATION.with(|slot| slot.set(Some(failure)));
            let result = catch_unwind(AssertUnwindSafe(|| match failure {
                "upload previous gradient" | "accumulate resident gradient" => {
                    x.add_cuda_grad_buffer_only(new)
                }
                "upload edited gradient" => {
                    x.cloned_cuda_f32_grad();
                }
                _ => x.add_grad(array(&[5.; 4])),
            }));
            assert!(result.is_err(), "failure was ignored: {failure}");
            assert!(
                FAIL_OPERATION.with(|slot| slot.get().is_none()),
                "failure was not reached: {failure}"
            );
            assert_eq!(x.grad().unwrap(), array(&[3.; 4]));
            assert_eq!(
                cuda::download_f32(&x.cloned_cuda_f32_grad().unwrap()).unwrap(),
                [3.; 4]
            );
        }
    });
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_invalid_incoming_gradient_does_not_change_old_total() {
    use crate::ops::cuda;
    use std::panic::{AssertUnwindSafe, catch_unwind};
    run(Device::Cuda, |device| {
        for route in 0..3 {
            let x = tensor(&[2, 2], &[0.; 4], DType::F32, 0, device);
            x.add_grad(array(&[3.; 4]));
            let short = cuda::upload_f32(&[5.; 3]).unwrap();
            assert!(
                catch_unwind(AssertUnwindSafe(|| match route {
                    0 => x.add_grad(ArrayD::zeros(IxDyn(&[4]))),
                    1 => x.add_grad_with_cuda_buffer(array(&[5.; 4]), Some(short)),
                    _ => x.add_cuda_grad_buffer_only(short),
                }))
                .is_err()
            );
            assert_eq!(x.grad().unwrap(), array(&[3.; 4]));
            assert_eq!(
                cuda::download_f32(&x.cloned_cuda_f32_grad().unwrap()).unwrap(),
                [3.; 4]
            );
        }
    });
}
