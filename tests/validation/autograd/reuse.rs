use crate::accuracy_gate::{check, run, tensor, values};
use crate::autograd::Tensor;
use crate::device::Device;
use crate::precision::DType;
use ndarray::{ArrayD, IxDyn};

fn audit(x: &Tensor, initial: &[f32], dtype: DType, order: usize, phase: &str) {
    if x.is_cuda() {
        assert!(
            !x.has_host_grad(),
            "strict CUDA leaf gradient must stay resident"
        );
        assert!(x.cloned_cuda_f32_grad().is_some());
    }
    if std::env::var_os("LUMEN_OPERATOR_AUDIT").is_some() {
        println!(
            "AUTOGRAD_REUSE_AUDIT={}",
            serde_json::json!({
                "device": format!("{:?}", x.device()), "dtype": format!("{dtype:?}"),
                "layout": order, "phase": phase, "x": initial,
                "dx": x.grad().unwrap().iter().copied().collect::<Vec<_>>()
            })
        );
    }
}

fn reuse_cases(device: Device) {
    for dtype in [DType::F32, DType::F16, DType::BF16, DType::I8] {
        for order in 0..3 {
            let x = tensor(&[2, 2], &[0.25, -0.5, 0.75, -1.0], dtype, order, device);
            let initial = values(&x);
            let square = &x * &x.reshape(vec![2, 2]);
            let loss = square.sum();
            for pass in 1..=3 {
                loss.backward();
                audit(&x, &initial, dtype, order, &format!("repeat-{pass}"));
                let expected = initial
                    .iter()
                    .map(|v| 2.0 * *v as f64 * pass as f64)
                    .collect::<Vec<_>>();
                check(
                    "repeated backward",
                    x.grad().unwrap().as_slice().unwrap(),
                    &expected,
                    2e-6,
                    2e-6,
                );
            }
            x.zero_grad();
            loss.backward();
            audit(&x, &initial, dtype, order, "zero-leaf");
            let expected = initial.iter().map(|v| 2.0 * *v as f64).collect::<Vec<_>>();
            check(
                "zero leaf then reuse",
                x.grad().unwrap().as_slice().unwrap(),
                &expected,
                2e-6,
                2e-6,
            );
            // A different loss reuses the same intermediate graph.
            let other_loss = (&square + &square).sum();
            other_loss.backward();
            audit(&x, &initial, dtype, order, "shared");
            let expected = initial.iter().map(|v| 6.0 * *v as f64).collect::<Vec<_>>();
            check(
                "shared subgraph",
                x.grad().unwrap().as_slice().unwrap(),
                &expected,
                2e-6,
                2e-6,
            );
            // A previous intermediate can itself become the next backward root.
            x.zero_grad();
            square.backward();
            audit(&x, &initial, dtype, order, "new-root");
            let expected = initial.iter().map(|v| 2.0 * *v as f64).collect::<Vec<_>>();
            check(
                "intermediate becomes root",
                x.grad().unwrap().as_slice().unwrap(),
                &expected,
                2e-6,
                2e-6,
            );

            x.zero_grad();
            // The explicit seed is deliberately non-contiguous in logical order.
            let upstream = ndarray::array![[0.5, 2.0], [-1.0, -0.25]]
                .reversed_axes()
                .into_dyn();
            square.backward_with_grad(upstream.clone());
            audit(&x, &initial, dtype, order, "explicit");
            let expected = initial
                .iter()
                .zip(upstream.iter())
                .map(|(&v, &g)| 2.0 * v as f64 * g as f64)
                .collect::<Vec<_>>();
            check(
                "explicit seed",
                x.grad().unwrap().as_slice().unwrap(),
                &expected,
                2e-6,
                2e-6,
            );
            square.backward();
            audit(&x, &initial, dtype, order, "explicit-then-ones");
            let expected = initial
                .iter()
                .zip(upstream.iter())
                .map(|(&v, &g)| 2.0 * v as f64 * (g as f64 + 1.0))
                .collect::<Vec<_>>();
            check(
                "fresh ones after explicit seed",
                x.grad().unwrap().as_slice().unwrap(),
                &expected,
                2e-6,
                2e-6,
            );
            other_loss.backward();
            audit(&x, &initial, dtype, order, "explicit-shared");
            let expected = initial
                .iter()
                .zip(upstream.iter())
                .map(|(&v, &g)| 2.0 * v as f64 * (g as f64 + 3.0))
                .collect::<Vec<_>>();
            check(
                "explicit shared graph",
                x.grad().unwrap().as_slice().unwrap(),
                &expected,
                2e-6,
                2e-6,
            );
            x.zero_grad();
            square.backward_with_grad(upstream);
            audit(&x, &initial, dtype, order, "explicit-reset");

            // A leaf used directly as the root must accumulate fresh seeds too.
            x.zero_grad();
            for pass in 1..=2 {
                x.backward();
                audit(&x, &initial, dtype, order, &format!("leaf-{pass}"));
                check(
                    "leaf root",
                    x.grad().unwrap().as_slice().unwrap(),
                    &[pass as f64; 4],
                    0.0,
                    0.0,
                );
            }
        }
    }
}

#[test]
fn cpu_graph_reuse_accumulates_only_fresh_leaf_gradients() {
    run(Device::Cpu, reuse_cases);
}

#[test]
fn aliased_cpu_data_reads_and_square_do_not_conflict() {
    run(Device::Cpu, |device| {
        for dtype in [DType::F32, DType::F16, DType::BF16, DType::I8] {
            for order in 0..3 {
                let x = tensor(&[2, 2], &[2.0, -3.0, 0.5, 1.0], dtype, order, device);
                let initial = values(&x);
                let first = x.data_ref();
                let second = x.data_ref();
                assert_eq!(*first, *second);
                drop((first, second));
                (&x * &x).sum().backward();
                check(
                    "aliased square",
                    x.grad().unwrap().as_slice().unwrap(),
                    &initial.iter().map(|v| 2.0 * *v as f64).collect::<Vec<_>>(),
                    1e-6,
                    1e-6,
                );
            }
        }
    });
}

#[test]
fn invalid_upstream_shape_does_not_clear_existing_gradients() {
    run(Device::Cpu, |device| {
        let x = tensor(&[2, 2], &[2.0, -3.0, 0.5, 1.0], DType::F32, 0, device);
        let square = &x * &x;
        square.backward();
        let before = x.grad().unwrap();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            square.backward_with_grad(ArrayD::ones(IxDyn(&[4])));
        }));
        assert!(result.is_err());
        assert_eq!(x.grad().unwrap(), before);
    });
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_graph_reuse_accumulates_only_fresh_leaf_gradients() {
    run(Device::Cuda, reuse_cases);
}

fn graph_lifetime_cases(device: Device) {
    use crate::layers::{
        activation::{ReLU, Softmax},
        attention::{encoding::RotaryEmbedding, self_attention::repeat_kv},
        basic::embedding::Embedding,
        norm::rms_norm::RMSNorm,
    };
    use crate::module::Module;
    use crate::ops::{convolution, fused, shape};
    use std::collections::HashSet;
    use std::rc::Rc;

    for op in 0..if device == Device::Cuda { 15 } else { 14 } {
        for backward in [false, true] {
            for _repeat in 0..3 {
                let x = tensor(&[2, 2], &[0.25, -0.5, 0.75, -1.0], DType::F32, 0, device);
                let y = match op {
                    0 => &x * &x,
                    1 => x.reshape(vec![4]),
                    2 => x.transpose(0, 1),
                    3 => shape::cat(&[x.clone(), x.clone()], 1),
                    4 => shape::slice_last_dim(&x, 0, 1),
                    5 => ReLU::new().forward(x.clone()),
                    6 => Softmax::new(1).forward(x.clone()),
                    7 => RMSNorm {
                        weight: tensor(&[2], &[1.0, 0.75], DType::F32, 0, device),
                        eps: 1e-5,
                    }
                    .forward(x.clone()),
                    8 => Embedding {
                        weight: x.clone(),
                        vocab_size: 2,
                        embed_dim: 2,
                    }
                    .forward(&tensor(&[2], &[0.0, 1.0], DType::F32, 0, device)),
                    9 => {
                        RotaryEmbedding::new(2, 4, 10000.0).forward(&x.reshape(vec![1, 1, 2, 2]), 0)
                    }
                    10 => repeat_kv(x.reshape(vec![1, 1, 2, 2]), 2),
                    11 => convolution::conv2d(
                        &x.reshape(vec![1, 1, 2, 2]),
                        &tensor(&[1, 1, 1, 1], &[0.5], DType::F32, 0, device),
                        None,
                        (1, 1),
                        (0, 0),
                    ),
                    12 => convolution::max_pool2d(&x.reshape(vec![1, 1, 2, 2]), (2, 2), (1, 1)),
                    13 => fused::fused_softmax(&x.reshape(vec![1, 1, 2, 2]), 0.7, false),
                    _ => {
                        let w = tensor(&[2, 2], &[0.5, 0.25, -0.75, 1.0], DType::F32, 0, device);
                        let (q, k, v) = fused::fused_qkv_train_tensors(
                            &x.reshape(vec![1, 2, 2]),
                            &w,
                            &w,
                            &w,
                            None,
                            None,
                            None,
                            1,
                            1,
                        )
                        .expect("native CUDA training QKV");
                        shape::cat(&[q, k, v], 1)
                    }
                };
                let mut seen = HashSet::new();
                let mut pending = vec![y.clone()];
                let mut weak_nodes = Vec::new();
                while let Some(node) = pending.pop() {
                    if seen.insert(Rc::as_ptr(&node.0) as usize) {
                        weak_nodes.push(Rc::downgrade(&node.0));
                        pending.extend(node.0.borrow().parents.iter().cloned());
                    }
                }
                drop(x);
                if backward {
                    y.sum().backward();
                }
                drop(y);
                assert!(
                    weak_nodes.iter().all(|node| node.upgrade().is_none()),
                    "graph retained by backward closure: op {op}, backward={backward}"
                );
            }
        }
    }
}

#[test]
fn cpu_backward_graph_releases_output_and_parents() {
    run(Device::Cpu, graph_lifetime_cases);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_backward_graph_releases_output_and_parents() {
    run(Device::Cuda, graph_lifetime_cases);
}
