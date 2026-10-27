//! Public-API CPU matmul VJP timing with caller-owned workers and explicit budgets.
use lumen::autograd::Tensor;
use lumen::cpu::CpuKernelContext;
use lumen::ops::matmul::{batch_matmul, matmul};
use lumen::precision::DType;
use ndarray::{ArrayD, Dimension, IxDyn};
use serde_json::json;
use std::{hint::black_box, num::NonZeroUsize, sync::Arc, time::Instant};

fn data(shape: &[usize], layout: usize, seed: usize) -> ArrayD<f32> {
    let array = ArrayD::from_shape_fn(IxDyn(shape), |index| {
        let i = index
            .slice()
            .iter()
            .zip(shape)
            .fold(0usize, |a, (&i, &n)| a * n + i);
        (((i * 17 + seed) % 251) as f32 - 125.) * 0.00371
    });
    if layout == 0 {
        array
    } else {
        array
            .reversed_axes()
            .as_standard_layout()
            .to_owned()
            .reversed_axes()
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args == ["--help"] {
        println!(
            "cpu_backward_bench [--threads N] [--tasks N] [--warmup N] [--samples N] [--iters N]; LUMEN_BENCH_DUMP=1 dumps full gradients after timing"
        );
        return Ok(());
    }
    let mut values = std::collections::HashMap::new();
    for pair in args.chunks(2) {
        if pair.len() != 2
            || !matches!(
                pair[0].as_str(),
                "--threads" | "--tasks" | "--warmup" | "--samples" | "--iters"
            )
        {
            return Err("expected known argument/value pairs".into());
        }
        if values
            .insert(pair[0].as_str(), pair[1].parse::<usize>()?)
            .is_some()
        {
            return Err(format!("duplicate argument: {}", pair[0]).into());
        }
    }
    let threads = *values.get("--threads").unwrap_or(&4);
    let tasks = *values.get("--tasks").unwrap_or(&threads);
    let warmup = *values.get("--warmup").unwrap_or(&20);
    let samples = *values.get("--samples").unwrap_or(&9);
    let iters = *values.get("--iters").unwrap_or(&10);
    if threads == 0 || tasks == 0 || samples == 0 || iters == 0 {
        return Err("threads, tasks, samples and iters must be positive".into());
    }
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()?,
    );
    let context =
        CpuKernelContext::with_pool(pool).with_task_limit(NonZeroUsize::new(tasks).unwrap());
    let dump = std::env::var_os("LUMEN_BENCH_DUMP").is_some();
    let pairs = [
        (DType::F32, DType::F32),
        (DType::F16, DType::F16),
        (DType::BF16, DType::BF16),
        (DType::I8, DType::I8),
        (DType::F32, DType::BF16),
    ];
    let shapes = [
        (vec![9, 37], vec![19, 37], 0),
        (vec![193, 257], vec![257, 257], 1),
        (vec![2, 3, 17, 13], vec![2, 3, 13, 19], 1),
        (vec![2, 3, 65, 129], vec![2, 3, 129, 257], 0),
    ];
    for (input_dtype, weight_dtype) in pairs {
        for (case, (a_shape, b_shape, layout)) in shapes.iter().enumerate() {
            let a = Tensor::new_with_dtype(data(a_shape, *layout, 3), input_dtype);
            let b = Tensor::new_with_dtype(data(b_shape, *layout, 7), weight_dtype);
            let out = context.scope(|| {
                if case < 2 {
                    matmul(&a, &b)
                } else {
                    batch_matmul(&a, &b)
                }
            });
            let seed = data(&out.shape_vec(), *layout, 11);
            let run = || {
                a.zero_grad();
                b.zero_grad();
                out.backward_with_grad(seed.clone());
                black_box(());
            };
            context.scope(|| {
                for _ in 0..warmup {
                    for _ in 0..iters {
                        run();
                    }
                }
            });
            let timings = context.scope(|| {
                (0..samples)
                    .map(|_| {
                        let start = Instant::now();
                        for _ in 0..iters {
                            run();
                        }
                        start.elapsed().as_secs_f64() * 1000. / iters as f64
                    })
                    .collect::<Vec<_>>()
            });
            let gradients = [&a,&b].map(|t| {
                let grad=t.grad().unwrap();
                assert!(grad.iter().all(|v|v.is_finite()));
                json!({"shape":t.shape_vec(),"sum_f64":grad.iter().map(|&v|v as f64).sum::<f64>(),"values":if dump {Some(grad.iter().copied().collect::<Vec<_>>())}else{None}})
            });
            println!(
                "{}",
                json!({"op":if case<2 {"matmul"}else{"batch_matmul"},"case":case,"input_dtype":format!("{input_dtype:?}"),"weight_dtype":format!("{weight_dtype:?}"),"input_shape":a_shape,"weight_shape":b_shape,"layout":layout,"threads":threads,"tasks":tasks,"warmup":warmup,"samples":samples,"iters":iters,"samples_ms":timings,"gradients":gradients})
            );
        }
    }
    Ok(())
}
