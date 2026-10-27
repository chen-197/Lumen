//! Warmed, shared-pool scaling of independent caller-owned CPU requests.
use lumen::autograd::{Tensor, no_grad};
use lumen::cpu::CpuKernelContext;
use lumen::layers::attention::self_attention::SelfAttention;
use lumen::ops::matmul::matmul;
use lumen::precision::{DType, set_allow_parameter_dtype_copies};
use std::hint::black_box;
use std::num::NonZeroUsize;
use std::sync::{Arc, Barrier};
use std::time::Instant;

#[derive(Clone)]
struct Args {
    kind: String,
    dtype: DType,
    m: usize,
    k: usize,
    n: usize,
    batch: usize,
    heads: usize,
    kv_heads: usize,
    threads: usize,
    requests: usize,
    tasks: Option<NonZeroUsize>,
    warmup: usize,
    iters: usize,
    samples: usize,
}

fn tensor(shape: &[usize], seed: usize, dtype: DType) -> Tensor {
    let count: usize = shape.iter().product();
    let values = (0..count)
        .map(|i| (((i.wrapping_mul(17).wrapping_add(seed)) % 257) as f32 - 128.) / 1024.)
        .collect();
    let t = Tensor::from_array_no_grad(ndarray::ArrayD::from_shape_vec(shape, values).unwrap());
    t.cast_inplace(dtype);
    t
}

fn values(t: Tensor) -> Vec<f32> {
    t.data_ref().iter().copied().collect()
}

fn run_request(args: &Args, policy: CpuKernelContext, barrier: &Barrier) -> serde_json::Value {
    // Tensor/Rc and precision/autograd state are created on each calling thread.
    // Only numerical slices enter the common worker pool.
    set_allow_parameter_dtype_copies(false);
    let input_dtype = if args.dtype == DType::I8 {
        DType::BF16
    } else {
        args.dtype
    };
    let input = tensor(
        &if args.kind == "attention" {
            vec![args.batch, args.m, args.k]
        } else {
            vec![args.m, args.k]
        },
        11,
        input_dtype,
    );
    let weights;
    let attention;
    let mut run: Box<dyn FnMut() -> Tensor> = if args.kind == "attention" {
        let mut layer = SelfAttention::new_with_dtypes(
            args.k,
            args.heads,
            args.kv_heads,
            args.m,
            10000.,
            true,
            args.dtype,
            input_dtype,
        );
        for (seed, projection) in [
            (23, &mut layer.w_q),
            (37, &mut layer.w_k),
            (53, &mut layer.w_v),
            (71, &mut layer.w_o),
        ] {
            projection.weight = tensor(&projection.weight.shape_vec(), seed, args.dtype);
        }
        attention = layer;
        // Includes projections, RoPE, attention and a fresh cache per prefill.
        Box::new(|| attention.forward(input.clone(), None).0)
    } else {
        weights = tensor(&[args.n, args.k], 23, args.dtype);
        Box::new(|| matmul(&input, &weights))
    };
    no_grad(|| {
        // Check against a serial integrated-kernel policy outside measured work.
        let serial = policy
            .clone()
            .with_task_limit(NonZeroUsize::new(1).unwrap());
        let reference = values(serial.scope(&mut run));
        let checksum: f64 = reference.iter().map(|&x| x as f64).sum();
        let mut max_error = 0.0f32;
        let mut check = |output: Tensor| {
            let actual = values(output);
            assert_eq!(actual.len(), reference.len());
            for (&a, &b) in actual.iter().zip(&reference) {
                let error = (a - b).abs();
                assert!(
                    a.is_finite() && b.is_finite() && error <= 0.002 * (1. + b.abs()),
                    "output mismatch: {a} vs {b}"
                );
                max_error = max_error.max(error);
            }
        };
        for _ in 0..args.warmup {
            check(policy.scope(&mut run));
        }
        let mut timings = Vec::with_capacity(args.samples);
        for _ in 0..args.samples {
            barrier.wait(); // ready; coordinator starts the wall clock
            barrier.wait(); // release all requests together
            let started = Instant::now();
            let output = policy.scope(|| {
                for _ in 1..args.iters {
                    black_box(run());
                }
                run()
            });
            timings.push(started.elapsed().as_secs_f64() * 1e6 / args.iters as f64);
            barrier.wait(); // output checking is outside both timed regions
            check(output);
        }
        serde_json::json!({"samples_us":timings,"reference_checksum":checksum,"max_abs_error":max_error})
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let allowed = [
        "--kind",
        "--dtype",
        "--m",
        "--k",
        "--n",
        "--batch",
        "--heads",
        "--kv-heads",
        "--threads",
        "--requests",
        "--tasks",
        "--warmup",
        "--iters",
        "--samples",
    ];
    if !raw.len().is_multiple_of(2) || raw.chunks(2).any(|p| !allowed.contains(&p[0].as_str())) {
        return Err("expected supported option/value pairs".into());
    }
    let value = |name: &str, default: &str| {
        raw.chunks(2)
            .find(|p| p[0] == name)
            .map_or_else(|| default.to_string(), |p| p[1].clone())
    };
    let number = |name: &str, default: &str| -> Result<usize, Box<dyn std::error::Error>> {
        Ok(value(name, default).parse()?)
    };
    let dtype_name = value("--dtype", "f16");
    let args = Args {
        kind: value("--kind", "matmul"),
        dtype: match dtype_name.as_str() {
            "f32" => DType::F32,
            "f16" => DType::F16,
            "bf16" => DType::BF16,
            "i8" => DType::I8,
            _ => return Err("unknown dtype".into()),
        },
        m: number("--m", "128")?,
        k: number("--k", "2048")?,
        n: number("--n", "2048")?,
        batch: number("--batch", "1")?,
        heads: number("--heads", "8")?,
        kv_heads: number("--kv-heads", "2")?,
        threads: number("--threads", "8")?,
        requests: number("--requests", "1")?,
        tasks: match value("--tasks", "auto").as_str() {
            "auto" => None,
            v => Some(NonZeroUsize::new(v.parse()?).ok_or("tasks must be positive")?),
        },
        warmup: number("--warmup", "3")?,
        iters: number("--iters", "3")?,
        samples: number("--samples", "7")?,
    };
    if [
        args.m,
        args.k,
        args.n,
        args.batch,
        args.heads,
        args.kv_heads,
        args.threads,
        args.requests,
        args.warmup,
        args.iters,
        args.samples,
    ]
    .contains(&0)
    {
        return Err("dimensions, workers, requests and sampling counts must be positive".into());
    }
    if !["matmul", "attention"].contains(&args.kind.as_str()) {
        return Err("kind must be matmul or attention".into());
    }
    if args.kind == "attention"
        && (!args.k.is_multiple_of(args.heads)
            || !args.heads.is_multiple_of(args.kv_heads)
            || !(args.k / args.heads).is_multiple_of(2))
    {
        return Err("attention requires compatible heads and an even head dimension".into());
    }
    args.m.checked_mul(args.k).ok_or("input size overflow")?;
    args.n.checked_mul(args.k).ok_or("weight size overflow")?;
    args.batch
        .checked_mul(args.m)
        .and_then(|v| v.checked_mul(args.k))
        .ok_or("batch size overflow")?;
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build()?,
    );
    let mut policy = CpuKernelContext::with_pool(pool);
    if let Some(limit) = args.tasks {
        policy = policy.with_task_limit(limit);
    }
    let barrier = Barrier::new(
        args.requests
            .checked_add(1)
            .ok_or("request count overflow")?,
    );
    let (requests, wall_us) = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..args.requests)
            .map(|_| {
                let policy = policy.clone();
                let args = &args;
                let barrier = &barrier;
                scope.spawn(move || run_request(args, policy, barrier))
            })
            .collect();
        let wall_us: Vec<f64> = (0..args.samples)
            .map(|_| {
                barrier.wait();
                let started = Instant::now();
                barrier.wait();
                barrier.wait();
                started.elapsed().as_secs_f64() * 1e6
            })
            .collect();
        let requests: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        (requests, wall_us)
    });
    let mut ordered = wall_us.clone();
    ordered.sort_by(f64::total_cmp);
    let median = ordered[ordered.len() / 2];
    println!(
        "{}",
        serde_json::json!({
            "kind":args.kind,"dtype":dtype_name,"m":args.m,"k":args.k,"n":args.n,
            "batch":args.batch,"heads":args.heads,"kv_heads":args.kv_heads,
            "pool_threads":args.threads,"requests":args.requests,"tasks":args.tasks.map(NonZeroUsize::get),
            "warmup":args.warmup,"iters":args.iters,"per_request":requests,"wall_samples_us":wall_us,
            "median_batch_us":median,"calls_per_second":args.requests as f64 * args.iters as f64 * 1e6/median,
            "attention_includes_projections_and_fresh_cache":args.kind=="attention",
            "timing_note":"wall includes barrier release/completion; per-request checks are outside timing"
        })
    );
    Ok(())
}
