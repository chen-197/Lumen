//! Focused CPU measurements. Each JSON line includes every warmed timing sample.
use half::{bf16, f16};
use lumen::autograd::{Tensor, no_grad};
use lumen::ops::fused::{fused_gate_up_silu_infer_into, fused_qkv_decode_infer_into};
use lumen::ops::matmul::{SliceRef, matmul_transposed_rhs, matvec_rowmajor_parallel_mixed};
use lumen::precision::{DType, set_allow_parameter_dtype_copies};
use std::{hint::black_box, time::Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let value = |name: &str, default: &str| -> Result<String, String> {
        match args.iter().position(|s| s == name) {
            Some(i) => args
                .get(i + 1)
                .cloned()
                .ok_or_else(|| format!("{name} needs a value")),
            None => Ok(default.to_string()),
        }
    };
    let kind = value("--kind", "matmul_transposed_rhs")?;
    let dtype = value("--dtype", "f16")?;
    let m = value("--m", "46")?.parse::<usize>()?;
    let k = value("--k", "2048")?.parse::<usize>()?;
    let n = value("--n", "2048")?.parse::<usize>()?;
    let threads = value("--threads", "8")?.parse::<usize>()?;
    let iters = value("--iters", "10")?.parse::<usize>()?;
    let samples = value("--samples", "9")?.parse::<usize>()?;
    let warmup = value("--warmup", "10")?.parse::<usize>()?;
    if [m, k, n, threads, iters, samples].contains(&0) {
        return Err("dimensions, threads, iters and samples must be positive".into());
    }
    if !["matmul_transposed_rhs", "matvec", "qkv", "gate"].contains(&kind.as_str()) {
        return Err("unknown kind".into());
    }
    if kind != "matmul_transposed_rhs" && m != 1 {
        return Err("decode cases require --m 1".into());
    }
    let dt = match dtype.as_str() {
        "f32" => DType::F32,
        "f16" => DType::F16,
        "bf16" => DType::BF16,
        "i8" => DType::I8,
        _ => return Err("unknown dtype".into()),
    };
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()?;
    set_allow_parameter_dtype_copies(false);
    let data = |len: usize, seed: usize| -> Vec<f32> {
        (0..len)
            .map(|i| (((i.wrapping_mul(17) + seed) % 257) as f32 - 128.) / 256.)
            .collect()
    };
    let input = data(m.checked_mul(k).ok_or("input size overflow")?, 11);
    let weights = data(n.checked_mul(k).ok_or("weight size overflow")?, 23);
    let tensor = |shape: Vec<usize>, values: Vec<f32>, dtype| {
        let t = Tensor::from_array_no_grad(ndarray::ArrayD::from_shape_vec(shape, values).unwrap());
        t.cast_inplace(dtype);
        t
    };
    // Keep activations F32 (BF16 for BF16 weights) to expose conversion/packing
    // costs separately from a full-model precision policy.
    let input_dtype = if dt == DType::BF16 {
        DType::BF16
    } else {
        DType::F32
    };
    let shape = if kind == "qkv" || kind == "gate" {
        vec![1, 1, k]
    } else {
        vec![m, k]
    };
    let a = tensor(shape, input.clone(), input_dtype);
    let b = tensor(vec![n, k], weights.clone(), dt);
    let wf: Vec<f16> = weights.iter().copied().map(f16::from_f32).collect();
    let wb: Vec<bf16> = weights.iter().copied().map(bf16::from_f32).collect();
    let xb: Vec<bf16> = input.iter().copied().map(bf16::from_f32).collect();
    let wi: Vec<i8> = weights.iter().map(|v| (v * 128.).round() as i8).collect();
    let lhs = if dt == DType::BF16 {
        SliceRef::BF16(&xb)
    } else {
        SliceRef::F32(&input)
    };
    let rhs = match dt {
        DType::F32 => SliceRef::F32(&weights),
        DType::F16 => SliceRef::F16(&wf),
        DType::BF16 => SliceRef::BF16(&wb),
        DType::I8 => SliceRef::I8(&wi, 1. / 128.),
    };
    let mut out = vec![0.; n];
    let mut ko = out.clone();
    let mut vo = out.clone();
    no_grad(|| {
        let mut run = || match kind.as_str() {
            "matmul_transposed_rhs" => {
                black_box(matmul_transposed_rhs(&a, &b));
            }
            "matvec" => {
                matvec_rowmajor_parallel_mixed(lhs, rhs, n, k, &mut out);
                black_box(&out);
            }
            "qkv" => {
                fused_qkv_decode_infer_into(&a, &b, &b, &b, &mut out, &mut ko, &mut vo);
                black_box(&out);
            }
            "gate" => {
                fused_gate_up_silu_infer_into(&a, &b, &b, &mut out);
                black_box(&out);
            }
            _ => unreachable!(),
        };
        for _ in 0..warmup {
            run();
        }
        let mut times = Vec::with_capacity(samples);
        for _ in 0..samples {
            let start = Instant::now();
            for _ in 0..iters {
                run();
            }
            times.push(start.elapsed().as_secs_f64() * 1e6 / iters as f64);
        }
        let mut ordered = times.clone();
        ordered.sort_by(f64::total_cmp);
        println!(
            "{}",
            serde_json::json!({"kind":kind,"dtype":dtype,"m":m,"k":k,"n":n,"threads":threads,"iters":iters,"warmup":warmup,"samples_us":times,"median_us":ordered[samples/2]})
        );
    });
    Ok(())
}
