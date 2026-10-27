//! Decode measurements with distinct fused weights and rotating working sets.
use half::{bf16, f16};
use lumen::autograd::{Tensor, no_grad};
use lumen::ops::fused::{fused_gate_up_silu_infer_into, fused_qkv_decode_infer_into};
use lumen::ops::matmul::{
    SliceRef, matvec_argmax_rowmajor_parallel_mixed, matvec_rowmajor_parallel_mixed,
};
use lumen::precision::{DType, set_allow_parameter_dtype_copies};
use std::{hint::black_box, time::Instant};

enum Native {
    F32(Vec<f32>),
    F16(Vec<f16>),
    BF16(Vec<bf16>),
    I8(Vec<i8>, f32),
}

impl Native {
    fn new(data: Vec<f32>, dtype: DType) -> Self {
        match dtype {
            DType::F32 => Self::F32(data),
            DType::F16 => Self::F16(data.into_iter().map(f16::from_f32).collect()),
            DType::BF16 => Self::BF16(data.into_iter().map(bf16::from_f32).collect()),
            DType::I8 => {
                let max = data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                let scale = if max > 0. { max / 127. } else { 1. };
                Self::I8(
                    data.into_iter()
                        .map(|v| (v / scale).round().clamp(-127., 127.) as i8)
                        .collect(),
                    scale,
                )
            }
        }
    }
    fn slice(&self) -> SliceRef<'_> {
        match self {
            Self::F32(v) => SliceRef::F32(v),
            Self::F16(v) => SliceRef::F16(v),
            Self::BF16(v) => SliceRef::BF16(v),
            Self::I8(v, s) => SliceRef::I8(v, *s),
        }
    }
}

enum Weights {
    Raw(Native),
    Fused(Vec<Tensor>),
}

fn dtype(name: &str) -> Result<DType, String> {
    match name {
        "f32" => Ok(DType::F32),
        "f16" => Ok(DType::F16),
        "bf16" => Ok(DType::BF16),
        "i8" => Ok(DType::I8),
        _ => Err(format!("unknown dtype: {name}")),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len().is_multiple_of(2) {
        return Err("each option needs a value".into());
    }
    let allowed = [
        "--kind",
        "--dtype",
        "--input-dtype",
        "--k",
        "--n",
        "--kv-n",
        "--threads",
        "--sets",
        "--warmup",
        "--iters",
        "--samples",
    ];
    if args[1..]
        .chunks(2)
        .any(|v| !allowed.contains(&v[0].as_str()))
    {
        return Err("unknown option".into());
    }
    let value = |flag: &str, default: &str| {
        args[1..]
            .chunks(2)
            .find(|v| v[0] == flag)
            .map_or_else(|| default.to_string(), |v| v[1].clone())
    };
    let kind = value("--kind", "matvec");
    if !["matvec", "argmax", "qkv", "gate"].contains(&kind.as_str()) {
        return Err("unknown kind".into());
    }
    let weight_name = value("--dtype", "f16");
    let dt = dtype(&weight_name)?;
    let default_input = match (kind.as_str(), dt) {
        ("matvec", DType::F32 | DType::F16 | DType::I8) => "f32",
        (_, DType::F32) => "f32",
        (_, DType::F16) => "f16",
        _ => "bf16",
    };
    let input_name = value("--input-dtype", default_input);
    let input_dt = dtype(&input_name)?;
    if input_dt == DType::I8 {
        return Err("input dtype must be f32/f16/bf16".into());
    }
    let k = value("--k", "2048").parse::<usize>()?;
    let n = value("--n", "2048").parse::<usize>()?;
    let kv = value("--kv-n", "256").parse::<usize>()?;
    let threads = value("--threads", "8").parse::<usize>()?;
    let sets = value("--sets", "1").parse::<usize>()?;
    let warmup = value("--warmup", "100").parse::<usize>()?;
    let iters = value("--iters", "20").parse::<usize>()?;
    let samples = value("--samples", "7").parse::<usize>()?;
    if [k, n, kv, threads, sets, iters, samples].contains(&0) {
        return Err("sizes and counts must be positive".into());
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()?;
    set_allow_parameter_dtype_copies(false);
    let data = |len: usize, seed: usize| -> Vec<f32> {
        (0..len)
            .map(|i| ((i.wrapping_mul(17).wrapping_add(seed) % 257) as f32 - 128.) / 256.)
            .collect()
    };
    let tensor = |shape: Vec<usize>, values: Vec<f32>, dt| {
        let t = Tensor::from_array_no_grad(ndarray::ArrayD::from_shape_vec(shape, values).unwrap());
        t.cast_inplace(dt);
        t
    };
    let input_values = data(k, 11);
    let input = Native::new(input_values.clone(), input_dt);
    let a = tensor(vec![1, 1, k], input_values, input_dt);
    let sizes = match kind.as_str() {
        "qkv" => vec![n, kv, kv],
        "gate" => vec![n, n],
        _ => vec![n],
    };
    let elements = sizes
        .iter()
        .try_fold(0usize, |sum, &rows| {
            rows.checked_mul(k).and_then(|v| sum.checked_add(v))
        })
        .ok_or("weight size overflow")?;
    let bytes = elements
        .checked_mul(match dt {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::I8 => 1,
        })
        .ok_or("byte size overflow")?;
    let working_bytes = bytes.checked_mul(sets).ok_or("working set size overflow")?;
    let mut weights = Vec::with_capacity(sets);
    for set in 0..sets {
        let mut matrices = Vec::new();
        for (i, &rows) in sizes.iter().enumerate() {
            let values = data(rows * k, 23 + set * 419 + i * 37);
            if kind == "matvec" || kind == "argmax" {
                weights.push(Weights::Raw(Native::new(values, dt)));
            } else {
                matrices.push(tensor(vec![rows, k], values, dt));
            }
        }
        if !matrices.is_empty() {
            weights.push(Weights::Fused(matrices));
        }
    }
    let mut out = vec![0.; n];
    let mut ko = vec![0.; kv];
    let mut vo = ko.clone();
    let mut cursor = 0;
    let mut last_argmax = 0;
    no_grad(|| {
        let mut run = || {
            let w = &weights[cursor];
            cursor = (cursor + 1) % weights.len();
            match w {
                Weights::Raw(w) if kind == "argmax" => {
                    last_argmax =
                        matvec_argmax_rowmajor_parallel_mixed(input.slice(), w.slice(), n, k);
                    black_box(last_argmax);
                }
                Weights::Raw(w) => {
                    matvec_rowmajor_parallel_mixed(input.slice(), w.slice(), n, k, &mut out);
                    black_box(&out);
                }
                Weights::Fused(w) if kind == "qkv" => {
                    fused_qkv_decode_infer_into(
                        &a, &w[0], &w[1], &w[2], &mut out, &mut ko, &mut vo,
                    );
                    black_box((&out, &ko, &vo));
                }
                Weights::Fused(w) => {
                    fused_gate_up_silu_infer_into(&a, &w[0], &w[1], &mut out);
                    black_box(&out);
                }
            }
        };
        for _ in 0..warmup.max(sets) {
            run();
        }
        let mut timings = Vec::with_capacity(samples);
        for _ in 0..samples {
            let start = Instant::now();
            for _ in 0..iters {
                run();
            }
            timings.push(start.elapsed().as_secs_f64() * 1e6 / iters as f64);
        }
        let mut sorted = timings.clone();
        sorted.sort_by(f64::total_cmp);
        let median = if samples.is_multiple_of(2) {
            (sorted[samples / 2 - 1] + sorted[samples / 2]) / 2.
        } else {
            sorted[samples / 2]
        };
        let checksum: f64 = out.iter().chain(&ko).chain(&vo).map(|&v| v as f64).sum();
        println!(
            "{}",
            serde_json::json!({"kind":kind,"dtype":weight_name,"input_dtype":input_name,"k":k,"n":n,"kv_n":kv,"threads":threads,"sets":sets,"warmup":warmup.max(sets),"iters":iters,"samples_us":timings,"median_us":median,"weight_bytes_per_call":bytes,"working_weight_bytes":working_bytes,"effective_weight_gb_s":bytes as f64/median/1e3,"argmax":last_argmax,"checksum":checksum})
        );
    });
    Ok(())
}
