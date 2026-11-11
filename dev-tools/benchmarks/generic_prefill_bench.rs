//! Warmed, reproducible measurements of general matrix and attention shapes.
use lumen::autograd::{Tensor, TensorStorageViewMut, no_grad};
use lumen::cpu::CpuKernelContext;
use lumen::layers::{AttentionMask, AttentionOptions, scaled_dot_product_attention};
use lumen::ops::matmul::{batch_matmul, matmul, matmul_transposed_rhs};
use lumen::precision::{DType, set_allow_parameter_dtype_copies};
use ndarray::{ArrayD, IxDyn};
use serde_json::json;
use std::{hint::black_box, time::Instant};

#[derive(Default)]
struct Selection {
    op: Option<String>,
    dtype: Option<String>,
    case: Option<usize>,
}

impl Selection {
    fn includes(&self, op: &str, dtype: DType, case: usize) -> bool {
        self.op.as_deref().is_none_or(|v| v == op)
            && self
                .dtype
                .as_deref()
                .is_none_or(|v| v == format!("{dtype:?}"))
            && self.case.is_none_or(|v| v == case)
    }

    fn has_cases(&self) -> bool {
        [
            (
                "matmul",
                &[DType::F32, DType::F16, DType::BF16, DType::I8][..],
                2,
            ),
            (
                "matmul_transposed_rhs",
                &[DType::F32, DType::F16, DType::BF16, DType::I8][..],
                2,
            ),
            (
                "batch_matmul",
                &[DType::F32, DType::F16, DType::BF16][..],
                3,
            ),
            ("attention", &[DType::F32][..], 7),
        ]
        .into_iter()
        .any(|(op, dtypes, count)| {
            dtypes
                .iter()
                .any(|&dtype| (0..count).any(|case| self.includes(op, dtype, case)))
        })
    }
}

fn ordered<A: Clone>(a: ArrayD<A>, layout: usize) -> ArrayD<A> {
    if layout == 1 {
        a.reversed_axes()
            .as_standard_layout()
            .to_owned()
            .reversed_axes()
    } else {
        a
    }
}

// These tensors are freshly constructed and have no device/cache mirrors.
// The native visitor verifies physical strides outside the timed region.
fn storage_strides(t: &Tensor) -> Vec<isize> {
    t.with_native_storage_view_mut(|view| match view {
        TensorStorageViewMut::F32(v) => v.strides().to_vec(),
        TensorStorageViewMut::F16(v) => v.strides().to_vec(),
        TensorStorageViewMut::BF16(v) => v.strides().to_vec(),
        TensorStorageViewMut::I8(v, _) => v.strides().to_vec(),
    })
}

fn tensor(shape: &[usize], dtype: DType, layout: usize, seed: usize) -> Tensor {
    let a = ArrayD::from_shape_fn(IxDyn(shape), |index| {
        let i = index
            .slice()
            .iter()
            .zip(shape)
            .fold(0usize, |a, (&i, &n)| a * n + i);
        (((i.wrapping_mul(17) + seed) % 251) as f32 - 125.0) * 0.00371
    });
    let result = Tensor::from_array_no_grad(ArrayD::zeros(IxDyn(&[0])));
    match dtype {
        DType::F32 => result.set_array_f32_with_dtype(ordered(a, layout), dtype),
        DType::F16 => {
            result.set_array_f16_with_dtype(ordered(a.mapv(half::f16::from_f32), layout), dtype)
        }
        DType::BF16 => {
            result.set_array_bf16_with_dtype(ordered(a.mapv(half::bf16::from_f32), layout), dtype)
        }
        DType::I8 => {
            let max = a.iter().fold(0f32, |max, &v| max.max(v.abs()));
            let scale = if max > 0. {
                (max / 127.).max(f32::MIN_POSITIVE)
            } else {
                1.
            };
            let inverse = scale.recip();
            let values = a.mapv(|v| (v * inverse).round().clamp(-127., 127.) as i8);
            result.set_array_i8_with_dtype(ordered(values, layout), scale, dtype);
        }
    }
    let strides = storage_strides(&result);
    if layout == 1 && shape.iter().filter(|&&n| n > 1).count() > 1 {
        assert_eq!(
            strides[0], 1,
            "strided benchmark must preserve native column-major storage"
        );
    }
    result
}
use ndarray::Dimension;

fn measure(
    mut operation: impl FnMut() -> Tensor,
    warmup: usize,
    samples: usize,
    iters: usize,
) -> serde_json::Value {
    for _ in 0..warmup {
        for _ in 0..iters {
            black_box(operation());
        }
    }
    let mut run_samples = || {
        let mut timings = Vec::new();
        for _ in 0..samples {
            let start = Instant::now();
            for _ in 0..iters {
                black_box(operation());
            }
            timings.push(start.elapsed().as_secs_f64() * 1000.0 / iters as f64);
        }
        timings
    };
    #[cfg(feature = "cpu-profiling")]
    let (timings, profile) = {
        let (timings, profile) = lumen::cpu::profiling::capture(&mut run_samples);
        (timings, json!(profile))
    };
    #[cfg(not(feature = "cpu-profiling"))]
    let (timings, profile) = (run_samples(), serde_json::Value::Null);
    let output = operation();
    let data = output.detach().data();
    assert!(data.iter().all(|x| x.is_finite()));
    let mut sorted = timings.clone();
    sorted.sort_by(f64::total_cmp);
    let dump = std::env::var_os("LUMEN_BENCH_DUMP").is_some();
    json!({"samples_ms":timings,"median_ms":sorted[sorted.len()/2],"warmup":warmup,"iterations_per_sample":iters,"output_shape":output.shape_vec(),"output_dtype":format!("{:?}",output.dtype()),"output_sum_f64":data.iter().map(|&v|v as f64).sum::<f64>(),"output_values":if dump || data.len()<=8192{Some(data.iter().copied().collect::<Vec<_>>())}else{None},"profile":profile})
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() == 2 && args[1] == "--help" {
        println!(
            "generic_prefill_bench [--threads N] [--warmup N] [--samples N] [--iters N] \
             [--op matmul|matmul_transposed_rhs|batch_matmul|attention] [--dtype F32|F16|BF16|I8] [--case N] [--tasks N] \
             [--input-dtype F32|F16|BF16|I8] [--shape M,K,N] [--layout 0|1] \
             (input-dtype, shape and layout require a matrix-product --op; custom shape uses case 0)"
        );
        return Ok(());
    }
    let mut seen = std::collections::HashSet::new();
    for pair in args[1..].chunks(2) {
        if !matches!(
            pair[0].as_str(),
            "--threads"
                | "--warmup"
                | "--samples"
                | "--iters"
                | "--op"
                | "--dtype"
                | "--case"
                | "--tasks"
                | "--input-dtype"
                | "--shape"
                | "--layout"
        ) {
            return Err(format!("unknown argument: {}", pair[0]).into());
        }
        if !seen.insert(pair[0].as_str()) {
            return Err(format!("duplicate argument: {}", pair[0]).into());
        }
        if pair.len() != 2 {
            return Err(format!("{} needs a value", pair[0]).into());
        }
    }
    let arg = |key: &str, default: &str| -> Result<String, String> {
        args.iter()
            .position(|v| v == key)
            .map_or(Ok(default.into()), |i| {
                args.get(i + 1)
                    .cloned()
                    .ok_or_else(|| format!("{key} needs a value"))
            })
    };
    let threads = arg("--threads", "8")?.parse::<usize>()?;
    let warmup = arg("--warmup", "2")?.parse::<usize>()?;
    let samples = arg("--samples", "5")?.parse::<usize>()?;
    let iters = arg("--iters", "3")?.parse::<usize>()?;
    let tasks = seen
        .contains("--tasks")
        .then(|| {
            arg("--tasks", "").and_then(|s| {
                s.parse::<std::num::NonZeroUsize>()
                    .map_err(|e| e.to_string())
            })
        })
        .transpose()?;
    let context = tasks.map_or_else(CpuKernelContext::default, |limit| {
        CpuKernelContext::default().with_task_limit(limit)
    });
    let selection = Selection {
        op: seen.contains("--op").then(|| arg("--op", "")).transpose()?,
        dtype: seen
            .contains("--dtype")
            .then(|| arg("--dtype", ""))
            .transpose()?,
        case: seen
            .contains("--case")
            .then(|| arg("--case", "").and_then(|s| s.parse::<usize>().map_err(|e| e.to_string())))
            .transpose()?,
    };
    if !selection.has_cases() {
        return Err("selection matches no benchmark cases (see --help)".into());
    }
    let matrix_op = matches!(
        selection.op.as_deref(),
        Some("matmul" | "matmul_transposed_rhs")
    );
    if (seen.contains("--shape") || seen.contains("--layout")) && !matrix_op {
        return Err("--shape and --layout require --op matmul or matmul_transposed_rhs".into());
    }
    let custom_shape = if seen.contains("--shape") {
        let dims = arg("--shape", "")?
            .split(',')
            .map(str::parse::<usize>)
            .collect::<Result<Vec<_>, _>>()?;
        if dims.len() != 3 || dims.contains(&0) {
            return Err("--shape expects three positive dimensions M,K,N".into());
        }
        for (a, b) in [(dims[0], dims[1]), (dims[1], dims[2]), (dims[0], dims[2])] {
            a.checked_mul(b)
                .and_then(|n| n.checked_mul(std::mem::size_of::<f32>()))
                .ok_or("--shape storage size overflows")?;
        }
        if selection.case.is_some_and(|case| case != 0) {
            return Err("a custom --shape uses --case 0".into());
        }
        Some((dims[0], dims[1], dims[2]))
    } else {
        None
    };
    let layout_override = if seen.contains("--layout") {
        let layout = arg("--layout", "")?.parse::<usize>()?;
        if layout > 1 {
            return Err("--layout must be 0 (row-major) or 1 (strided)".into());
        }
        Some(layout)
    } else {
        None
    };
    let input_dtype = if seen.contains("--input-dtype") {
        if !matches!(
            selection.op.as_deref(),
            Some("matmul" | "matmul_transposed_rhs")
        ) {
            return Err("--input-dtype requires --op matmul or matmul_transposed_rhs".into());
        }
        Some(match arg("--input-dtype", "")?.as_str() {
            "F32" => DType::F32,
            "F16" => DType::F16,
            "BF16" => DType::BF16,
            "I8" => DType::I8,
            _ => return Err("unsupported --input-dtype (see --help)".into()),
        })
    } else {
        None
    };
    if [threads, samples, iters].contains(&0) {
        return Err("threads, samples and iterations must be positive".into());
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()?;
    set_allow_parameter_dtype_copies(false);
    context.scope(|| no_grad(|| {
        for op in ["matmul", "matmul_transposed_rhs"] {
        for dtype in [DType::F32, DType::F16, DType::BF16, DType::I8] {
            let shapes = custom_shape.map_or_else(|| vec![(32, 513, 767), (129, 1024, 1536)], |shape| vec![shape]);
            for (case, (m, k, n)) in shapes.into_iter().enumerate() {
                if !selection.includes(op, dtype, case) {
                    continue;
                }
                let layout = layout_override.unwrap_or(case % 2);
                let input_dtype = input_dtype.unwrap_or(dtype);
                let a = tensor(&[m, k], input_dtype, layout, 3);
                let b_shape = if op == "matmul" { [k, n] } else { [n, k] };
                let b = tensor(&b_shape, dtype, layout, 7);
                let input_strides=storage_strides(&a);
                let rhs_strides=storage_strides(&b);
                println!(
                    "{}",
                    json!({"op":op,"case":case,"threads":threads,"task_limit":tasks.map(|n|n.get()),"input_dtype":format!("{input_dtype:?}"),"dtype":format!("{dtype:?}"),"shape":[m,k,n],"layout":layout,"input_strides":input_strides,"rhs_strides":rhs_strides,"result":measure(|| if op == "matmul" { matmul(&a, &b) } else { matmul_transposed_rhs(&a, &b) },warmup,samples,iters)})
                );
            }
        }
        }
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            for (case, (b, h, m, k, n)) in [
                (1, 1, 513, 129, 257),
                (2, 3, 127, 65, 97),
                (1, 8, 193, 64, 129),
            ]
            .into_iter()
            .enumerate()
            {
                if !selection.includes("batch_matmul", dtype, case) {
                    continue;
                }
                let layout = case % 2;
                let a = tensor(&[b, h, m, k], dtype, layout, 3);
                let r = tensor(&[b, h, k, n], dtype, layout, 7);
                println!(
                    "{}",
                    json!({"op":"batch_matmul","case":case,"threads":threads,"task_limit":tasks.map(|n|n.get()),"dtype":format!("{dtype:?}"),"shape":[b,h,m,k,n],"layout":layout,"result":measure(||batch_matmul(&a,&r),warmup,samples,iters)})
                );
            }
        }
        for (case, (b, h, hkv, q, k, d, dv, mask_kind, causal, offset)) in [
            (1, 1, 1, 7, 9, 5, 3, 0, false, 0),
            (2, 6, 2, 129, 191, 65, 33, 1, true, 31),
            (1, 1, 1, 1024, 1024, 64, 80, 0, true, 0),
            (1, 12, 3, 193, 257, 64, 48, 2, false, 0),
            (4, 2, 1, 257, 385, 33, 17, 1, false, 0),
            (2, 6, 1, 1, 513, 65, 33, 0, true, 512),
            (1, 8, 2, 513, 769, 96, 64, 0, false, 0),
        ]
        .into_iter()
        .enumerate()
        {
            let dtype = DType::F32;
            if !selection.includes("attention", dtype, case) {
                continue;
            }
            let layout = case % 2;
            let query = tensor(&[b, h, q, d], dtype, layout, 3);
            let key = tensor(&[b, hkv, k, d], dtype, layout, 7);
            let value = tensor(&[b, hkv, k, dv], dtype, layout, 11);
            let keep = ArrayD::from_shape_fn(IxDyn(&[b, 1, 1, k]), |i| {
                i[3] % 7 != 2 && !(b > 1 && i[0] == b - 1)
            });
            let bias = tensor(&[1, 1, q, k], dtype, layout, 13);
            let options = AttentionOptions {
                scale: None,
                causal,
                query_offset: offset,
            };
            let operation = || {
                scaled_dot_product_attention(
                    &query,
                    &key,
                    &value,
                    match mask_kind {
                        1 => Some(AttentionMask::Keep(&keep)),
                        2 => Some(AttentionMask::Additive(&bias)),
                        _ => None,
                    },
                    options,
                )
            };
            println!(
                "{}",
                json!({"op":"attention","case":case,"threads":threads,"task_limit":tasks.map(|n|n.get()),"dtype":"F32","shape":[b,h,hkv,q,k,d,dv],"layout":layout,"mask_kind":mask_kind,"causal":causal,"offset":offset,"result":measure(operation,warmup,samples,iters)})
            );
        }
    }));
    Ok(())
}
