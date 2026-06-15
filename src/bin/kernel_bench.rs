use half::{bf16, f16};
use lumen::arch;
use lumen::autograd::{Tensor, no_grad};
use lumen::ops::fp_kernels::{
    active_bf16_bf16_backend_name, active_f16_f16_backend_name, active_float_backend_name,
    dot_bf16_bf16_arch, dot_f16_f16_arch, dot2_bf16_bf16_arch, dot2_f16_f16_arch,
    dot3_bf16_bf16_arch, dot3_f16_f16_arch,
};
use lumen::ops::fused::{fused_gate_up_silu_infer_into, fused_qkv_decode_infer_into};
use lumen::ops::int8_kernels::{
    I8ScaledRow, active_i8_i8_backend_name, active_int8_backend_name, dot2_i8_i8_arch,
    dot3_i8_i8_arch,
};
use lumen::ops::matmul::{
    SliceRef, dual_matvec_rowmajor_parallel, dual_matvec_rowmajor_parallel_mixed,
    dual_matvec_silu_mul_rowmajor_parallel, dual_matvec_silu_mul_rowmajor_parallel_mixed, matmul,
    matvec_argmax_rowmajor_parallel, matvec_argmax_rowmajor_parallel_mixed,
    matvec_rowmajor_parallel, matvec_rowmajor_parallel_mixed,
};
use lumen::optim::{Adam, Optimizer, SGD};
use lumen::precision::{DType, set_allow_parameter_dtype_copies, set_default_parameter_dtype};
use ndarray::Array;
use std::env;
use std::hint::black_box;
use std::time::{Duration, Instant};

#[derive(Clone, Copy)]
struct Args {
    iters: usize,
    samples: usize,
    hidden: usize,
    inter: usize,
    vocab: usize,
}

fn parse_args() -> Result<Args, String> {
    let argv: Vec<String> = env::args().collect();
    let mut iters = 200usize;
    let mut samples = 3usize;
    let mut hidden = 2048usize;
    let mut inter = 5632usize;
    let mut vocab = 32000usize;

    let mut i = 1usize;
    while i < argv.len() {
        match argv[i].as_str() {
            "--iters" => {
                i += 1;
                iters = argv
                    .get(i)
                    .ok_or("--iters 缺少数字")?
                    .parse::<usize>()
                    .map_err(|_| "--iters 需要 usize")?;
            }
            "--samples" => {
                i += 1;
                samples = argv
                    .get(i)
                    .ok_or("--samples 缺少数字")?
                    .parse::<usize>()
                    .map_err(|_| "--samples 需要 usize")?;
                if samples == 0 {
                    return Err("--samples 必须 >= 1".to_string());
                }
            }
            "--hidden" => {
                i += 1;
                hidden = argv
                    .get(i)
                    .ok_or("--hidden 缺少数字")?
                    .parse::<usize>()
                    .map_err(|_| "--hidden 需要 usize")?;
            }
            "--inter" => {
                i += 1;
                inter = argv
                    .get(i)
                    .ok_or("--inter 缺少数字")?
                    .parse::<usize>()
                    .map_err(|_| "--inter 需要 usize")?;
            }
            "--vocab" => {
                i += 1;
                vocab = argv
                    .get(i)
                    .ok_or("--vocab 缺少数字")?
                    .parse::<usize>()
                    .map_err(|_| "--vocab 需要 usize")?;
            }
            "-h" | "--help" => {
                println!(
                    "Usage: cargo run --release --bin kernel_bench -- [--iters N] [--samples N] [--hidden N] [--inter N] [--vocab N]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("未知参数: {other}")),
        }
        i += 1;
    }

    Ok(Args {
        iters,
        samples,
        hidden,
        inter,
        vocab,
    })
}

fn make_f32(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let v = ((i * 1315423911usize) ^ (i.rotate_left(5))) & 1023;
            (v as f32) / 1024.0 - 0.5
        })
        .collect()
}

fn make_bf16(src: &[f32]) -> Vec<bf16> {
    src.iter().map(|&v| bf16::from_f32(v)).collect()
}

fn make_f16(src: &[f32]) -> Vec<f16> {
    src.iter().map(|&v| f16::from_f32(v)).collect()
}

fn make_i8(len: usize, seed: usize) -> Vec<i8> {
    (0..len)
        .map(|i| {
            let v = ((i.wrapping_mul(1103515245usize) ^ seed).rotate_left(7) & 255) as i32 - 128;
            v.clamp(-127, 127) as i8
        })
        .collect()
}

fn elapsed_per_iter(total: Duration, iters: usize) -> Duration {
    Duration::from_secs_f64(total.as_secs_f64() / iters as f64)
}

fn bench(iters: usize, mut f: impl FnMut()) -> Duration {
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    start.elapsed()
}

fn bench_sampled(samples: usize, iters: usize, mut f: impl FnMut()) -> Duration {
    let mut runs = Vec::with_capacity(samples);
    for _ in 0..samples {
        runs.push(bench(iters, &mut f));
    }
    runs.sort_unstable();
    runs[samples / 2]
}

fn max_abs_diff(lhs: &[f32], rhs: &[f32]) -> f32 {
    assert_eq!(lhs.len(), rhs.len(), "benchmark vector length mismatch");
    lhs.iter()
        .zip(rhs.iter())
        .map(|(&a, &b)| {
            if a.is_finite() && b.is_finite() {
                (a - b).abs()
            } else {
                f32::INFINITY
            }
        })
        .fold(0.0f32, f32::max)
}

fn print_case(name: &str, f32_total: Duration, mixed_total: Duration, diff: f32, iters: usize) {
    let f32_us = elapsed_per_iter(f32_total, iters).as_secs_f64() * 1e6;
    let mixed_us = elapsed_per_iter(mixed_total, iters).as_secs_f64() * 1e6;
    let ratio = mixed_us / f32_us;
    println!(
        "{name:<18} f32={f32_us:>9.2} us  mixed={mixed_us:>9.2} us  ratio={ratio:>5.2}x  max_abs_diff={diff:.6}"
    );
}

fn print_pair(name: &str, lhs_total: Duration, rhs_total: Duration, iters: usize) {
    let lhs_us = elapsed_per_iter(lhs_total, iters).as_secs_f64() * 1e6;
    let rhs_us = elapsed_per_iter(rhs_total, iters).as_secs_f64() * 1e6;
    let speedup = lhs_us / rhs_us;
    println!(
        "{name:<18} no_copy={lhs_us:>9.2} us  cached={rhs_us:>9.2} us  speedup={speedup:>5.2}x"
    );
}

fn print_pair_named(
    name: &str,
    lhs_label: &str,
    rhs_label: &str,
    lhs_total: Duration,
    rhs_total: Duration,
    iters: usize,
) {
    let lhs_us = elapsed_per_iter(lhs_total, iters).as_secs_f64() * 1e6;
    let rhs_us = elapsed_per_iter(rhs_total, iters).as_secs_f64() * 1e6;
    let speedup = lhs_us / rhs_us;
    println!(
        "{name:<18} {lhs_label}={lhs_us:>9.2} us  {rhs_label}={rhs_us:>9.2} us  speedup={speedup:>5.2}x"
    );
}

fn print_single(name: &str, total: Duration, iters: usize) {
    let us = elapsed_per_iter(total, iters).as_secs_f64() * 1e6;
    println!("{name:<18} time={us:>9.2} us");
}

fn print_arch_case(
    name: &str,
    scalar_total: Duration,
    arch_total: Duration,
    diff: f32,
    iters: usize,
) {
    let scalar_us = elapsed_per_iter(scalar_total, iters).as_secs_f64() * 1e6;
    let arch_us = elapsed_per_iter(arch_total, iters).as_secs_f64() * 1e6;
    let speedup = scalar_us / arch_us;
    println!(
        "{name:<18} scalar={scalar_us:>9.2} us  arch={arch_us:>9.2} us  speedup={speedup:>5.2}x  max_abs_diff={diff:.6}"
    );
}

fn make_tensor(shape: &[usize], data: Vec<f32>, dtype: DType) -> Tensor {
    let tensor = Tensor::from_array_no_grad(
        Array::from_shape_vec(shape.to_vec(), data)
            .unwrap()
            .into_dyn(),
    );
    tensor.cast_inplace(dtype);
    tensor
}

fn make_parameter(shape: &[usize], data: Vec<f32>, dtype: DType) -> Tensor {
    let tensor = Tensor::parameter(
        Array::from_shape_vec(shape.to_vec(), data)
            .unwrap()
            .into_dyn(),
    );
    tensor.cast_inplace(dtype);
    tensor
}

fn make_quantized_f32_tensor(shape: &[usize], data: Vec<f32>, quantized_dtype: DType) -> Tensor {
    let quantized = make_tensor(shape, data, quantized_dtype);
    let quantized_f32 = quantized.data_ref().iter().copied().collect::<Vec<_>>();
    make_tensor(shape, quantized_f32, DType::F32)
}

fn grad_array(shape: &[usize], data: &[f32]) -> ndarray::ArrayD<f32> {
    Array::from_shape_vec(shape.to_vec(), data.to_vec())
        .expect("grad shape mismatch")
        .into_dyn()
}

fn scalar_dot_f16_f16(x: &[f16], row: &[f16]) -> f32 {
    x.iter()
        .zip(row.iter())
        .map(|(&xv, &rv)| xv.to_f32() * rv.to_f32())
        .sum()
}

fn scalar_dot2_f16_f16(x: &[f16], row0: &[f16], row1: &[f16]) -> (f32, f32) {
    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    for ((&xv, &r0), &r1) in x.iter().zip(row0.iter()).zip(row1.iter()) {
        let xv = xv.to_f32();
        sum0 += xv * r0.to_f32();
        sum1 += xv * r1.to_f32();
    }
    (sum0, sum1)
}

fn scalar_dot3_f16_f16(x: &[f16], row0: &[f16], row1: &[f16], row2: &[f16]) -> (f32, f32, f32) {
    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    let mut sum2 = 0.0f32;
    for (((&xv, &r0), &r1), &r2) in x.iter().zip(row0.iter()).zip(row1.iter()).zip(row2.iter()) {
        let xv = xv.to_f32();
        sum0 += xv * r0.to_f32();
        sum1 += xv * r1.to_f32();
        sum2 += xv * r2.to_f32();
    }
    (sum0, sum1, sum2)
}

fn scalar_dot_bf16_bf16(x: &[bf16], row: &[bf16]) -> f32 {
    x.iter()
        .zip(row.iter())
        .map(|(&xv, &rv)| xv.to_f32() * rv.to_f32())
        .sum()
}

fn scalar_dot2_bf16_bf16(x: &[bf16], row0: &[bf16], row1: &[bf16]) -> (f32, f32) {
    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    for ((&xv, &r0), &r1) in x.iter().zip(row0.iter()).zip(row1.iter()) {
        let xv = xv.to_f32();
        sum0 += xv * r0.to_f32();
        sum1 += xv * r1.to_f32();
    }
    (sum0, sum1)
}

fn scalar_dot3_bf16_bf16(
    x: &[bf16],
    row0: &[bf16],
    row1: &[bf16],
    row2: &[bf16],
) -> (f32, f32, f32) {
    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    let mut sum2 = 0.0f32;
    for (((&xv, &r0), &r1), &r2) in x.iter().zip(row0.iter()).zip(row1.iter()).zip(row2.iter()) {
        let xv = xv.to_f32();
        sum0 += xv * r0.to_f32();
        sum1 += xv * r1.to_f32();
        sum2 += xv * r2.to_f32();
    }
    (sum0, sum1, sum2)
}

fn scalar_dot2_i8_i8(
    x: &[i8],
    x_scale: f32,
    row0: &[i8],
    scale0: f32,
    row1: &[i8],
    scale1: f32,
) -> (f32, f32) {
    let mut sum0 = 0i32;
    let mut sum1 = 0i32;
    for ((&xv, &r0), &r1) in x.iter().zip(row0.iter()).zip(row1.iter()) {
        let xv = xv as i32;
        sum0 += r0 as i32 * xv;
        sum1 += r1 as i32 * xv;
    }
    (
        sum0 as f32 * x_scale * scale0,
        sum1 as f32 * x_scale * scale1,
    )
}

fn scalar_dot3_i8_i8(x: &[i8], x_scale: f32, rows: [I8ScaledRow<'_>; 3]) -> (f32, f32, f32) {
    let [row0, row1, row2] = rows;
    let mut sum0 = 0i32;
    let mut sum1 = 0i32;
    let mut sum2 = 0i32;
    for (((&xv, &r0), &r1), &r2) in x
        .iter()
        .zip(row0.values.iter())
        .zip(row1.values.iter())
        .zip(row2.values.iter())
    {
        let xv = xv as i32;
        sum0 += r0 as i32 * xv;
        sum1 += r1 as i32 * xv;
        sum2 += r2 as i32 * xv;
    }
    (
        sum0 as f32 * x_scale * row0.scale,
        sum1 as f32 * x_scale * row1.scale,
        sum2 as f32 * x_scale * row2.scale,
    )
}

fn bench_sgd_update(samples: usize, iters: usize, dtype: DType, len: usize) -> Duration {
    let shape = [len];
    let param = make_parameter(&shape, make_f32(len), dtype);
    let grad = make_f32(len)
        .into_iter()
        .map(|v| v * 0.001)
        .collect::<Vec<_>>();
    let grad_template = grad_array(&shape, &grad);
    let mut opt = SGD::new_with_dtype(vec![param.clone()], 0.0001, DType::F32);

    bench_sampled(samples, iters, || {
        param.zero_grad();
        param.add_grad(grad_template.clone());
        opt.step();
        black_box(param.shape_vec());
    })
}

fn bench_sgd_momentum_update(samples: usize, iters: usize, dtype: DType, len: usize) -> Duration {
    let shape = [len];
    let param = make_parameter(&shape, make_f32(len), dtype);
    let grad = make_f32(len)
        .into_iter()
        .map(|v| v * 0.001)
        .collect::<Vec<_>>();
    let grad_template = grad_array(&shape, &grad);
    let mut opt = SGD::new_with_dtype(vec![param.clone()], 0.0001, DType::F32).with_momentum(0.9);

    bench_sampled(samples, iters, || {
        param.zero_grad();
        param.add_grad(grad_template.clone());
        opt.step();
        black_box(param.shape_vec());
    })
}

fn bench_adam_update(samples: usize, iters: usize, dtype: DType, len: usize) -> Duration {
    let shape = [len];
    let param = make_parameter(&shape, make_f32(len), dtype);
    let grad = make_f32(len)
        .into_iter()
        .map(|v| v * 0.001)
        .collect::<Vec<_>>();
    let grad_template = grad_array(&shape, &grad);
    let mut opt = Adam::new_with_dtype(vec![param.clone()], 0.0001, DType::F32);

    bench_sampled(samples, iters, || {
        param.zero_grad();
        param.add_grad(grad_template.clone());
        opt.step();
        black_box(param.shape_vec());
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args().map_err(|e| format!("参数错误: {e}"))?;
    println!(
        "kernel bench: iters={} samples={} hidden={} inter={} vocab={}",
        args.iters, args.samples, args.hidden, args.inter, args.vocab
    );
    println!(
        "backend: float={} bf16_bf16={} f16_f16={} int8={} i8_i8={} avx512fp16={}",
        active_float_backend_name(),
        active_bf16_bf16_backend_name(),
        active_f16_f16_backend_name(),
        active_int8_backend_name(),
        active_i8_i8_backend_name(),
        if arch::x86_avx512_fp16_kernel_runtime_available() {
            "nightly-enabled"
        } else {
            "unavailable-or-stable-build"
        }
    );

    let x = make_f32(args.hidden);
    let x_bf16 = make_bf16(&x);
    let x_f16 = make_f16(&x);
    let x_i8 = make_i8(args.hidden, 0x1234);
    let row0_i8 = make_i8(args.hidden, 0x2345);
    let row1_i8 = make_i8(args.hidden, 0x3456);
    let row2_i8 = make_i8(args.hidden, 0x4567);
    let w_f32 = make_f32(args.inter * args.hidden);
    let w_bf16 = make_bf16(&w_f32);
    let w_f16 = make_f16(&w_f32);

    let w2_f32 = make_f32(args.inter * args.hidden);
    let w2_bf16 = make_bf16(&w2_f32);
    let w2_f16 = make_f16(&w2_f32);

    let vocab_w_f32 = make_f32(args.vocab * args.hidden);
    let vocab_w_bf16 = make_bf16(&vocab_w_f32);
    let vocab_w_f16 = make_f16(&vocab_w_f32);

    let x_scale = 0.03125f32;
    let scale0 = 0.015625f32;
    let scale1 = 0.0078125f32;
    let scale2 = 0.00390625f32;

    let dot_row0_f32 = make_f32(args.hidden);
    let dot_row1_f32 = dot_row0_f32
        .iter()
        .enumerate()
        .map(|(i, &v)| v.mul_add(-0.5, ((i & 7) as f32 - 3.0) * 0.01))
        .collect::<Vec<_>>();
    let dot_row2_f32 = dot_row0_f32
        .iter()
        .enumerate()
        .map(|(i, &v)| v.mul_add(0.25, ((i & 15) as f32 - 7.0) * 0.005))
        .collect::<Vec<_>>();
    let dot_row0_f16 = make_f16(&dot_row0_f32);
    let dot_row1_f16 = make_f16(&dot_row1_f32);
    let dot_row2_f16 = make_f16(&dot_row2_f32);
    let dot_row0_bf16 = make_bf16(&dot_row0_f32);
    let dot_row1_bf16 = make_bf16(&dot_row1_f32);
    let dot_row2_bf16 = make_bf16(&dot_row2_f32);

    let scalar_dot_bf16 = bench_sampled(args.samples, args.iters, || {
        black_box(scalar_dot_bf16_bf16(
            black_box(&x_bf16),
            black_box(&dot_row0_bf16),
        ));
    });
    if dot_bf16_bf16_arch(&x_bf16, &dot_row0_bf16).is_some() {
        let arch_dot_bf16 = bench_sampled(args.samples, args.iters, || {
            let out = dot_bf16_bf16_arch(black_box(&x_bf16), black_box(&dot_row0_bf16))
                .expect("bf16 dot arch path should stay available during bench");
            black_box(out);
        });
        let scalar = scalar_dot_bf16_bf16(&x_bf16, &dot_row0_bf16);
        let arch = dot_bf16_bf16_arch(&x_bf16, &dot_row0_bf16).expect("bf16 dot arch");
        print_arch_case(
            "dot_bf16_bf16",
            scalar_dot_bf16,
            arch_dot_bf16,
            (scalar - arch).abs(),
            args.iters,
        );
    } else {
        print_single("dot_bf16_bf16_scalar", scalar_dot_bf16, args.iters);
        println!("dot_bf16_bf16    arch=skipped (runtime feature unavailable)");
    }

    let scalar_dot2_bf16 = bench_sampled(args.samples, args.iters, || {
        black_box(scalar_dot2_bf16_bf16(
            black_box(&x_bf16),
            black_box(&dot_row0_bf16),
            black_box(&dot_row1_bf16),
        ));
    });
    if dot2_bf16_bf16_arch(&x_bf16, &dot_row0_bf16, &dot_row1_bf16).is_some() {
        let arch_dot2_bf16 = bench_sampled(args.samples, args.iters, || {
            let out = dot2_bf16_bf16_arch(
                black_box(&x_bf16),
                black_box(&dot_row0_bf16),
                black_box(&dot_row1_bf16),
            )
            .expect("bf16 dot2 arch path should stay available during bench");
            black_box(out);
        });
        let scalar = scalar_dot2_bf16_bf16(&x_bf16, &dot_row0_bf16, &dot_row1_bf16);
        let arch =
            dot2_bf16_bf16_arch(&x_bf16, &dot_row0_bf16, &dot_row1_bf16).expect("bf16 dot2 arch");
        print_arch_case(
            "dot2_bf16_bf16",
            scalar_dot2_bf16,
            arch_dot2_bf16,
            (scalar.0 - arch.0).abs().max((scalar.1 - arch.1).abs()),
            args.iters,
        );
    } else {
        print_single("dot2_bf16_bf16_scalar", scalar_dot2_bf16, args.iters);
        println!("dot2_bf16_bf16   arch=skipped (runtime feature unavailable)");
    }

    let scalar_dot3_bf16 = bench_sampled(args.samples, args.iters, || {
        black_box(scalar_dot3_bf16_bf16(
            black_box(&x_bf16),
            black_box(&dot_row0_bf16),
            black_box(&dot_row1_bf16),
            black_box(&dot_row2_bf16),
        ));
    });
    if dot3_bf16_bf16_arch(&x_bf16, &dot_row0_bf16, &dot_row1_bf16, &dot_row2_bf16).is_some() {
        let arch_dot3_bf16 = bench_sampled(args.samples, args.iters, || {
            let out = dot3_bf16_bf16_arch(
                black_box(&x_bf16),
                black_box(&dot_row0_bf16),
                black_box(&dot_row1_bf16),
                black_box(&dot_row2_bf16),
            )
            .expect("bf16 dot3 arch path should stay available during bench");
            black_box(out);
        });
        let scalar = scalar_dot3_bf16_bf16(&x_bf16, &dot_row0_bf16, &dot_row1_bf16, &dot_row2_bf16);
        let arch = dot3_bf16_bf16_arch(&x_bf16, &dot_row0_bf16, &dot_row1_bf16, &dot_row2_bf16)
            .expect("bf16 dot3 arch");
        print_arch_case(
            "dot3_bf16_bf16",
            scalar_dot3_bf16,
            arch_dot3_bf16,
            (scalar.0 - arch.0)
                .abs()
                .max((scalar.1 - arch.1).abs())
                .max((scalar.2 - arch.2).abs()),
            args.iters,
        );
    } else {
        print_single("dot3_bf16_bf16_scalar", scalar_dot3_bf16, args.iters);
        println!("dot3_bf16_bf16   arch=skipped (runtime feature unavailable)");
    }

    let scalar_dot_f16 = bench_sampled(args.samples, args.iters, || {
        black_box(scalar_dot_f16_f16(
            black_box(&x_f16),
            black_box(&dot_row0_f16),
        ));
    });
    if dot_f16_f16_arch(&x_f16, &dot_row0_f16).is_some() {
        let arch_dot_f16 = bench_sampled(args.samples, args.iters, || {
            let out = dot_f16_f16_arch(black_box(&x_f16), black_box(&dot_row0_f16))
                .expect("f16 dot arch path should stay available during bench");
            black_box(out);
        });
        let scalar = scalar_dot_f16_f16(&x_f16, &dot_row0_f16);
        let arch = dot_f16_f16_arch(&x_f16, &dot_row0_f16).expect("f16 dot arch");
        print_arch_case(
            "dot_f16_f16",
            scalar_dot_f16,
            arch_dot_f16,
            (scalar - arch).abs(),
            args.iters,
        );
    } else {
        print_single("dot_f16_f16_scalar", scalar_dot_f16, args.iters);
        println!("dot_f16_f16      arch=skipped (runtime feature unavailable)");
    }

    let scalar_dot2_f16 = bench_sampled(args.samples, args.iters, || {
        black_box(scalar_dot2_f16_f16(
            black_box(&x_f16),
            black_box(&dot_row0_f16),
            black_box(&dot_row1_f16),
        ));
    });
    if dot2_f16_f16_arch(&x_f16, &dot_row0_f16, &dot_row1_f16).is_some() {
        let arch_dot2_f16 = bench_sampled(args.samples, args.iters, || {
            let out = dot2_f16_f16_arch(
                black_box(&x_f16),
                black_box(&dot_row0_f16),
                black_box(&dot_row1_f16),
            )
            .expect("f16 dot2 arch path should stay available during bench");
            black_box(out);
        });
        let scalar = scalar_dot2_f16_f16(&x_f16, &dot_row0_f16, &dot_row1_f16);
        let arch = dot2_f16_f16_arch(&x_f16, &dot_row0_f16, &dot_row1_f16).expect("f16 dot2 arch");
        print_arch_case(
            "dot2_f16_f16",
            scalar_dot2_f16,
            arch_dot2_f16,
            (scalar.0 - arch.0).abs().max((scalar.1 - arch.1).abs()),
            args.iters,
        );
    } else {
        print_single("dot2_f16_f16_scalar", scalar_dot2_f16, args.iters);
        println!("dot2_f16_f16     arch=skipped (runtime feature unavailable)");
    }

    let scalar_dot3_f16 = bench_sampled(args.samples, args.iters, || {
        black_box(scalar_dot3_f16_f16(
            black_box(&x_f16),
            black_box(&dot_row0_f16),
            black_box(&dot_row1_f16),
            black_box(&dot_row2_f16),
        ));
    });
    if dot3_f16_f16_arch(&x_f16, &dot_row0_f16, &dot_row1_f16, &dot_row2_f16).is_some() {
        let arch_dot3_f16 = bench_sampled(args.samples, args.iters, || {
            let out = dot3_f16_f16_arch(
                black_box(&x_f16),
                black_box(&dot_row0_f16),
                black_box(&dot_row1_f16),
                black_box(&dot_row2_f16),
            )
            .expect("f16 dot3 arch path should stay available during bench");
            black_box(out);
        });
        let scalar = scalar_dot3_f16_f16(&x_f16, &dot_row0_f16, &dot_row1_f16, &dot_row2_f16);
        let arch = dot3_f16_f16_arch(&x_f16, &dot_row0_f16, &dot_row1_f16, &dot_row2_f16)
            .expect("f16 dot3 arch");
        print_arch_case(
            "dot3_f16_f16",
            scalar_dot3_f16,
            arch_dot3_f16,
            (scalar.0 - arch.0)
                .abs()
                .max((scalar.1 - arch.1).abs())
                .max((scalar.2 - arch.2).abs()),
            args.iters,
        );
    } else {
        print_single("dot3_f16_f16_scalar", scalar_dot3_f16, args.iters);
        println!("dot3_f16_f16     arch=skipped (runtime feature unavailable)");
    }

    let scalar_dot2 = bench_sampled(args.samples, args.iters, || {
        black_box(scalar_dot2_i8_i8(
            &x_i8, x_scale, &row0_i8, scale0, &row1_i8, scale1,
        ));
    });
    if dot2_i8_i8_arch(&x_i8, x_scale, &row0_i8, scale0, &row1_i8, scale1).is_some() {
        let arch_dot2 = bench_sampled(args.samples, args.iters, || {
            let out = dot2_i8_i8_arch(
                black_box(&x_i8),
                x_scale,
                black_box(&row0_i8),
                scale0,
                black_box(&row1_i8),
                scale1,
            )
            .expect("i8 dot2 arch path should stay available during bench");
            black_box(out);
        });
        print_pair_named(
            "dot2_i8_i8",
            "scalar",
            "arch",
            scalar_dot2,
            arch_dot2,
            args.iters,
        );
    } else {
        print_single("dot2_i8_i8_scalar", scalar_dot2, args.iters);
        println!("dot2_i8_i8        arch=skipped (runtime feature unavailable)");
    }

    let scalar_dot3 = bench_sampled(args.samples, args.iters, || {
        black_box(scalar_dot3_i8_i8(
            &x_i8,
            x_scale,
            [
                I8ScaledRow {
                    values: &row0_i8,
                    scale: scale0,
                },
                I8ScaledRow {
                    values: &row1_i8,
                    scale: scale1,
                },
                I8ScaledRow {
                    values: &row2_i8,
                    scale: scale2,
                },
            ],
        ));
    });
    if dot3_i8_i8_arch(
        &x_i8,
        x_scale,
        [
            I8ScaledRow {
                values: &row0_i8,
                scale: scale0,
            },
            I8ScaledRow {
                values: &row1_i8,
                scale: scale1,
            },
            I8ScaledRow {
                values: &row2_i8,
                scale: scale2,
            },
        ],
    )
    .is_some()
    {
        let arch_dot3 = bench_sampled(args.samples, args.iters, || {
            let out = dot3_i8_i8_arch(
                black_box(&x_i8),
                x_scale,
                [
                    I8ScaledRow {
                        values: black_box(&row0_i8),
                        scale: scale0,
                    },
                    I8ScaledRow {
                        values: black_box(&row1_i8),
                        scale: scale1,
                    },
                    I8ScaledRow {
                        values: black_box(&row2_i8),
                        scale: scale2,
                    },
                ],
            )
            .expect("i8 dot3 arch path should stay available during bench");
            black_box(out);
        });
        print_pair_named(
            "dot3_i8_i8",
            "scalar",
            "arch",
            scalar_dot3,
            arch_dot3,
            args.iters,
        );
    } else {
        print_single("dot3_i8_i8_scalar", scalar_dot3, args.iters);
        println!("dot3_i8_i8        arch=skipped (runtime feature unavailable)");
    }

    let mut out_f32 = vec![0.0f32; args.inter];
    let mut out_mixed = vec![0.0f32; args.inter];
    let mut out0_f32 = vec![0.0f32; args.inter];
    let mut out1_f32 = vec![0.0f32; args.inter];
    let mut out0_mixed = vec![0.0f32; args.inter];
    let mut out1_mixed = vec![0.0f32; args.inter];

    let warmup = args.iters.clamp(4, 16);
    for _ in 0..warmup {
        matvec_rowmajor_parallel(&x, &w_f32, args.inter, args.hidden, &mut out_f32);
        matvec_rowmajor_parallel_mixed(
            SliceRef::F32(&x),
            SliceRef::BF16(&w_bf16),
            args.inter,
            args.hidden,
            &mut out_mixed,
        );
    }

    let matvec_f32 = bench_sampled(args.samples, args.iters, || {
        matvec_rowmajor_parallel(
            black_box(&x),
            black_box(&w_f32),
            args.inter,
            args.hidden,
            black_box(&mut out_f32),
        );
        black_box(out_f32[0]);
    });
    let matvec_mixed = bench_sampled(args.samples, args.iters, || {
        matvec_rowmajor_parallel_mixed(
            SliceRef::F32(black_box(&x)),
            SliceRef::BF16(black_box(&w_bf16)),
            args.inter,
            args.hidden,
            black_box(&mut out_mixed),
        );
        black_box(out_mixed[0]);
    });
    matvec_rowmajor_parallel(&x, &w_f32, args.inter, args.hidden, &mut out_f32);
    matvec_rowmajor_parallel_mixed(
        SliceRef::F32(&x),
        SliceRef::BF16(&w_bf16),
        args.inter,
        args.hidden,
        &mut out_mixed,
    );
    print_case(
        "matvec",
        matvec_f32,
        matvec_mixed,
        max_abs_diff(&out_f32, &out_mixed),
        args.iters,
    );
    let matvec_bf16 = bench_sampled(args.samples, args.iters, || {
        matvec_rowmajor_parallel_mixed(
            SliceRef::BF16(black_box(&x_bf16)),
            SliceRef::BF16(black_box(&w_bf16)),
            args.inter,
            args.hidden,
            black_box(&mut out_mixed),
        );
        black_box(out_mixed[0]);
    });
    matvec_rowmajor_parallel_mixed(
        SliceRef::BF16(&x_bf16),
        SliceRef::BF16(&w_bf16),
        args.inter,
        args.hidden,
        &mut out_mixed,
    );
    print_case(
        "matvec_bf16io",
        matvec_f32,
        matvec_bf16,
        max_abs_diff(&out_f32, &out_mixed),
        args.iters,
    );
    let matvec_f16 = bench_sampled(args.samples, args.iters, || {
        matvec_rowmajor_parallel_mixed(
            SliceRef::F16(black_box(&x_f16)),
            SliceRef::F16(black_box(&w_f16)),
            args.inter,
            args.hidden,
            black_box(&mut out_mixed),
        );
        black_box(out_mixed[0]);
    });
    matvec_rowmajor_parallel_mixed(
        SliceRef::F16(&x_f16),
        SliceRef::F16(&w_f16),
        args.inter,
        args.hidden,
        &mut out_mixed,
    );
    print_case(
        "matvec_f16io",
        matvec_f32,
        matvec_f16,
        max_abs_diff(&out_f32, &out_mixed),
        args.iters,
    );

    let dual_f32 = bench_sampled(args.samples, args.iters, || {
        dual_matvec_rowmajor_parallel(
            black_box(&x),
            black_box(&w_f32),
            black_box(&w2_f32),
            args.inter,
            args.hidden,
            black_box(&mut out0_f32),
            black_box(&mut out1_f32),
        );
        black_box(out0_f32[0] + out1_f32[0]);
    });
    let dual_mixed = bench_sampled(args.samples, args.iters, || {
        dual_matvec_rowmajor_parallel_mixed(
            SliceRef::F32(black_box(&x)),
            SliceRef::BF16(black_box(&w_bf16)),
            SliceRef::BF16(black_box(&w2_bf16)),
            args.inter,
            args.hidden,
            black_box(&mut out0_mixed),
            black_box(&mut out1_mixed),
        );
        black_box(out0_mixed[0] + out1_mixed[0]);
    });
    dual_matvec_rowmajor_parallel(
        &x,
        &w_f32,
        &w2_f32,
        args.inter,
        args.hidden,
        &mut out0_f32,
        &mut out1_f32,
    );
    dual_matvec_rowmajor_parallel_mixed(
        SliceRef::F32(&x),
        SliceRef::BF16(&w_bf16),
        SliceRef::BF16(&w2_bf16),
        args.inter,
        args.hidden,
        &mut out0_mixed,
        &mut out1_mixed,
    );
    print_case(
        "dual_matvec",
        dual_f32,
        dual_mixed,
        max_abs_diff(&out0_f32, &out0_mixed).max(max_abs_diff(&out1_f32, &out1_mixed)),
        args.iters,
    );
    let dual_bf16 = bench_sampled(args.samples, args.iters, || {
        dual_matvec_rowmajor_parallel_mixed(
            SliceRef::BF16(black_box(&x_bf16)),
            SliceRef::BF16(black_box(&w_bf16)),
            SliceRef::BF16(black_box(&w2_bf16)),
            args.inter,
            args.hidden,
            black_box(&mut out0_mixed),
            black_box(&mut out1_mixed),
        );
        black_box(out0_mixed[0] + out1_mixed[0]);
    });
    dual_matvec_rowmajor_parallel_mixed(
        SliceRef::BF16(&x_bf16),
        SliceRef::BF16(&w_bf16),
        SliceRef::BF16(&w2_bf16),
        args.inter,
        args.hidden,
        &mut out0_mixed,
        &mut out1_mixed,
    );
    print_case(
        "dual_bf16io",
        dual_f32,
        dual_bf16,
        max_abs_diff(&out0_f32, &out0_mixed).max(max_abs_diff(&out1_f32, &out1_mixed)),
        args.iters,
    );
    let dual_f16 = bench_sampled(args.samples, args.iters, || {
        dual_matvec_rowmajor_parallel_mixed(
            SliceRef::F16(black_box(&x_f16)),
            SliceRef::F16(black_box(&w_f16)),
            SliceRef::F16(black_box(&w2_f16)),
            args.inter,
            args.hidden,
            black_box(&mut out0_mixed),
            black_box(&mut out1_mixed),
        );
        black_box(out0_mixed[0] + out1_mixed[0]);
    });
    dual_matvec_rowmajor_parallel_mixed(
        SliceRef::F16(&x_f16),
        SliceRef::F16(&w_f16),
        SliceRef::F16(&w2_f16),
        args.inter,
        args.hidden,
        &mut out0_mixed,
        &mut out1_mixed,
    );
    print_case(
        "dual_f16io",
        dual_f32,
        dual_f16,
        max_abs_diff(&out0_f32, &out0_mixed).max(max_abs_diff(&out1_f32, &out1_mixed)),
        args.iters,
    );

    let silu_f32 = bench_sampled(args.samples, args.iters, || {
        dual_matvec_silu_mul_rowmajor_parallel(
            black_box(&x),
            black_box(&w_f32),
            black_box(&w2_f32),
            args.inter,
            args.hidden,
            black_box(&mut out_f32),
        );
        black_box(out_f32[0]);
    });
    let silu_mixed = bench_sampled(args.samples, args.iters, || {
        dual_matvec_silu_mul_rowmajor_parallel_mixed(
            SliceRef::F32(black_box(&x)),
            SliceRef::BF16(black_box(&w_bf16)),
            SliceRef::BF16(black_box(&w2_bf16)),
            args.inter,
            args.hidden,
            black_box(&mut out_mixed),
        );
        black_box(out_mixed[0]);
    });
    dual_matvec_silu_mul_rowmajor_parallel(
        &x,
        &w_f32,
        &w2_f32,
        args.inter,
        args.hidden,
        &mut out_f32,
    );
    dual_matvec_silu_mul_rowmajor_parallel_mixed(
        SliceRef::F32(&x),
        SliceRef::BF16(&w_bf16),
        SliceRef::BF16(&w2_bf16),
        args.inter,
        args.hidden,
        &mut out_mixed,
    );
    print_case(
        "silu_fused",
        silu_f32,
        silu_mixed,
        max_abs_diff(&out_f32, &out_mixed),
        args.iters,
    );
    let silu_bf16 = bench_sampled(args.samples, args.iters, || {
        dual_matvec_silu_mul_rowmajor_parallel_mixed(
            SliceRef::BF16(black_box(&x_bf16)),
            SliceRef::BF16(black_box(&w_bf16)),
            SliceRef::BF16(black_box(&w2_bf16)),
            args.inter,
            args.hidden,
            black_box(&mut out_mixed),
        );
        black_box(out_mixed[0]);
    });
    dual_matvec_silu_mul_rowmajor_parallel_mixed(
        SliceRef::BF16(&x_bf16),
        SliceRef::BF16(&w_bf16),
        SliceRef::BF16(&w2_bf16),
        args.inter,
        args.hidden,
        &mut out_mixed,
    );
    print_case(
        "silu_bf16io",
        silu_f32,
        silu_bf16,
        max_abs_diff(&out_f32, &out_mixed),
        args.iters,
    );
    let silu_f16 = bench_sampled(args.samples, args.iters, || {
        dual_matvec_silu_mul_rowmajor_parallel_mixed(
            SliceRef::F16(black_box(&x_f16)),
            SliceRef::F16(black_box(&w_f16)),
            SliceRef::F16(black_box(&w2_f16)),
            args.inter,
            args.hidden,
            black_box(&mut out_mixed),
        );
        black_box(out_mixed[0]);
    });
    dual_matvec_silu_mul_rowmajor_parallel_mixed(
        SliceRef::F16(&x_f16),
        SliceRef::F16(&w_f16),
        SliceRef::F16(&w2_f16),
        args.inter,
        args.hidden,
        &mut out_mixed,
    );
    print_case(
        "silu_f16io",
        silu_f32,
        silu_f16,
        max_abs_diff(&out_f32, &out_mixed),
        args.iters,
    );

    let argmax_f32 = bench_sampled(args.samples, args.iters, || {
        let idx = matvec_argmax_rowmajor_parallel(
            black_box(&x),
            black_box(&vocab_w_f32),
            args.vocab,
            args.hidden,
        );
        black_box(idx);
    });
    let argmax_mixed = bench_sampled(args.samples, args.iters, || {
        let idx = matvec_argmax_rowmajor_parallel_mixed(
            SliceRef::F32(black_box(&x)),
            SliceRef::BF16(black_box(&vocab_w_bf16)),
            args.vocab,
            args.hidden,
        );
        black_box(idx);
    });
    let idx_f32 = matvec_argmax_rowmajor_parallel(&x, &vocab_w_f32, args.vocab, args.hidden);
    let idx_mixed = matvec_argmax_rowmajor_parallel_mixed(
        SliceRef::F32(&x),
        SliceRef::BF16(&vocab_w_bf16),
        args.vocab,
        args.hidden,
    );
    let argmax_diff = if idx_f32 == idx_mixed { 0.0 } else { 1.0 };
    print_case("argmax", argmax_f32, argmax_mixed, argmax_diff, args.iters);
    let argmax_bf16 = bench_sampled(args.samples, args.iters, || {
        let idx = matvec_argmax_rowmajor_parallel_mixed(
            SliceRef::BF16(black_box(&x_bf16)),
            SliceRef::BF16(black_box(&vocab_w_bf16)),
            args.vocab,
            args.hidden,
        );
        black_box(idx);
    });
    let idx_bf16 = matvec_argmax_rowmajor_parallel_mixed(
        SliceRef::BF16(&x_bf16),
        SliceRef::BF16(&vocab_w_bf16),
        args.vocab,
        args.hidden,
    );
    let argmax_bf16_diff = if idx_f32 == idx_bf16 { 0.0 } else { 1.0 };
    print_case(
        "argmax_bf16io",
        argmax_f32,
        argmax_bf16,
        argmax_bf16_diff,
        args.iters,
    );
    let argmax_f16 = bench_sampled(args.samples, args.iters, || {
        let idx = matvec_argmax_rowmajor_parallel_mixed(
            SliceRef::F16(black_box(&x_f16)),
            SliceRef::F16(black_box(&vocab_w_f16)),
            args.vocab,
            args.hidden,
        );
        black_box(idx);
    });
    let idx_f16 = matvec_argmax_rowmajor_parallel_mixed(
        SliceRef::F16(&x_f16),
        SliceRef::F16(&vocab_w_f16),
        args.vocab,
        args.hidden,
    );
    let argmax_f16_diff = if idx_f32 == idx_f16 { 0.0 } else { 1.0 };
    print_case(
        "argmax_f16io",
        argmax_f32,
        argmax_f16,
        argmax_f16_diff,
        args.iters,
    );

    set_default_parameter_dtype(DType::BF16);
    let input = Tensor::from_array_no_grad(
        Array::from_shape_vec((1, 1, args.hidden), x.clone())
            .unwrap()
            .into_dyn(),
    );
    let input_bf16 = Tensor::from_array_no_grad(
        Array::from_shape_vec((1, 1, args.hidden), x.clone())
            .unwrap()
            .into_dyn(),
    );
    let input_f16 = Tensor::from_array_no_grad(
        Array::from_shape_vec((1, 1, args.hidden), x.clone())
            .unwrap()
            .into_dyn(),
    );
    input_bf16.cast_inplace(DType::BF16);
    input_f16.cast_inplace(DType::F16);
    let weight_no_copy = Tensor::parameter(
        Array::from_shape_vec((args.inter, args.hidden), w_f32.clone())
            .unwrap()
            .into_dyn(),
    );
    let weight_cached = Tensor::parameter(
        Array::from_shape_vec((args.inter, args.hidden), w_f32.clone())
            .unwrap()
            .into_dyn(),
    );
    let weight_no_copy_f16 = make_parameter(&[args.inter, args.hidden], w_f32.clone(), DType::F16);
    let weight_cached_f16 = make_parameter(&[args.inter, args.hidden], w_f32.clone(), DType::F16);
    let weight_i8 = make_tensor(&[args.inter, args.hidden], w_f32.clone(), DType::I8);
    let weight_i8_ref =
        make_quantized_f32_tensor(&[args.inter, args.hidden], w_f32.clone(), DType::I8);

    set_allow_parameter_dtype_copies(false);
    no_grad(|| {
        let _ = black_box(matmul(&input, &weight_no_copy));
    });
    let tensor_no_copy = bench_sampled(args.samples, args.iters, || {
        no_grad(|| {
            let out = matmul(black_box(&input), black_box(&weight_no_copy));
            black_box(out.data_ref()[[0, 0, 0]]);
        });
    });

    set_allow_parameter_dtype_copies(true);
    no_grad(|| {
        let _ = black_box(matmul(&input, &weight_cached));
    });
    let tensor_cached = bench_sampled(args.samples, args.iters, || {
        no_grad(|| {
            let out = matmul(black_box(&input), black_box(&weight_cached));
            black_box(out.data_ref()[[0, 0, 0]]);
        });
    });
    print_pair("tensor_matmul", tensor_no_copy, tensor_cached, args.iters);

    set_allow_parameter_dtype_copies(false);
    no_grad(|| {
        let _ = black_box(matmul(&input_bf16, &weight_no_copy));
    });
    let tensor_bf16_no_copy = bench_sampled(args.samples, args.iters, || {
        no_grad(|| {
            let out = matmul(black_box(&input_bf16), black_box(&weight_no_copy));
            black_box(out.data_ref()[[0, 0, 0]]);
        });
    });

    set_allow_parameter_dtype_copies(true);
    no_grad(|| {
        let _ = black_box(matmul(&input_bf16, &weight_cached));
    });
    let tensor_bf16_cached = bench_sampled(args.samples, args.iters, || {
        no_grad(|| {
            let out = matmul(black_box(&input_bf16), black_box(&weight_cached));
            black_box(out.data_ref()[[0, 0, 0]]);
        });
    });
    print_pair(
        "tensor_matmul_bf16",
        tensor_bf16_no_copy,
        tensor_bf16_cached,
        args.iters,
    );

    set_allow_parameter_dtype_copies(false);
    no_grad(|| {
        let _ = black_box(matmul(&input_f16, &weight_no_copy_f16));
    });
    let tensor_f16_no_copy = bench_sampled(args.samples, args.iters, || {
        no_grad(|| {
            let out = matmul(black_box(&input_f16), black_box(&weight_no_copy_f16));
            black_box(out.data_ref()[[0, 0, 0]]);
        });
    });

    set_allow_parameter_dtype_copies(true);
    no_grad(|| {
        let _ = black_box(matmul(&input_f16, &weight_cached_f16));
    });
    let tensor_f16_cached = bench_sampled(args.samples, args.iters, || {
        no_grad(|| {
            let out = matmul(black_box(&input_f16), black_box(&weight_cached_f16));
            black_box(out.data_ref()[[0, 0, 0]]);
        });
    });
    print_pair(
        "tensor_matmul_f16",
        tensor_f16_no_copy,
        tensor_f16_cached,
        args.iters,
    );

    no_grad(|| {
        let _ = black_box(matmul(&input, &weight_i8_ref));
        let _ = black_box(matmul(&input, &weight_i8));
    });
    let tensor_i8_ref = bench_sampled(args.samples, args.iters, || {
        no_grad(|| {
            let out = matmul(black_box(&input), black_box(&weight_i8_ref));
            black_box(out.data_ref()[[0, 0, 0]]);
        });
    });
    let tensor_i8 = bench_sampled(args.samples, args.iters, || {
        no_grad(|| {
            let out = matmul(black_box(&input), black_box(&weight_i8));
            black_box(out.data_ref()[[0, 0, 0]]);
        });
    });
    print_pair_named(
        "tensor_matmul_i8",
        "qref",
        "i8",
        tensor_i8_ref,
        tensor_i8,
        args.iters,
    );

    let q_weight_data = make_f32(args.hidden * args.hidden);
    let k_weight_data = make_f32(args.hidden * args.hidden);
    let v_weight_data = make_f32(args.hidden * args.hidden);
    let q_weight_f32 = make_tensor(
        &[args.hidden, args.hidden],
        q_weight_data.clone(),
        DType::F32,
    );
    let k_weight_f32 = make_tensor(
        &[args.hidden, args.hidden],
        k_weight_data.clone(),
        DType::F32,
    );
    let v_weight_f32 = make_tensor(
        &[args.hidden, args.hidden],
        v_weight_data.clone(),
        DType::F32,
    );
    let q_weight_bf16 = make_tensor(
        &[args.hidden, args.hidden],
        q_weight_data.clone(),
        DType::BF16,
    );
    let k_weight_bf16 = make_tensor(
        &[args.hidden, args.hidden],
        k_weight_data.clone(),
        DType::BF16,
    );
    let v_weight_bf16 = make_tensor(
        &[args.hidden, args.hidden],
        v_weight_data.clone(),
        DType::BF16,
    );
    let q_weight_f16 = make_tensor(
        &[args.hidden, args.hidden],
        q_weight_data.clone(),
        DType::F16,
    );
    let k_weight_f16 = make_tensor(
        &[args.hidden, args.hidden],
        k_weight_data.clone(),
        DType::F16,
    );
    let v_weight_f16 = make_tensor(
        &[args.hidden, args.hidden],
        v_weight_data.clone(),
        DType::F16,
    );
    let q_weight_i8 = make_tensor(
        &[args.hidden, args.hidden],
        q_weight_data.clone(),
        DType::I8,
    );
    let k_weight_i8 = make_tensor(
        &[args.hidden, args.hidden],
        k_weight_data.clone(),
        DType::I8,
    );
    let v_weight_i8 = make_tensor(
        &[args.hidden, args.hidden],
        v_weight_data.clone(),
        DType::I8,
    );
    let q_weight_i8_ref =
        make_quantized_f32_tensor(&[args.hidden, args.hidden], q_weight_data, DType::I8);
    let k_weight_i8_ref =
        make_quantized_f32_tensor(&[args.hidden, args.hidden], k_weight_data, DType::I8);
    let v_weight_i8_ref =
        make_quantized_f32_tensor(&[args.hidden, args.hidden], v_weight_data, DType::I8);
    let decode_input_f32 = make_tensor(&[1, 1, args.hidden], x.clone(), DType::F32);
    let decode_input_bf16 = make_tensor(&[1, 1, args.hidden], x.clone(), DType::BF16);
    let decode_input_f16 = make_tensor(&[1, 1, args.hidden], x.clone(), DType::F16);
    let mut q_out = vec![0.0f32; args.hidden];
    let mut k_out = vec![0.0f32; args.hidden];
    let mut v_out = vec![0.0f32; args.hidden];
    let mut q_out_bf16 = vec![0.0f32; args.hidden];
    let mut k_out_bf16 = vec![0.0f32; args.hidden];
    let mut v_out_bf16 = vec![0.0f32; args.hidden];
    let mut q_out_f16 = vec![0.0f32; args.hidden];
    let mut k_out_f16 = vec![0.0f32; args.hidden];
    let mut v_out_f16 = vec![0.0f32; args.hidden];

    no_grad(|| {
        fused_qkv_decode_infer_into(
            &decode_input_f32,
            &q_weight_f32,
            &k_weight_f32,
            &v_weight_f32,
            &mut q_out,
            &mut k_out,
            &mut v_out,
        );
        fused_qkv_decode_infer_into(
            &decode_input_bf16,
            &q_weight_bf16,
            &k_weight_bf16,
            &v_weight_bf16,
            &mut q_out_bf16,
            &mut k_out_bf16,
            &mut v_out_bf16,
        );
    });
    let fused_qkv_f32 = bench_sampled(args.samples, args.iters, || {
        no_grad(|| {
            fused_qkv_decode_infer_into(
                black_box(&decode_input_f32),
                black_box(&q_weight_f32),
                black_box(&k_weight_f32),
                black_box(&v_weight_f32),
                black_box(&mut q_out),
                black_box(&mut k_out),
                black_box(&mut v_out),
            );
            black_box(q_out[0] + k_out[0] + v_out[0]);
        });
    });
    let fused_qkv_bf16 = bench_sampled(args.samples, args.iters, || {
        no_grad(|| {
            fused_qkv_decode_infer_into(
                black_box(&decode_input_bf16),
                black_box(&q_weight_bf16),
                black_box(&k_weight_bf16),
                black_box(&v_weight_bf16),
                black_box(&mut q_out_bf16),
                black_box(&mut k_out_bf16),
                black_box(&mut v_out_bf16),
            );
            black_box(q_out_bf16[0] + k_out_bf16[0] + v_out_bf16[0]);
        });
    });
    no_grad(|| {
        fused_qkv_decode_infer_into(
            &decode_input_f32,
            &q_weight_f32,
            &k_weight_f32,
            &v_weight_f32,
            &mut q_out,
            &mut k_out,
            &mut v_out,
        );
        fused_qkv_decode_infer_into(
            &decode_input_bf16,
            &q_weight_bf16,
            &k_weight_bf16,
            &v_weight_bf16,
            &mut q_out_bf16,
            &mut k_out_bf16,
            &mut v_out_bf16,
        );
    });
    let fused_qkv_diff = max_abs_diff(&q_out, &q_out_bf16)
        .max(max_abs_diff(&k_out, &k_out_bf16))
        .max(max_abs_diff(&v_out, &v_out_bf16));
    print_case(
        "fused_qkv",
        fused_qkv_f32,
        fused_qkv_bf16,
        fused_qkv_diff,
        args.iters,
    );
    no_grad(|| {
        fused_qkv_decode_infer_into(
            &decode_input_f16,
            &q_weight_f16,
            &k_weight_f16,
            &v_weight_f16,
            &mut q_out_f16,
            &mut k_out_f16,
            &mut v_out_f16,
        );
    });
    let fused_qkv_f16 = bench_sampled(args.samples, args.iters, || {
        no_grad(|| {
            fused_qkv_decode_infer_into(
                black_box(&decode_input_f16),
                black_box(&q_weight_f16),
                black_box(&k_weight_f16),
                black_box(&v_weight_f16),
                black_box(&mut q_out_f16),
                black_box(&mut k_out_f16),
                black_box(&mut v_out_f16),
            );
            black_box(q_out_f16[0] + k_out_f16[0] + v_out_f16[0]);
        });
    });
    no_grad(|| {
        fused_qkv_decode_infer_into(
            &decode_input_f16,
            &q_weight_f16,
            &k_weight_f16,
            &v_weight_f16,
            &mut q_out_f16,
            &mut k_out_f16,
            &mut v_out_f16,
        );
    });
    let fused_qkv_f16_diff = max_abs_diff(&q_out, &q_out_f16)
        .max(max_abs_diff(&k_out, &k_out_f16))
        .max(max_abs_diff(&v_out, &v_out_f16));
    print_case(
        "fused_qkv_f16",
        fused_qkv_f32,
        fused_qkv_f16,
        fused_qkv_f16_diff,
        args.iters,
    );
    no_grad(|| {
        fused_qkv_decode_infer_into(
            &decode_input_f32,
            &q_weight_i8_ref,
            &k_weight_i8_ref,
            &v_weight_i8_ref,
            &mut q_out,
            &mut k_out,
            &mut v_out,
        );
        fused_qkv_decode_infer_into(
            &decode_input_f32,
            &q_weight_i8,
            &k_weight_i8,
            &v_weight_i8,
            &mut q_out_bf16,
            &mut k_out_bf16,
            &mut v_out_bf16,
        );
    });
    let fused_qkv_i8_ref = bench_sampled(args.samples, args.iters, || {
        no_grad(|| {
            fused_qkv_decode_infer_into(
                black_box(&decode_input_f32),
                black_box(&q_weight_i8_ref),
                black_box(&k_weight_i8_ref),
                black_box(&v_weight_i8_ref),
                black_box(&mut q_out),
                black_box(&mut k_out),
                black_box(&mut v_out),
            );
            black_box(q_out[0] + k_out[0] + v_out[0]);
        });
    });
    let fused_qkv_i8 = bench_sampled(args.samples, args.iters, || {
        no_grad(|| {
            fused_qkv_decode_infer_into(
                black_box(&decode_input_f32),
                black_box(&q_weight_i8),
                black_box(&k_weight_i8),
                black_box(&v_weight_i8),
                black_box(&mut q_out_bf16),
                black_box(&mut k_out_bf16),
                black_box(&mut v_out_bf16),
            );
            black_box(q_out_bf16[0] + k_out_bf16[0] + v_out_bf16[0]);
        });
    });
    let fused_qkv_i8_diff = max_abs_diff(&q_out, &q_out_bf16)
        .max(max_abs_diff(&k_out, &k_out_bf16))
        .max(max_abs_diff(&v_out, &v_out_bf16));
    print_case(
        "fused_qkv_i8",
        fused_qkv_i8_ref,
        fused_qkv_i8,
        fused_qkv_i8_diff,
        args.iters,
    );

    let gate_weight_f32 = make_tensor(&[args.inter, args.hidden], w_f32.clone(), DType::F32);
    let up_weight_f32 = make_tensor(&[args.inter, args.hidden], w2_f32.clone(), DType::F32);
    let gate_weight_bf16 = make_tensor(&[args.inter, args.hidden], w_f32.clone(), DType::BF16);
    let up_weight_bf16 = make_tensor(&[args.inter, args.hidden], w2_f32.clone(), DType::BF16);
    let gate_weight_f16 = make_tensor(&[args.inter, args.hidden], w_f32.clone(), DType::F16);
    let up_weight_f16 = make_tensor(&[args.inter, args.hidden], w2_f32.clone(), DType::F16);
    let gate_weight_i8 = make_tensor(&[args.inter, args.hidden], w_f32.clone(), DType::I8);
    let up_weight_i8 = make_tensor(&[args.inter, args.hidden], w2_f32.clone(), DType::I8);
    let gate_weight_i8_ref =
        make_quantized_f32_tensor(&[args.inter, args.hidden], w_f32.clone(), DType::I8);
    let up_weight_i8_ref =
        make_quantized_f32_tensor(&[args.inter, args.hidden], w2_f32.clone(), DType::I8);
    let mut fused_out_f32 = vec![0.0f32; args.inter];
    let mut fused_out_bf16 = vec![0.0f32; args.inter];
    let mut fused_out_f16 = vec![0.0f32; args.inter];
    no_grad(|| {
        fused_gate_up_silu_infer_into(
            &decode_input_f32,
            &gate_weight_f32,
            &up_weight_f32,
            &mut fused_out_f32,
        );
        fused_gate_up_silu_infer_into(
            &decode_input_bf16,
            &gate_weight_bf16,
            &up_weight_bf16,
            &mut fused_out_bf16,
        );
    });
    let fused_gate_f32 = bench_sampled(args.samples, args.iters, || {
        no_grad(|| {
            fused_gate_up_silu_infer_into(
                black_box(&decode_input_f32),
                black_box(&gate_weight_f32),
                black_box(&up_weight_f32),
                black_box(&mut fused_out_f32),
            );
            black_box(fused_out_f32[0]);
        });
    });
    let fused_gate_bf16 = bench_sampled(args.samples, args.iters, || {
        no_grad(|| {
            fused_gate_up_silu_infer_into(
                black_box(&decode_input_bf16),
                black_box(&gate_weight_bf16),
                black_box(&up_weight_bf16),
                black_box(&mut fused_out_bf16),
            );
            black_box(fused_out_bf16[0]);
        });
    });
    no_grad(|| {
        fused_gate_up_silu_infer_into(
            &decode_input_f32,
            &gate_weight_f32,
            &up_weight_f32,
            &mut fused_out_f32,
        );
        fused_gate_up_silu_infer_into(
            &decode_input_bf16,
            &gate_weight_bf16,
            &up_weight_bf16,
            &mut fused_out_bf16,
        );
    });
    print_case(
        "fused_gateup",
        fused_gate_f32,
        fused_gate_bf16,
        max_abs_diff(&fused_out_f32, &fused_out_bf16),
        args.iters,
    );
    no_grad(|| {
        fused_gate_up_silu_infer_into(
            &decode_input_f16,
            &gate_weight_f16,
            &up_weight_f16,
            &mut fused_out_f16,
        );
    });
    let fused_gate_f16 = bench_sampled(args.samples, args.iters, || {
        no_grad(|| {
            fused_gate_up_silu_infer_into(
                black_box(&decode_input_f16),
                black_box(&gate_weight_f16),
                black_box(&up_weight_f16),
                black_box(&mut fused_out_f16),
            );
            black_box(fused_out_f16[0]);
        });
    });
    no_grad(|| {
        fused_gate_up_silu_infer_into(
            &decode_input_f16,
            &gate_weight_f16,
            &up_weight_f16,
            &mut fused_out_f16,
        );
    });
    print_case(
        "fused_gate_f16",
        fused_gate_f32,
        fused_gate_f16,
        max_abs_diff(&fused_out_f32, &fused_out_f16),
        args.iters,
    );
    no_grad(|| {
        fused_gate_up_silu_infer_into(
            &decode_input_f32,
            &gate_weight_i8_ref,
            &up_weight_i8_ref,
            &mut fused_out_f32,
        );
        fused_gate_up_silu_infer_into(
            &decode_input_f32,
            &gate_weight_i8,
            &up_weight_i8,
            &mut fused_out_bf16,
        );
    });
    let fused_gate_i8_ref = bench_sampled(args.samples, args.iters, || {
        no_grad(|| {
            fused_gate_up_silu_infer_into(
                black_box(&decode_input_f32),
                black_box(&gate_weight_i8_ref),
                black_box(&up_weight_i8_ref),
                black_box(&mut fused_out_f32),
            );
            black_box(fused_out_f32[0]);
        });
    });
    let fused_gate_i8 = bench_sampled(args.samples, args.iters, || {
        no_grad(|| {
            fused_gate_up_silu_infer_into(
                black_box(&decode_input_f32),
                black_box(&gate_weight_i8),
                black_box(&up_weight_i8),
                black_box(&mut fused_out_bf16),
            );
            black_box(fused_out_bf16[0]);
        });
    });
    print_case(
        "fused_gate_i8",
        fused_gate_i8_ref,
        fused_gate_i8,
        max_abs_diff(&fused_out_f32, &fused_out_bf16),
        args.iters,
    );

    let q_weight_param = make_parameter(
        &[args.hidden, args.hidden],
        make_f32(args.hidden * args.hidden),
        DType::BF16,
    );
    let k_weight_param = make_parameter(
        &[args.hidden, args.hidden],
        make_f32(args.hidden * args.hidden),
        DType::BF16,
    );
    let v_weight_param = make_parameter(
        &[args.hidden, args.hidden],
        make_f32(args.hidden * args.hidden),
        DType::BF16,
    );
    let gate_weight_param = make_parameter(&[args.inter, args.hidden], w_f32.clone(), DType::BF16);
    let up_weight_param = make_parameter(&[args.inter, args.hidden], w2_f32.clone(), DType::BF16);
    let q_weight_param_f16 = make_parameter(
        &[args.hidden, args.hidden],
        make_f32(args.hidden * args.hidden),
        DType::F16,
    );
    let k_weight_param_f16 = make_parameter(
        &[args.hidden, args.hidden],
        make_f32(args.hidden * args.hidden),
        DType::F16,
    );
    let v_weight_param_f16 = make_parameter(
        &[args.hidden, args.hidden],
        make_f32(args.hidden * args.hidden),
        DType::F16,
    );
    let gate_weight_param_f16 =
        make_parameter(&[args.inter, args.hidden], w_f32.clone(), DType::F16);
    let up_weight_param_f16 =
        make_parameter(&[args.inter, args.hidden], w2_f32.clone(), DType::F16);

    set_allow_parameter_dtype_copies(false);
    no_grad(|| {
        fused_qkv_decode_infer_into(
            &decode_input_bf16,
            &q_weight_param,
            &k_weight_param,
            &v_weight_param,
            &mut q_out_bf16,
            &mut k_out_bf16,
            &mut v_out_bf16,
        );
        fused_gate_up_silu_infer_into(
            &decode_input_bf16,
            &gate_weight_param,
            &up_weight_param,
            &mut fused_out_bf16,
        );
    });
    let fused_qkv_no_copy = bench_sampled(args.samples, args.iters, || {
        no_grad(|| {
            fused_qkv_decode_infer_into(
                black_box(&decode_input_bf16),
                black_box(&q_weight_param),
                black_box(&k_weight_param),
                black_box(&v_weight_param),
                black_box(&mut q_out_bf16),
                black_box(&mut k_out_bf16),
                black_box(&mut v_out_bf16),
            );
            black_box(q_out_bf16[0] + k_out_bf16[0] + v_out_bf16[0]);
        });
    });
    let fused_gate_no_copy = bench_sampled(args.samples, args.iters, || {
        no_grad(|| {
            fused_gate_up_silu_infer_into(
                black_box(&decode_input_bf16),
                black_box(&gate_weight_param),
                black_box(&up_weight_param),
                black_box(&mut fused_out_bf16),
            );
            black_box(fused_out_bf16[0]);
        });
    });

    set_allow_parameter_dtype_copies(true);
    no_grad(|| {
        fused_qkv_decode_infer_into(
            &decode_input_bf16,
            &q_weight_param,
            &k_weight_param,
            &v_weight_param,
            &mut q_out_bf16,
            &mut k_out_bf16,
            &mut v_out_bf16,
        );
        fused_gate_up_silu_infer_into(
            &decode_input_bf16,
            &gate_weight_param,
            &up_weight_param,
            &mut fused_out_bf16,
        );
    });
    let fused_qkv_cached = bench_sampled(args.samples, args.iters, || {
        no_grad(|| {
            fused_qkv_decode_infer_into(
                black_box(&decode_input_bf16),
                black_box(&q_weight_param),
                black_box(&k_weight_param),
                black_box(&v_weight_param),
                black_box(&mut q_out_bf16),
                black_box(&mut k_out_bf16),
                black_box(&mut v_out_bf16),
            );
            black_box(q_out_bf16[0] + k_out_bf16[0] + v_out_bf16[0]);
        });
    });
    let fused_gate_cached = bench_sampled(args.samples, args.iters, || {
        no_grad(|| {
            fused_gate_up_silu_infer_into(
                black_box(&decode_input_bf16),
                black_box(&gate_weight_param),
                black_box(&up_weight_param),
                black_box(&mut fused_out_bf16),
            );
            black_box(fused_out_bf16[0]);
        });
    });
    print_pair(
        "fused_qkv_cache",
        fused_qkv_no_copy,
        fused_qkv_cached,
        args.iters,
    );
    print_pair(
        "fused_gate_cache",
        fused_gate_no_copy,
        fused_gate_cached,
        args.iters,
    );

    set_allow_parameter_dtype_copies(false);
    no_grad(|| {
        fused_qkv_decode_infer_into(
            &decode_input_f16,
            &q_weight_param_f16,
            &k_weight_param_f16,
            &v_weight_param_f16,
            &mut q_out_f16,
            &mut k_out_f16,
            &mut v_out_f16,
        );
        fused_gate_up_silu_infer_into(
            &decode_input_f16,
            &gate_weight_param_f16,
            &up_weight_param_f16,
            &mut fused_out_f16,
        );
    });
    let fused_qkv_no_copy_f16 = bench_sampled(args.samples, args.iters, || {
        no_grad(|| {
            fused_qkv_decode_infer_into(
                black_box(&decode_input_f16),
                black_box(&q_weight_param_f16),
                black_box(&k_weight_param_f16),
                black_box(&v_weight_param_f16),
                black_box(&mut q_out_f16),
                black_box(&mut k_out_f16),
                black_box(&mut v_out_f16),
            );
            black_box(q_out_f16[0] + k_out_f16[0] + v_out_f16[0]);
        });
    });
    let fused_gate_no_copy_f16 = bench_sampled(args.samples, args.iters, || {
        no_grad(|| {
            fused_gate_up_silu_infer_into(
                black_box(&decode_input_f16),
                black_box(&gate_weight_param_f16),
                black_box(&up_weight_param_f16),
                black_box(&mut fused_out_f16),
            );
            black_box(fused_out_f16[0]);
        });
    });

    set_allow_parameter_dtype_copies(true);
    no_grad(|| {
        fused_qkv_decode_infer_into(
            &decode_input_f16,
            &q_weight_param_f16,
            &k_weight_param_f16,
            &v_weight_param_f16,
            &mut q_out_f16,
            &mut k_out_f16,
            &mut v_out_f16,
        );
        fused_gate_up_silu_infer_into(
            &decode_input_f16,
            &gate_weight_param_f16,
            &up_weight_param_f16,
            &mut fused_out_f16,
        );
    });
    let fused_qkv_cached_f16 = bench_sampled(args.samples, args.iters, || {
        no_grad(|| {
            fused_qkv_decode_infer_into(
                black_box(&decode_input_f16),
                black_box(&q_weight_param_f16),
                black_box(&k_weight_param_f16),
                black_box(&v_weight_param_f16),
                black_box(&mut q_out_f16),
                black_box(&mut k_out_f16),
                black_box(&mut v_out_f16),
            );
            black_box(q_out_f16[0] + k_out_f16[0] + v_out_f16[0]);
        });
    });
    let fused_gate_cached_f16 = bench_sampled(args.samples, args.iters, || {
        no_grad(|| {
            fused_gate_up_silu_infer_into(
                black_box(&decode_input_f16),
                black_box(&gate_weight_param_f16),
                black_box(&up_weight_param_f16),
                black_box(&mut fused_out_f16),
            );
            black_box(fused_out_f16[0]);
        });
    });
    print_pair(
        "fused_qkv_cache_f16",
        fused_qkv_no_copy_f16,
        fused_qkv_cached_f16,
        args.iters,
    );
    print_pair(
        "fused_gate_cache_f16",
        fused_gate_no_copy_f16,
        fused_gate_cached_f16,
        args.iters,
    );

    let opt_len = args.hidden.saturating_mul(8).max(1024);
    let sgd_f32 = bench_sgd_update(args.samples, args.iters, DType::F32, opt_len);
    let sgd_bf16 = bench_sgd_update(args.samples, args.iters, DType::BF16, opt_len);
    let sgd_f16 = bench_sgd_update(args.samples, args.iters, DType::F16, opt_len);
    let sgd_i8 = bench_sgd_update(args.samples, args.iters, DType::I8, opt_len);
    print_case("sgd_bf16", sgd_f32, sgd_bf16, 0.0, args.iters);
    print_case("sgd_f16", sgd_f32, sgd_f16, 0.0, args.iters);
    print_case("sgd_i8", sgd_f32, sgd_i8, 0.0, args.iters);

    let sgd_momentum_f32 = bench_sgd_momentum_update(args.samples, args.iters, DType::F32, opt_len);
    let sgd_momentum_bf16 =
        bench_sgd_momentum_update(args.samples, args.iters, DType::BF16, opt_len);
    let sgd_momentum_f16 = bench_sgd_momentum_update(args.samples, args.iters, DType::F16, opt_len);
    let sgd_momentum_i8 = bench_sgd_momentum_update(args.samples, args.iters, DType::I8, opt_len);
    print_case(
        "sgd_momentum_bf16",
        sgd_momentum_f32,
        sgd_momentum_bf16,
        0.0,
        args.iters,
    );
    print_case(
        "sgd_momentum_f16",
        sgd_momentum_f32,
        sgd_momentum_f16,
        0.0,
        args.iters,
    );
    print_case(
        "sgd_momentum_i8",
        sgd_momentum_f32,
        sgd_momentum_i8,
        0.0,
        args.iters,
    );

    let adam_f32 = bench_adam_update(args.samples, args.iters, DType::F32, opt_len);
    let adam_bf16 = bench_adam_update(args.samples, args.iters, DType::BF16, opt_len);
    let adam_f16 = bench_adam_update(args.samples, args.iters, DType::F16, opt_len);
    let adam_i8 = bench_adam_update(args.samples, args.iters, DType::I8, opt_len);
    print_case("adam_bf16", adam_f32, adam_bf16, 0.0, args.iters);
    print_case("adam_f16", adam_f32, adam_f16, 0.0, args.iters);
    print_case("adam_i8", adam_f32, adam_i8, 0.0, args.iters);

    Ok(())
}

#[cfg(test)]
mod accuracy_tests {
    use super::max_abs_diff;

    #[test]
    fn max_abs_diff_rejects_non_finite_values() {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(max_abs_diff(&[value], &[value]).is_infinite());
        }
    }
}
