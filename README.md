# Lumen

A Rust neural-network library with dynamic autograd, reusable layers, dtype-aware
CPU/CUDA execution, and a built-in Llama decoder.

[中文文档](./README_zh-CN.md) · [Changelog](./CHANGELOG.md) · [License](./LICENSE)

[Features](#features) · [Quick start](#quick-start) · [Precision](#precision-and-training) ·
[CPU scheduling](#cpu-scheduling) · [Inference](#model-inference) · [Benchmarks](#benchmarks) ·
[Testing](#testing) · [Repository layout](#repository-layout)

## Features

- Tensor operations and dynamic automatic differentiation with F32 gradients.
- Linear, Embedding, Dropout, Conv2D, MaxPool2D, RNN, GRU and LSTM layers.
- LayerNorm, RMSNorm and scaled dot-product attention for MHA/GQA/MQA and cross-attention.
- SGD, momentum and Adam, with optional F32 master weights for low-precision training.
- F32, F16, BF16 and I8 storage, safetensors loading, streaming and offline I8 quantization.
- Portable CPU execution, optional x86/ARM kernels, and caller-controlled Rayon scheduling.
- Optional CUDA kernels, cuBLAS and cuDNN, including resident tensors and gradient paths.
- Llama inference with RoPE, grouped KV heads, greedy/sampling decode and KV caches.

The Tensor/layer APIs are independent of the built-in model. The CLI expects a
checkpoint matching its configured Llama architecture.

## Quick start

### Build from source

```bash
git clone https://github.com/chen-197/Lumen.git
cd Lumen
cargo build --release
```

Use a recent stable Rust toolchain (edition 2024). CUDA is optional for CPU builds.
To use the library directly from Git:

```toml
[dependencies]
lumen = { git = "https://github.com/chen-197/Lumen.git" }
ndarray = "0.15"
rayon = "1.7"
```

### Train a small model

```rust
use lumen::autograd::Tensor;
use lumen::loss::MSELoss;
use lumen::ops::matmul::matmul;
use lumen::optim::{Optimizer, SGD};
use lumen::precision::DType;
use ndarray::arr2;

fn main() {
    let input = Tensor::from_array_no_grad(arr2(&[[1.0, 0.0], [0.0, 1.0]]).into_dyn());
    let target = Tensor::from_array_no_grad(arr2(&[[2.0], [4.0]]).into_dyn());
    let weight = Tensor::parameter_with_dtype(arr2(&[[0.0, 0.0]]).into_dyn(), DType::F32);
    let mut optimizer = SGD::new_with_dtype(vec![weight.clone()], 0.1, DType::F32);

    for _ in 0..100 {
        optimizer.zero_grad();
        let prediction = matmul(&input, &weight);
        let loss = MSELoss::apply(&prediction, &target);
        loss.backward();
        optimizer.step();
    }

    let loss = MSELoss::apply(&matmul(&input, &weight), &target);
    let value = *loss.data().first().unwrap();
    println!("loss = {value:.6}");
    assert!(value < 1e-6);
}
```

`matmul` computes `A[..., K] @ B[N, K]ᵀ`, producing `[..., N]`.
Create a fresh forward graph after each parameter update and call `zero_grad()`
before the next batch. Leaf gradients accumulate across backward calls;
`backward_with_grad(seed)` accepts a custom seed.

### General attention and normalization

```rust
use lumen::autograd::Tensor;
use lumen::layers::{AttentionMask, AttentionOptions, LayerNorm, scaled_dot_product_attention};
use lumen::module::Module;
use ndarray::{ArrayD, IxDyn};

let q = Tensor::new(ArrayD::zeros(IxDyn(&[2, 4, 3, 5])));
let k = Tensor::new(ArrayD::zeros(IxDyn(&[2, 2, 7, 5])));
let v = Tensor::new(ArrayD::ones(IxDyn(&[2, 2, 7, 6])));
let keep = ArrayD::from_elem(IxDyn(&[2, 1, 1, 7]), true);
let context = scaled_dot_product_attention(
    &q, &k, &v, Some(AttentionMask::Keep(&keep)), AttentionOptions::default(),
);
let output = LayerNorm::new(vec![3, 6], 1e-5).forward(context);
assert_eq!(output.shape_vec(), vec![2, 4, 3, 6]);
```

Q/K/V have shapes `[B,Hq,Q,D]`, `[B,Hkv,K,D]`, `[B,Hkv,K,Dv]`.
Boolean masks use `true` to permit attention and broadcast to `[B,Hq,Q,K]`.
Additive masks are applied after scaling. Fully masked rows return zero context
and score gradients. `AttentionOptions` provides causal masking and a query offset;
projection, positional encoding and cache updates remain the caller's responsibility.

## Build options

```bash
# CUDA backend
cargo build --release --features cuda

# x86 kernels and benchmark tools
cargo build --release --features "dev-tools x86-fp-kernels x86-int8-kernels"

# Native code generation for a machine-specific performance build
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

In PowerShell, set `$env:RUSTFLAGS = "-C target-cpu=native"` before building.
CUDA builds need a CUDA toolkit and compatible NVIDIA driver. `build.rs` discovers
CUDA/cuDNN through environment variables, `nvcc` and system installation paths.

| Feature | Purpose |
|---|---|
| `cuda` | CUDA C++ backend, cuBLAS and optional cuDNN |
| `dev-tools` | Benchmark and diagnostic binaries |
| `cpu-profiling` | Kernel timing instrumentation; includes `dev-tools` |
| `x86-fp-kernels` | Stable x86 F32/F16/BF16 kernels |
| `x86-fp-kernels-nightly` | Native AVX-512 FP16 intrinsics; requires nightly |
| `x86-int8-kernels` | x86 I8 kernels |
| `arm64-fp-kernels` | ARM64 floating-point kernels |
| `arm64-int8-kernels` | ARM64 I8 kernels |

Architecture features are opt-in; runtime detection selects supported kernels.
Native AVX-512 FP16 can be built with
`cargo +nightly build --release --features x86-fp-kernels-nightly`.
Use a non-instrumented build for performance comparisons.

## Precision and training

- Parameters, activations and KV storage can use different dtypes on supported paths.
  KV caches use floating-point storage; I8 weights commonly pair with BF16 runtime/KV.
- Gradients are F32. Operations tracked by autograd produce F32 outputs; same-dtype
  lowp matrix inference preserves its storage dtype. Mixed F32/F16 activation ×
  BF16 weight paths preserve the declared activation values.
- Matmul and batch matmul support empty axes and zero-length reductions. Frozen
  inputs do not receive operator gradients. Flatten and shape operations preserve
  empty output shapes and their corresponding input gradient shapes.
- `no_grad` disables graph construction. `detach()` disables tracking and owns
  independent storage. Use `backward_with_grad`, rather than preloading a seed with `add_grad`.
- `backward_with_cuda_grad` accepts a resident F32 CUDA seed. Reading a CUDA
  gradient through `grad()`, `grad_arc()` or `grad_ref()` materializes a cached
  host representation. Keep gradient computation on device to avoid this transfer.
- CPU optimizer steps retain gradients; successful CUDA updates consume the
  applied gradients. Call `zero_grad()` before each batch on either backend.
  SGD and Adam validate every gradient shape before changing any parameter or
  optimizer state; a shape mismatch panics.
- Module checkpoints contain parameters. Optimizer moments, master-weight residuals
  and RNG state are not serialized.

For low-precision training, choose F32 optimizer state and call
`.with_fp32_master_weights()` on SGD or Adam. For trainable I8 ranges,
`.with_dynamic_i8_master_weights()` recalibrates scales each step. These options
must be chosen before training and add F32 master storage. Rebuild the optimizer
after replacing parameter storage, shapes, dtypes, scales or devices.

I8 activation quantization is explicit: `--activation-quantization dynamic-i8`.
The default is `off`; use the same policy when comparing outputs or performance.

## CPU scheduling

```rust
use lumen::cpu::CpuKernelContext;
use std::{num::NonZeroUsize, sync::Arc};

let pool = Arc::new(rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap());
let context = CpuKernelContext::with_pool(pool)
    .with_task_limit(NonZeroUsize::new(2).unwrap());

context.scope(|| {
    // Forward, backward and optimizer calls stay on the calling thread.
    // Numerical kernels use the shared pool within this task budget.
});
```

Create pools outside compute loops and share them between callers. `scope` keeps
Tensors and autograd state on the calling thread; numerical slices enter the pool.
The task budget is per integrated kernel, not a process-wide thread limit.
Small jobs and nested numerical work execute serially. The library does not set
a global pool, pin threads or infer policy from a particular CPU model.

MatVec, matmul/batch-matmul forward and backward, fused projections, argmax,
prefill, inference attention and native lowp optimizer-state kernels observe
these contexts. Enter the scope around **backward as well as forward**; creating
a graph inside it does not bind a later traversal to that policy.
`CpuInferenceContext` provides separate prefill/decode budgets over a shared pool.
Tensor handles use `Rc<RefCell<_>>` and are not `Send`/`Sync`; separate callers
own their model/Tensor handles while sharing numerical worker pools.
No-grad, precision, initialization and device-selection guards are also
`!Send`/`!Sync`. Create and drop these scopes on the thread that executes the
computation so their thread-local settings are restored correctly.

## Model inference

```bash
cargo run --release --bin lumen -- \
  --weights path/to/model.safetensors \
  --tokenizer path/to/tokenizer.json
```

Add `--features cuda` and `--device cuda` for GPU inference. Use
`cargo run --release --bin lumen -- --help` for precision, cache, sampling,
streaming and CPU task options. Parameter, activation and KV dtypes must be
supported by the selected path. Loading a different architecture requires a
matching model implementation/configuration and tokenizer.

### Offline quantization

```bash
cargo run --release --bin quantize_safetensors -- \
  --input path/to/model.safetensors \
  --output path/to/model.i8.safetensors \
  --dtype i8
```

`--scale` optionally supplies a manual I8 scale. The loader also supports
on-load quantization and streamed safetensors loading.

The converter uses symmetric per-tensor quantization to `[-127, 127]`, with
half-integer ties rounded away from zero. Manual scales must be finite and
positive; positive subnormal scales are supported. Each I8 tensor is accompanied
by a scalar F32 `<tensor-name>.scale` entry required when loading its weights.

## Benchmarks

Benchmark tools require `dev-tools` and should run with `--release`.

| Binary | Purpose |
|---|---|
| `kernel_bench` | CPU kernel microbenchmarks |
| `generic_prefill_bench` | Matmul, batch matmul and general attention |
| `cpu_backward_bench` | Matmul and batch-matmul gradients |
| `cpu_projection_bench` | Projection, QKV and gated MLP kernels |
| `cpu_decode_bench` | Decode working sets and vocabulary argmax |
| `cpu_scaling_bench` | Pool size, task budgets and concurrent callers |
| `cuda_cpu_bench` | CPU/CUDA operators and training paths |
| `prefill_decode_bench` | Real-weight prefill/decode, replay and sampling |
| `manual_state_check` | Public API, checkpoint and state checks |

```bash
cargo run --release --features "dev-tools x86-fp-kernels x86-int8-kernels" \
  --bin generic_prefill_bench -- --threads 4 --tasks 2 --warmup 5 --samples 7 --iters 5

cargo run --release --features "dev-tools cuda x86-fp-kernels x86-int8-kernels" \
  --bin prefill_decode_bench -- --help
```

`--samples-output` saves model timing samples; `--replay-token-ids` provides
fixed-input decode measurements. Check free-generated text separately from replay.
`cpu-profiling` adds diagnostic timing regions; its overhead should be excluded
from final performance comparisons.

For a representative CPU suite, build the tools once and run the Python runner
against the resulting release directory:

```bash
cargo build --release --locked --features dev-tools --bins
python dev-tools/check_performance.py --threads 8 --plan
python dev-tools/check_performance.py --threads 8 \
  --binary-directory path/to/release --output-directory target/performance-new
```

The default selection runs 21 processes at more than one worker, with warmup
before timing. `--select argmax` narrows the selection; `--extended` enables
additional task budgets and precision/request combinations. Use a new output
directory for each run and run timing serially, without competing builds or
reference checks. See the [benchmark runner documentation](./dev-tools/README.md#bounded-performance-runs)
for output and runtime-library requirements.

### TinyLlama 1.1B comparison

Measured 2026-10-28 on Windows/x86-64 with Ryzen 9 8945HX, RTX 5070 Laptop
(8 GB) and eight CPU compute threads. Rust 1.98.1 release builds used
`target-cpu=native` with CUDA and x86 floating/I8 kernels enabled.
PyTorch 2.14.0+cu130 / Transformers 5.17.0 used SDPA + DynamicCache,
TF32 and reduced-precision reductions disabled, without graph compilation.
Both engines used a 46-token prompt and a fixed 64-token replay, with two
warmups and five measured runs per process. CUDA F16/BF16 used counterbalanced
Lumen/PyTorch/PyTorch/Lumen ordering after the initial prefill range exceeded
25% of its median in either engine, giving ten measured samples per engine;
other rows use five samples per engine. All samples are retained, and the table
shows medians. Model loading and external input construction are excluded;
forward, argmax, host-token completion and the final decode forward are included.
Lumen uses preallocated KV storage, so the cache implementations differ.

| Device / weights | Lumen prefill ms | PyTorch prefill ms | Lumen decode tok/s | PyTorch decode tok/s |
|---|---:|---:|---:|---:|
| cpu / F32 | 312.332 | 273.558 | 11.46 | 8.52 |
| cpu / F16 | 269.304 | 1160.605 | 20.98 | 18.62 |
| cpu / BF16 | 444.203 | 102.229 | 21.10 | 16.70 |
| cpu / I8 (BF16 runtime/KV) | 264.231 | — | 35.16 | — |
| cuda / F32 | 31.008 | 22.979 | 49.46 | 58.73 |
| cuda / F16 | 19.883 | 17.697 | 90.88 | 52.91 |
| cuda / BF16 | 20.115 | 17.982 | 90.82 | 49.88 |
| cuda / I8 (BF16 runtime/KV) | 39.530 | — | 149.21 | — |

I8 uses BF16 activations/runtime/KV with dynamic activation I8 disabled; no
PyTorch I8 speed ratio is implied. CPU BF16 prefill takes about 4.35 times as long
as PyTorch here; CUDA F32 decode throughput is about 15.8% lower. Results depend on
shape, precision, cache strategy and hardware;
GPU reference timing also varies. This comparison does not cover compiled graphs,
StaticCache or dedicated inference engines.

## Testing

```bash
cargo test --release --locked --all-targets --features dev-tools
cargo test --release --locked --doc --features dev-tools
cargo fmt --all --check
```

Independent references require a Python environment with PyTorch. Real-weight
checks also require Transformers and matching model/tokenizer files. Run the
PowerShell scripts with PowerShell 7. Choose the runner for the required scope:

```powershell
# Independent CPU numerical references
./dev-tools/check_operator_accuracy.ps1 -Python python

# Release checks on portable and native CPU
./dev-tools/check_release.ps1 -NativeCpu -Python python

# CPU/CUDA release checks, including endurance and real-weight KV references
./dev-tools/check_release.ps1 -Cuda -Endurance -Python python -ModelDirectory "path/to/TinyLlama-1.1B-Chat-v1.0"
```

Use `check_release.ps1 -NativeCpu` for portable and native CPU coverage without
CUDA. Run endurance once on the optimized configuration; legacy scheduler
ablations and exhaustive performance grids are opt-in investigations. See
[suite selection](./tests/README.md#selecting-a-suite) and the
[representative performance runner](./dev-tools/README.md#bounded-performance-runs).

Default tests do not require model downloads. Tests cover scalar FP64/PyTorch
references, mixed and strided tensors, I8 scale/rounding/saturation boundaries,
empty outputs and gradients, VJPs,
frozen inputs, retained graphs, gradient accumulation, checkpoints, optimizer
state, KV routes and ownership lifetimes. Documentation tests enforce the
thread restrictions of execution/precision guards.

Endurance checks run small networks for 1000 updates and require finite values
and an overall downward loss trend, allowing intermediate rebounds. Real-weight
KV checks compare the full vocabulary logits at selected cache positions;
free-generation quality is checked separately from timing replay. With a small
model, use simple English questions and inspect relevance, fluency, corruption
and repetition. Compare against PyTorch at matching floating precision when
investigating differences; a fluent answer can still contain factual errors.
Long tests are explicitly selected, and missing real-model assets are reported as skipped.
See the [portable CPU workflow](./.github/workflows/operator-accuracy.yml) for CI.

## Repository layout

```text
Lumen/
├── Cargo.toml
├── Cargo.lock
├── CHANGELOG.md
├── build.rs
├── .github/
│   └── workflows/
│       └── operator-accuracy.yml
├── src/
│   ├── lib.rs
│   ├── main.rs
│   ├── bin/
│   │   └── quantize_safetensors.rs
│   ├── autograd/
│   │   ├── mod.rs
│   │   ├── backward.rs
│   │   ├── checkpoint.rs
│   │   ├── creation.rs
│   │   ├── device.rs
│   │   ├── diagnostics.rs
│   │   ├── operations.rs
│   │   ├── optimizer.rs
│   │   ├── quantization.rs
│   │   ├── storage.rs
│   │   └── views.rs
│   ├── arch.rs
│   ├── cpu/
│   │   ├── mod.rs
│   │   └── profiling.rs
│   ├── precision.rs
│   ├── module.rs
│   ├── init.rs
│   ├── loader.rs
│   ├── tokenizer.rs
│   ├── loss.rs
│   ├── optim/
│   │   ├── mod.rs
│   │   ├── master.rs
│   │   ├── lowp_cpu.rs
│   │   └── lowp_cuda.rs
│   ├── ops/
│   │   ├── arithmetic.rs
│   │   ├── shape.rs
│   │   ├── convolution.rs
│   │   ├── fused.rs
│   │   ├── matmul/
│   │   │   ├── mod.rs
│   │   │   ├── argmax.rs
│   │   │   ├── backward.rs
│   │   │   ├── conversion.rs
│   │   │   ├── cuda_route.rs
│   │   │   ├── dot.rs
│   │   │   ├── matvec.rs
│   │   │   ├── policy.rs
│   │   │   ├── prefill.rs
│   │   │   ├── projection.rs
│   │   │   ├── rows.rs
│   │   │   └── tensor.rs
│   │   ├── cpu_gemm.rs
│   │   ├── fp_kernels.rs
│   │   ├── int8_kernels.rs
│   │   └── cuda/
│   │       ├── mod.rs
│   │       ├── stub.rs
│   │       ├── generic.rs
│   │       ├── api/
│   │       ├── imp/
│   │       │   ├── mod.rs
│   │       │   └── ffi.rs
│   │       ├── lumen_cuda.cu
│   │       └── modules/
│   ├── layers/
│   │   ├── basic/
│   │   ├── conv/
│   │   ├── rnn/
│   │   ├── norm/
│   │   │   ├── layer_norm.rs
│   │   │   └── rms_norm.rs
│   │   └── attention/
│   │       ├── scaled_dot_product.rs
│   │       ├── self_attention.rs
│   │       ├── encoding.rs
│   │       └── cpu_infer.rs
│   └── models/
│       └── llama.rs
├── tests/
│   ├── README.md
│   ├── unit/
│   ├── validation/
│   └── profiling/
└── dev-tools/
    ├── README.md
    ├── bin/
    │   └── manual_state_check.rs
    ├── benchmarks/
    │   ├── kernel_bench.rs
    │   ├── generic_prefill_bench.rs
    │   ├── cpu_backward_bench.rs
    │   ├── cpu_projection_bench.rs
    │   ├── cpu_decode_bench.rs
    │   ├── cpu_scaling_bench.rs
    │   ├── prefill_decode_bench/
    │   │   ├── main.rs
    │   │   ├── cpu_tuning.rs
    │   │   ├── replay_audit.rs
    │   │   └── samples.rs
    │   └── cuda_cpu_bench/
    │       ├── main.rs
    │       └── paths.rs
    ├── check_operator_accuracy.ps1
    ├── check_manual_state.ps1
    ├── check_endurance.ps1
    ├── check_release.ps1
    ├── check_performance.py
    ├── common.ps1
    ├── check_operator_accuracy.py
    ├── check_dense_cross_entropy.py
    ├── check_autograd_reuse.py
    ├── check_gradient_flags.py
    ├── check_gradient_accumulation.py
    ├── check_generic_nn.py
    ├── check_generic_prefill.py
    ├── check_row_matmul.py
    ├── check_manual_state.py
    └── check_endurance.py
```

`src/` contains library and production CLI implementation. `cpu/mod.rs` owns
CPU scheduling and execution contexts; `cpu/profiling.rs` is its optional
instrumentation submodule. CPU math kernels remain under `ops/`.

`autograd/` separates Tensor construction, storage, quantization, views, device
transfers, gradient execution, and optimizer updates. `ops/matmul/` groups dot
products, MatVec, fused projections, prefill, backward, and CUDA routing.
`ops/cuda/api/` contains operator-facing wrappers; the private `imp/` directory
contains validation, allocation/dispatch, and isolated FFI declarations.
Public Tensor and operator paths remain available through their module entries.

[`tests/`](./tests/README.md) groups unit tests, cross-module validation, and
explicit profiling tests. Private unit tests are attached to their original
modules with `#[cfg(test)]` and `#[path]`, without widening the public API.
[`dev-tools/`](./dev-tools/README.md) contains developer binaries, benchmark
helpers, check runners, and independent Python references. Cargo target names
and `cargo run --bin ...` commands are unchanged. Generate API documentation
with `cargo doc --open`.

## Limitations

- Built-in checkpoint inference targets the configured Llama architecture;
  the CLI is not an adapter for arbitrary model families.
- CUDA targets a single device, and operator/dtype coverage varies by backend.
- Release validation covers Windows/x86-64 CPU and a CUDA GPU. ARM64 and the
  nightly-only AVX-512 FP16 path require separate platform validation.
- Low-precision reduction and argmax rounding can change generated token choices.
  Model knowledge and instruction following depend on the loaded weights.
- The library provides neural-network components and a training core, rather
  than a distributed trainer or serving system.

## License

See [LICENSE](./LICENSE).
