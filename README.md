# Lumen

> A compact Rust-first neural-network library for studying and building dtype-aware CPU/CUDA execution, dynamic autograd, reusable layers, and Llama-family inference.

[中文说明](./README_zh-CN.md)

---

## What this project is

Lumen is a **small Rust-first neural-network library and deep-learning core**. It keeps several reusable layers of a modern ML stack in one repository:

- a Tensor core with dynamic autograd;
- reusable layers, modules, losses, and optimizers;
- a Llama-style decoder implementation as a built-in model family;
- safetensors loading with optional streamed loading;
- runtime dtype control for parameters, activations, and KV cache;
- optional on-load and offline `i8` quantization;
- CPU execution paths with x86/ARM kernel work;
- optional CUDA acceleration through CUDA C++ kernels and NVIDIA libraries;
- benchmark tools for CPU kernels, CUDA kernels, training-path checks, and end-to-end Llama prefill/decode tests.

The project is best understood as a **learning- and experimentation-oriented neural-network library**. A compact Llama-family implementation is maintained as an important end-to-end model path, but it is not the project's only identity.

It is not intended to be a production serving system, a full training framework, or a plug-and-play launcher for arbitrary checkpoints.

Lumen is **Rust-first, not Rust-only**. Rust owns the high-level runtime, tensor/autograd system, model code, loader, tokenizer wrapper, dtype policy, CPU backend, and benchmark tools. The optional CUDA backend uses CUDA C++ kernels, cuBLAS/cuDNN, and an FFI boundary to accelerate selected paths.

---

## Status at a glance

| Area | Current status |
|---|---|
| Autograd and training | Dynamic autograd, F32 gradients, SGD/SGD-momentum/Adam, CPU and growing CUDA training paths |
| DTypes | F32, F16, BF16, and I8 storage/runtime paths; I8 uses explicit quantization scales |
| CPU kernels | Portable fallback plus opt-in x86 and ARM low-precision kernels |
| CUDA | CUDA-resident tensors, custom kernels, cuBLAS, optional cuDNN, forward and selected backward/training paths |
| Model support | Built-in Llama-family decoder path with RoPE, GQA, RMSNorm, SwiGLU-style MLP, and KV cache |
| Validation | Accuracy checks, F32-gradient checks, SGD loss-trend checks, kernel benchmarks, and real-model text checks |

### Precision contract

- Parameters and activations may remain in F32, F16, BF16, or I8 storage when a supported native path is available.
- Backward gradients are represented and accumulated in **F32**, including when forward data is low precision.
- Same-dtype low-precision kernels read low-precision storage directly. Some CPU paths accumulate into wider lanes, such as BF16/F16 into F32 or I8 into I32, without first materializing the full input as F32.
- I8 is quantized computation, not a floating-point training dtype. Its scale must be finite and positive.
- Supported native paths are tested against scalar, quantized, or CPU references; tolerances depend on dtype and reduction order.

---

## Current focus

The current codebase focuses on general tensor/autograd/layer behavior, dtype-aware CPU execution, and actively improving CUDA paths. Llama-family support remains an important built-in model path.

Important pieces include:

- dynamic autograd and general Tensor ops;
- reusable neural-network layers and sequence-modeling components, including Llama decoder pieces such as RMSNorm, RoPE, GQA, SwiGLU-style MLP, and KV-cache decode;
- F32, F16, BF16, and I8 storage, loading, and runtime configurations, with F32 gradients;
- optional CUDA execution behind the `cuda` feature;
- CUDA-resident tensors, KV-cache updates, decode-oriented kernels, forward paths, and a growing backward/training path;
- x86 backend variants such as AVX-512 BF16, AVX2/F16C, and AVX-512BW/AVX2 I8 kernels;
- optional parameter dtype copies for mixed-precision execution;
- optional streamed weight loading to reduce peak memory usage;
- development-only benchmark binaries for CPU/CUDA tuning and end-to-end inference measurements.

---

## Design overview

```text
Rust side
  ├─ Tensor representation and dynamic autograd graph
  ├─ Layers, modules, losses, and optimizers
  ├─ Model implementations, including Llama-family support
  ├─ dtype / precision / quantization policy
  ├─ safetensors loading and tokenizer integration
  ├─ CPU kernels and backend dispatch
  └─ FFI wrappers for CUDA calls

CUDA side
  ├─ device memory allocation and reuse
  ├─ custom CUDA kernels
  ├─ cuBLAS-backed matrix operations
  ├─ optional cuDNN-backed primitives
  ├─ KV-cache and decode-oriented kernels
  └─ selected forward/backward/training kernels
```

Rust manages the framework structure, type organization, runtime policy, and safety boundary. CUDA is used where direct GPU execution is more appropriate.

---

## Repository layout

```text
src/
├─ autograd.rs              # Tensor + dynamic autograd core
├─ module.rs                # Module trait / macros
├─ loader.rs                # Safetensors loading and streamed loading
├─ tokenizer.rs             # Tokenizer wrapper
├─ precision.rs             # DType / runtime precision configuration
├─ ops/                     # Tensor ops, CPU kernels, optional CUDA ops
│  └─ cuda/                 # CUDA/cuDNN/cuBLAS-backed kernels and modules
├─ layers/                  # Neural-network layers and attention building blocks
├─ models/llama.rs          # Llama model implementation
├─ main.rs                  # Minimal local inference CLI
└─ bin/
   ├─ quantize_safetensors.rs  # Offline quantization utility
   ├─ kernel_bench.rs          # Development CPU kernel benchmark
   ├─ prefill_decode_bench.rs  # End-to-end prefill/decode benchmark
   ├─ cuda_cpu_bench.rs        # CPU/CUDA ops, NN, backward, and path benchmark
   └─ cuda_cpu_bench_path.rs   # Path checks used by cuda_cpu_bench
```

---

## Build

CPU-only release build:

```bash
cargo build --release
```

Use native CPU codegen when collecting local performance numbers:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

PowerShell:

```powershell
$env:RUSTFLAGS = "-C target-cpu=native"
cargo build --release
```

CUDA build:

```bash
cargo build --release --features cuda
```

Development benchmark builds:

```bash
cargo build --release --features dev-tools --bin kernel_bench
cargo build --release --features dev-tools --bin prefill_decode_bench
cargo build --release --features dev-tools --bin cuda_cpu_bench
```

For x86 performance tests, enable the x86 backend features explicitly:

```bash
cargo build --release --features "dev-tools x86-fp-kernels x86-int8-kernels" --bin kernel_bench
cargo build --release --features "dev-tools cuda x86-fp-kernels x86-int8-kernels" --bin prefill_decode_bench
cargo build --release --features "dev-tools cuda x86-fp-kernels x86-int8-kernels" --bin cuda_cpu_bench
```

`x86-fp-kernels` builds on stable Rust and includes the stable AVX2/F16C, AVX-512F, and AVX-512 BF16 paths. True AVX-512 FP16 same-dtype compute depends on nightly Rust's `stdarch_x86_avx512_f16`; use:

```bash
cargo +nightly build --release --features "dev-tools x86-fp-kernels-nightly x86-int8-kernels" --bin kernel_bench
```

The CUDA build script searches for CUDA/cuDNN through environment variables, `nvcc`, and common platform install locations. CPU-only builds do not require CUDA.

### Feature guide

| Feature | Purpose |
|---|---|
| `cuda` | Build the optional CUDA C++ backend |
| `dev-tools` | Build benchmark and path-check binaries |
| `x86-fp-kernels` | Enable stable x86 BF16/F16/F32 fast paths |
| `x86-fp-kernels-nightly` | Enable true AVX-512 FP16 intrinsics in addition to `x86-fp-kernels` |
| `x86-int8-kernels` | Enable x86 I8 fast paths |
| `arm64-fp-kernels` | Enable ARM64 floating-point/low-precision fast paths |
| `arm64-int8-kernels` | Enable ARM64 I8 fast paths |

Architecture-specific CPU features are opt-in. Enable them when evaluating CPU or CUDA performance so CPU fallback and helper work are not measured through the portable backend.

---

## Minimal inference CLI

```bash
cargo run --release --bin lumen -- \
  --weights path/to/model.safetensors \
  --tokenizer path/to/tokenizer.json
```

Useful flags:

- `--system TEXT`
- `--temperature FLOAT`
- `--top-p FLOAT`
- `--repetition-penalty FLOAT`
- `--recent-window N`
- `--max-gen N`
- `--parameter-dtype f32|f16|bf16|i8`
- `--runtime-dtype f32|f16|bf16`
- `--activation-dtype f32|f16|bf16|i8`
- `--kv-cache-dtype f32|f16|bf16`
- `--quantize off|i8`
- `--quant-scale FLOAT`
- `--allow-parameter-copies`
- `--stream-weights`
- `--max-seq-len N`
- `--load-only`
- `--device cpu|cuda`

BF16 example:

```bash
cargo run --release --bin lumen -- \
  --weights path/to/model.safetensors \
  --tokenizer path/to/tokenizer.json \
  --parameter-dtype bf16 \
  --runtime-dtype bf16 \
  --activation-dtype bf16 \
  --kv-cache-dtype bf16 \
  --allow-parameter-copies
```

I8 weights + BF16 runtime example:

```bash
cargo run --release --bin lumen -- \
  --weights path/to/model.safetensors \
  --tokenizer path/to/tokenizer.json \
  --parameter-dtype i8 \
  --runtime-dtype bf16 \
  --activation-dtype bf16 \
  --kv-cache-dtype bf16 \
  --quantize i8 \
  --allow-parameter-copies
```

Backend diagnostics can be printed with:

```bash
LUMEN_SHOW_BACKENDS=1 cargo run --release --bin lumen -- \
  --weights path/to/model.safetensors \
  --tokenizer path/to/tokenizer.json
```

Interactive commands:

- `/reset` clears chat history and KV cache;
- `/exit` quits the process.

---

## Offline quantization

Generate an `i8` safetensors checkpoint:

```bash
cargo run --release --bin quantize_safetensors -- \
  --input path/to/model.safetensors \
  --output path/to/model.i8.safetensors \
  --dtype i8
```

Optional manual scale:

```bash
cargo run --release --bin quantize_safetensors -- \
  --input path/to/model.safetensors \
  --output path/to/model.i8.safetensors \
  --dtype i8 \
  --scale 0.02
```

---

## Benchmark tools

Use `--release` for performance numbers. Debug builds are useful for correctness work but are not representative for speed.

### Kernel benchmark

```bash
cargo run --release --features "dev-tools x86-fp-kernels x86-int8-kernels" --bin kernel_bench -- \
  --iters 400 --samples 8 --hidden 2048 --inter 5632 --vocab 32000
```

The `dot_bf16_bf16`, `dot2_bf16_bf16`, `dot3_bf16_bf16`, `dot_f16_f16`, `dot2_f16_f16`, `dot3_f16_f16`, `dot2_i8_i8`, and `dot3_i8_i8` rows are same-dtype low-precision dot microbenchmarks for the bottom-level kernels. BF16 prefers `_mm256_dpbf16_ps` when AVX-512 BF16 is available; otherwise stable `x86-fp-kernels` uses AVX2/FMA to read BF16 storage directly and accumulate in F32 lanes. Stable F16 uses AVX2/F16C to read F16 storage directly and accumulate in F32 lanes; nightly `x86-fp-kernels-nightly` prefers the true `_mm512_*_ph` FP16 kernels when AVX-512 FP16 is available at runtime. I8 same-dtype prefers AVX-512BW; otherwise stable `x86-int8-kernels` uses AVX2 to sign-extend I8 storage to I16, accumulate through `_mm256_madd_epi16` into I32, then apply scales.

To benchmark true AVX-512 FP16 kernels such as `_mm512_loadu_ph`, `_mm512_storeu_ph`, and `_mm512_reduce_add_ph`, use the nightly feature:

```bash
cargo +nightly run --release --features "dev-tools x86-fp-kernels-nightly x86-int8-kernels" --bin kernel_bench -- \
  --iters 400 --samples 8 --hidden 2048 --inter 5632 --vocab 32000
```

### CPU/CUDA ops and training benchmark

```bash
cargo run --release --features "dev-tools cuda x86-fp-kernels x86-int8-kernels" --bin cuda_cpu_bench -- \
  --suite all --size medium --dtype bf16 --runs 5 --warmup 2 --check
```

Useful options:

- `--suite all|ops|nn|backward|path`
- `--size small|medium|large`
- `--dtype f32|f16|bf16|i8`
- `--case TEXT` to run cases whose names contain `TEXT`
- `--check` to run CPU/CUDA correctness checks and path checks

### End-to-end prefill/decode benchmark

```bash
cargo run --release --features "dev-tools cuda x86-fp-kernels x86-int8-kernels" --bin prefill_decode_bench -- \
  --weights path/to/model.safetensors \
  --tokenizer path/to/tokenizer.json \
  --prompt "Explain Transformer KV cache." \
  --runs 3 --warmup 1 --max-gen 64 --mode greedy \
  --stop-on-eos --stop-on-chat-marker \
  --device cuda \
  --parameter-dtype bf16 \
  --runtime-dtype bf16 \
  --activation-dtype bf16 \
  --kv-cache-dtype bf16 \
  --allow-parameter-copies
```

Real-model path check:

```bash
cargo run --release --features "dev-tools cuda x86-fp-kernels x86-int8-kernels" --bin cuda_cpu_bench -- \
  --suite path --check --path-device cuda \
  --weights path/to/model.safetensors \
  --tokenizer path/to/tokenizer.json \
  --max-gen 32 --show-output
```

Path checks are not pure microbenchmarks. They are intended to catch algorithmic problems:

- the training path runs a tiny SGD-like trace and checks that loss behaves plausibly;
- the inference path can load a real Llama/TinyLlama checkpoint and check generated text for obvious corruption.

---

## Current local performance snapshot

The following numbers are a local development snapshot refreshed on 2026-06-17, not a universal benchmark claim. They were collected on a Windows machine with:

- CPU: AMD Ryzen 9 8945HX with Radeon Graphics
- GPU: NVIDIA GeForce RTX 5070 Laptop GPU, 8151 MiB VRAM
- RAM: 32 GB
- NVIDIA driver/runtime CUDA reported by `nvidia-smi`: 610.62 / CUDA 13.3
- CUDA toolkit: 13.0
- cuDNN: 9.21.1
- Rust: stable MSVC toolchain, `rustc 1.95.0`

The most recent rerun enabled both CUDA and x86 backend features:

```text
backend: float=x86-avx512 bf16_bf16=x86-avx512bf16 f16_f16=x86-avx2-f16c
         int8=x86-avx512bw i8_i8=x86-avx512bw avx512fp16=unavailable-or-stable-build
```

This is important because CPU fallback and CPU-side helper paths should not be measured with the portable backend when evaluating this machine.

### Comprehensive accuracy and training checks

Commands run for this refresh:

```bash
cargo fmt --all -- --check
cargo test --release --all-targets --features "cuda x86-fp-kernels x86-int8-kernels"
cargo test --release --lib --features "cuda x86-fp-kernels x86-int8-kernels" -- --ignored --nocapture
cargo test --release --lib --no-default-features --features "x86-fp-kernels x86-int8-kernels"
cargo clippy --release --all-targets --features "cuda x86-fp-kernels x86-int8-kernels" -- -D warnings
cargo clippy --release --lib --no-default-features --features "x86-fp-kernels x86-int8-kernels" -- -D warnings

# Run once for each of f32/f16/bf16/i8
cargo run --release --features "dev-tools cuda x86-fp-kernels x86-int8-kernels" --bin cuda_cpu_bench -- \
  --suite all --size medium --dtype DTYPE --runs 3 --warmup 1 --check

# Run once for each of f32/f16/bf16/i8
cargo run --release --features "dev-tools cuda x86-fp-kernels x86-int8-kernels" --bin cuda_cpu_bench -- \
  --suite path --dtype DTYPE --check

# Run for CPU and CUDA across f32/f16/bf16/i8+bf16
cargo run --release --features "dev-tools cuda x86-fp-kernels x86-int8-kernels" --bin prefill_decode_bench -- \
  --weights C:\Users\chen-\Downloads\model.safetensors \
  --tokenizer C:\Users\chen-\Downloads\tokenizer.json \
  --device DEVICE --parameter-dtype PARAM_DTYPE --runtime-dtype RUNTIME_DTYPE \
  --activation-dtype ACTIVATION_DTYPE --kv-cache-dtype KV_DTYPE --quantize QUANTIZE \
  --max-gen 24 --max-seq-len 256 --runs 1 --warmup 0 --mode greedy \
  --stop-on-eos --stop-on-chat-marker --system "You are a concise assistant." \
  --prompt "Write one short sentence about neural networks." --show-output
```

Results:

- CUDA all-target regression: library `441 passed; 0 failed; 9 ignored`, empty main binary test target passed, quantization tool `4 passed; 0 failed`;
- all 9 explicit performance smoke tests: `9 passed; 0 failed`;
- CPU-only regression: `251 passed; 0 failed; 6 ignored`;
- CUDA and CPU clippy with `-D warnings`: passed;
- TinyLlama real-model inference passed for CPU and CUDA across F32, F16, BF16, and I8 weights + BF16 runtime, with `replacement=0`, `trailing_replacement=0`, and `control=0` in every generated sample;
- CPU/CUDA forward, backward, F32-gradient, optimizer, compact Llama F32/F16/BF16 training checks, and I8 parameter-training checks passed;
- native I8 Adam state and a fully I8 Llama runtime are intentionally skipped; I8 parameters with F32 optimizer state and I8 + BF16 runtime path checks passed.

Representative same-dtype CPU/CUDA differences from `cuda_cpu_bench --suite all --check`:

| Check | F32 max abs | F16 max abs | BF16 max abs | I8 max abs |
|---|---:|---:|---:|---:|
| matrix matmul forward | `3.815e-6` | `7.629e-6` | `3.008e-2` | `1.144e-5` |
| matrix matmul lhs grad | `2.384e-7` | `2.384e-7` | `3.576e-7` | `2.384e-7` |
| matrix matmul rhs grad | `4.768e-7` | `4.768e-7` | `4.768e-7` | `4.768e-7` |

BF16 forward error is visibly larger than F16 because of BF16 mantissa precision and reduction-order differences; gradients remain F32. The I8 forward numbers are quantized-path checks, not proof of floating-equivalent arithmetic.

The training-path check now covers 23 CPU/CUDA paths for each dtype: scalar SGD, MLP, GELU MLP, Dropout MLP, BatchMatMul, gated MLP, residual/gated branch mixing, row-broadcast affine parameters, Embedding classifier, shape/view chain, shared/tied parameter reuse, gradient accumulation, SGD batched optimizer update, RMSNorm, RNN, GRU, LSTM, Adam, Adam batched optimizer update, Conv2D, Conv2D+MaxPool, SelfAttention, and a compact Transformer block. I8 attention and Transformer block checks use I8 parameters with BF16 runtime/KV-cache data.

Observed loss trend for the 24-step SGD + momentum path:

| DType | CPU first -> last | CUDA first -> last | Increases during trace |
|---|---:|---:|---:|
| F32 | `9.0 -> 1e-6` | `9.0 -> 1e-6` | 7 |
| F16 | `9.0 -> 2e-6` | `9.0 -> 2e-6` | 6 |
| BF16 | `9.0 -> 0` | `9.0 -> 0` | 3 |
| I8 | `9.0 -> 4e-4` | `9.0 -> 4e-4` | 3 |

These traces are not monotonic, but all have a clear downward trend, which is the intended SGD-path criterion.

Latest 2026-06-17 timing for the tiny 24-step SGD + momentum check:

| DType | CPU us/step | CUDA us/step | Notes |
|---|---:|---:|---|
| F32 | 93.35 | 2405.87 | CUDA gradients and momentum state remained F32 |
| F16 | 114.42 | 2399.96 | low-precision parameters, F32 gradients |
| BF16 | 136.10 | 2577.54 | low-precision parameters, F32 gradients |
| I8 | 86.48 | 2515.04 | quantized parameters, F32 gradients |

Batched optimizer path timing snapshot. Both CUDA paths assert that pointer-batched optimizer kernels were used, while gradients and optimizer state remain F32:

| Path | DType | CPU us/step | CUDA us/step | Loss first -> last |
|---|---:|---:|---:|---:|
| SGD momentum batched update | F32 | 719.18 | 2081.82 | `0.042612 -> 0.035707` |
| SGD momentum batched update | F16 | 3292.39 | 1805.36 | `0.042624 -> 0.035717` |
| SGD momentum batched update | BF16 | 2774.70 | 2048.36 | `0.042695 -> 0.035773` |
| SGD momentum batched update | I8 | 2388.58 | 2838.08 | `0.042661 -> 0.037453` |
| Adam batched update | F32 | 980.41 | 1301.71 | `0.032873 -> 0.000897` |
| Adam batched update | F16 | 3457.87 | 1466.73 | `0.032875 -> 0.000898` |
| Adam batched update | BF16 | 2910.04 | 1728.02 | `0.032754 -> 0.000915` |
| Adam batched update | I8 | 2559.43 | 3499.50 | `0.032910 -> 0.017634` |

These are small end-to-end training graphs with eight Linear shards, so CUDA is still often launch- and dispatch-bound outside the optimizer update itself.

Latest 2026-06-17 performance-smoke highlights:

| Case | Result |
|---|---:|
| BF16 same-dtype dot/dot2/dot3 backend | `x86-avx512bf16` |
| BF16 dot / dot2 / dot3 | 0.308 / 0.450 / 0.630 us |
| F16 same-dtype dot/dot2/dot3 backend | `x86-avx2-f16c` |
| F16 dot / dot2 / dot3 | 0.393 / 0.441 / 0.460 us |
| I8 same-dtype dot2 / dot3 backend | `x86-avx512bw` |
| I8 dot2 / dot3 | 0.347 / 0.351 us |
| CPU batch matmul backward, BF16xBF16 / I8xI8 / BF16xI8 / F32xI8 | 2335.0 / 400.9 / 476.1 / 808.2 us |
| CUDA dynamic I8 quantize, 1M elements | 87.1 us |
| CUDA lowp sum, 1M elements F32 / F16 / BF16 / I8 | 240.3 / 162.0 / 134.2 / 140.4 us |
| CUDA I8xI8 matmul, F32 out | 22.4 us, `kernel_err=0.00002` |
| CUDA I8xI8 matmul, typed I8 out | 117.9 us, `quant_err=1.00382` |
| CUDA I8xI8 batch matmul, F32 / typed I8 out | 14.3 / 86.4 us |

### CPU kernel snapshot

Command:

```bash
cargo run --release --features "dev-tools x86-fp-kernels x86-int8-kernels" --bin kernel_bench -- \
  --iters 300 --samples 7 --hidden 2048 --inter 5632 --vocab 32000
```

Active backends:

```text
float=x86-avx512  bf16_bf16=x86-avx512bf16  f16_f16=x86-avx2-f16c
int8=x86-avx512bw  i8_i8=x86-avx512bw  avx512fp16=unavailable-or-stable-build
```

| Case | Reference | Fast path | Speedup / ratio | Max abs diff |
|---|---:|---:|---:|---:|
| `dot_bf16_bf16` | 1.81 us | 0.06 us | 29.69x | `7.02e-4` |
| `dot2_bf16_bf16` | 3.18 us | 0.08 us | 39.80x | `7.02e-4` |
| `dot3_bf16_bf16` | 4.41 us | 0.09 us | 47.98x | `7.02e-4` |
| `dot_f16_f16` | 4.79 us | 0.10 us | 47.94x | `1.114e-3` |
| `dot2_f16_f16` | 7.40 us | 0.11 us | 65.45x | `1.114e-3` |
| `dot3_f16_f16` | 9.50 us | 0.12 us | 77.22x | `1.114e-3` |
| `dot2_i8_i8` | 0.56 us | 0.07 us | 7.70x | exact I32 accumulation |
| `dot3_i8_i8` | 1.17 us | 0.12 us | 9.45x | exact I32 accumulation |
| `tensor_matmul_i8` | 481.46 us | 73.68 us | 6.53x | quantized reference |
| `fused_qkv_i8` | 589.79 us | 80.27 us | 0.14x time | `0` |
| `fused_gate_i8` | 1521.28 us | 132.94 us | 0.09x time | `0` |
| `sgd_bf16` | 2.53 us | 2.29 us | 0.90x time | `0` |
| `adam_bf16` | 15.18 us | 11.65 us | 0.77x time | `0` |
| `adam_i8` | 15.18 us | 15.75 us | 1.04x time | `0` |

The true AVX-512 FP16 path still requires nightly Rust and `x86-fp-kernels-nightly`. Cached copies are path-dependent: cached BF16/F16 tensor matmul helped in this run, while cached fused F16 QKV/GateUp was substantially slower than no-copy.

### Detailed CPU/CUDA operator snapshot

Command family:

```bash
# Run once for each of f32/f16/bf16/i8
cargo run --release --features "dev-tools cuda x86-fp-kernels x86-int8-kernels" --bin cuda_cpu_bench -- \
  --suite all --size medium --dtype DTYPE --runs 3 --warmup 1 --check
```

Each cell is `CPU ms / CUDA ms / CUDA speedup`. These are small-to-medium development shapes, not hardware peak-throughput claims.

| Operator | F32 | F16 | BF16 | I8 |
|---|---:|---:|---:|---:|
| `matmul.forward` | 2.327 / 0.098 / 23.67x | 1.113 / 0.034 / 32.56x | 0.642 / 0.068 / 9.45x | 2.947 / 0.124 / 23.67x |
| `batch_matmul.forward` | 0.009 / 0.021 / 0.43x | 0.054 / 0.013 / 4.20x | 0.017 / 0.016 / 1.10x | 0.155 / 0.075 / 2.08x |
| `matmul.backward` | 8.131 / 0.647 / 12.56x | 26.064 / 0.787 / 33.13x | 5.270 / 0.614 / 8.59x | 4.551 / 0.769 / 5.92x |
| `elementwise.mul_add.forward` | 0.431 / 0.160 / 2.69x | 0.220 / 0.189 / 1.16x | 0.347 / 0.158 / 2.20x | 0.455 / 0.324 / 1.40x |
| `elementwise.mul_add.backward` | 3.260 / 1.188 / 2.74x | 3.399 / 1.308 / 2.60x | 3.596 / 1.170 / 3.07x | 3.092 / 1.394 / 2.22x |
| `binary.row_broadcast.forward` | 1.342 / 0.077 / 17.38x | 0.110 / 0.110 / 1.00x | 0.165 / 0.087 / 1.90x | 0.215 / 0.168 / 1.28x |
| `elementwise.mixed_mul.backward` | 2.461 / 0.684 / 3.60x | 3.432 / 0.683 / 5.02x | 2.438 / 0.917 / 2.66x | 2.026 / 0.833 / 2.43x |
| `unary.silu.forward` | 0.310 / 0.097 / 3.20x | 3.074 / 0.081 / 38.00x | 1.107 / 0.088 / 12.51x | 12.138 / 0.181 / 66.98x |
| `unary.silu.backward` | 4.674 / 1.262 / 3.71x | 6.126 / 1.385 / 4.42x | 5.582 / 1.534 / 3.64x | 4.882 / 1.197 / 4.08x |
| `fused_gateup.forward` | 2.716 / 0.062 / 43.74x | 1.818 / 0.115 / 15.77x | 3.168 / 0.134 / 23.67x | 3.588 / 0.137 / 26.23x |
| `fused_qkv.prefill` | 1.031 / 0.090 / 11.45x | 1.148 / 0.098 / 11.68x | 1.041 / 0.083 / 12.51x | 1.058 / 0.318 / 3.33x |
| `softmax.forward` | 0.526 / 0.088 / 5.99x | 3.706 / 0.163 / 22.67x | 1.236 / 0.165 / 7.49x | 6.165 / 0.160 / 38.43x |
| `fused_softmax.forward` | 2.215 / 0.081 / 27.51x | 6.282 / 0.121 / 51.70x | 3.091 / 0.085 / 36.28x | 4.754 / 0.083 / 57.00x |
| `cross_entropy.forward` | 0.503 / 0.076 / 6.61x | 1.735 / 0.239 / 7.26x | 1.346 / 0.195 / 6.90x | 9.043 / 0.252 / 35.83x |
| `mse_loss.forward` | 0.498 / 0.073 / 6.86x | 1.025 / 0.072 / 14.20x | 1.149 / 0.077 / 14.87x | 8.404 / 0.083 / 101.01x |
| `optimizer.sgd.step` | 0.167 / 0.090 / 1.85x | 0.141 / 0.119 / 1.19x | 0.186 / 0.139 / 1.34x | 0.138 / 0.140 / 0.98x |
| `optimizer.adam_f32_state.step` | 0.849 / 0.249 / 3.41x | 0.370 / 0.273 / 1.35x | 0.541 / 0.252 / 2.15x | 0.319 / 0.261 / 1.22x |
| `conv2d.forward` | 1.033 / 0.664 / 1.56x | 0.831 / 0.246 / 3.37x | 1.020 / 0.232 / 4.39x | 0.810 / 0.255 / 3.18x |
| `conv2d.backward` | 3.073 / 1.745 / 1.76x | 3.099 / 1.238 / 2.50x | 3.203 / 1.760 / 1.82x | 2.581 / 1.638 / 1.58x |
| `self_attention.forward` | 0.337 / 0.280 / 1.20x | 0.853 / 0.528 / 1.61x | 0.771 / 0.482 / 1.60x | skipped |
| `llama.prefill_decode` | 2.346 / 0.926 / 2.53x | 2.234 / 1.213 / 1.84x | 2.055 / 1.050 / 1.96x | skipped |
| `llama.train.step` | 3.592 / 2.555 / 1.41x | 3.887 / 2.983 / 1.30x | 4.477 / 3.042 / 1.47x | skipped |

All enabled correctness checks passed. CUDA is strongest on dense, fused, softmax, loss, and larger broadcast work. Single-token QKV decode, tiny broadcast reductions, batched optimizer cases, and some compact attention/training cases remain launch- or dispatch-bound.

### End-to-end Llama prefill/decode snapshot

The TinyLlama rerun used the local `tokenizer.json` and `model.safetensors`, `prompt_tokens=42`, `max_gen=24`, greedy decode, 1 measured run, 0 warmup, and `--stop-on-eos --stop-on-chat-marker`.

| Configuration | Device | Prefill forward | Decode forward | End-to-end decode | Total |
|---|---|---:|---:|---:|---:|
| F32 | CPU | 38.70 tok/s | 10.41 tok/s | 5.94 tok/s | 2526.79 ms |
| F16 | CPU | 126.20 tok/s | 16.54 tok/s | 11.96 tok/s | 1253.83 ms |
| BF16 | CPU | 134.25 tok/s | 16.75 tok/s | 12.40 tok/s | 1209.48 ms |
| I8 weights + BF16 runtime | CPU | 198.61 tok/s | 25.83 tok/s | 18.91 tok/s | 793.35 ms |
| F32 | CUDA | 259.89 tok/s | 42.01 tok/s | 28.87 tok/s | 519.61 ms |
| F16 | CUDA | 140.60 tok/s | 27.88 tok/s | 17.91 tok/s | 837.63 ms |
| BF16 | CUDA | 159.69 tok/s | 27.53 tok/s | 18.55 tok/s | 808.65 ms |
| I8 weights + BF16 runtime | CUDA | 207.70 tok/s | 67.52 tok/s | 35.28 tok/s | 425.18 ms |

All eight generated samples were fluent for the short prompt and reported `replacement=0`, `trailing_replacement=0`, and `control=0`. Every configuration generated: "Neural networks are a powerful tool for analyzing and understanding complex data." The rerun shows real generation remains decode-bound: decode-forward dominates measured time, while I8 weights with BF16 runtime/activation/KV cache have the best measured end-to-end decode throughput in this short real-model run.

---

## Design limitations

`src/main.rs` and some benchmark paths intentionally use a hard-coded `model_config()`. The current Llama runtime is compact and inspectable, but this also means:

- the loaded checkpoint must match the expected architecture;
- adapting to a different model may require editing hidden size, layer count, KV-head layout, tokenizer behavior, or prompt formatting;
- the CLI is a local runner, not a universal inference frontend.

Benchmark tools are development tools for kernel/runtime tuning. They are useful for regression tracking and local comparison, but they are not a polished public benchmarking suite.

CUDA support is functional but still evolving. Not every operation or dtype combination has an equally optimized CUDA path, and small workloads may be slower than CPU because of launch and dispatch overhead. The practical target today is a single CUDA device. Future multi-GPU work would require explicit device-index plumbing through tensors, modules, and CUDA calls.

---

## Who this project is for

Lumen is a good fit if you want to:

- study how a Rust tensor/autograd core can be structured;
- inspect a small Llama runtime without a large framework around it;
- experiment with dtype policy, quantization, CPU kernels, and CUDA kernels;
- benchmark and tune a compact Rust inference stack on your own machine;
- explore how Rust high-level runtime code can cooperate with CUDA low-level kernels.

It is probably not the right fit if you need:

- large-scale distributed training;
- a mature serving system;
- mature multi-GPU deployment tooling;
- plug-and-play support for arbitrary model families.

---

## License

This repository is released under the license included in [`LICENSE`](./LICENSE).
