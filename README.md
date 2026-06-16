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

The following numbers are a local development snapshot refreshed on 2026-06-16, not a universal benchmark claim. They were collected on a Windows machine with:

- CPU: AMD Ryzen 9 8945HX with Radeon Graphics
- GPU: NVIDIA GeForce RTX 5070 Laptop GPU, 8 GB VRAM
- RAM: 32 GB
- NVIDIA driver/runtime CUDA reported by `nvidia-smi`: 610.47 / CUDA 13.3
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
  --suite path --check --case path.train --dtype DTYPE --runs 1 --warmup 0

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

- CUDA all-target regression: library `440 passed; 0 failed; 9 ignored`, empty main binary test target passed, quantization tool `4 passed; 0 failed`;
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

The training-path check now covers 21 CPU/CUDA paths for each dtype: scalar SGD, MLP, GELU MLP, Dropout MLP, BatchMatMul, gated MLP, Embedding classifier, shape/view chain, shared/tied parameter reuse, gradient accumulation, SGD batched optimizer update, RMSNorm, RNN, GRU, LSTM, Adam, Adam batched optimizer update, Conv2D, Conv2D+MaxPool, SelfAttention, and a compact Transformer block. I8 attention and Transformer block checks use I8 parameters with BF16 runtime/KV-cache data.

Observed loss trend for the 24-step SGD + momentum path:

| DType | CPU first -> last | CUDA first -> last | Increases during trace |
|---|---:|---:|---:|
| F32 | `9.0 -> 1e-6` | `9.0 -> 1e-6` | 7 |
| F16 | `9.0 -> 2e-6` | `9.0 -> 2e-6` | 6 |
| BF16 | `9.0 -> 0` | `9.0 -> 0` | 3 |
| I8 | `9.0 -> 4e-4` | `9.0 -> 4e-4` | 3 |

These traces are not monotonic, but all have a clear downward trend, which is the intended SGD-path criterion.

Latest 2026-06-16 timing for the tiny 24-step SGD + momentum check:

| DType | CPU us/step | CUDA us/step | Notes |
|---|---:|---:|---|
| F32 | 51.72 | 2197.00 | CUDA gradients and momentum state remained F32 |
| F16 | 79.85 | 2778.32 | low-precision parameters, F32 gradients |
| BF16 | 51.07 | 2282.88 | low-precision parameters, F32 gradients |
| I8 | 134.49 | 2433.09 | quantized parameters, F32 gradients |

Batched optimizer path timing snapshot. Both CUDA paths assert that pointer-batched optimizer kernels were used, while gradients and optimizer state remain F32:

| Path | DType | CPU us/step | CUDA us/step | Loss first -> last |
|---|---:|---:|---:|---:|
| SGD momentum batched update | F32 | 576.59 | 2992.01 | `0.042612 -> 0.035707` |
| SGD momentum batched update | F16 | 3133.51 | 4016.52 | `0.042624 -> 0.035717` |
| SGD momentum batched update | BF16 | 2022.76 | 1878.86 | `0.042695 -> 0.035773` |
| SGD momentum batched update | I8 | 1940.68 | 4371.20 | `0.042661 -> 0.037453` |
| Adam batched update | F32 | 807.70 | 1610.01 | `0.032873 -> 0.000897` |
| Adam batched update | F16 | 2955.92 | 1607.95 | `0.032875 -> 0.000898` |
| Adam batched update | BF16 | 2394.90 | 1625.08 | `0.032754 -> 0.000915` |
| Adam batched update | I8 | 2052.58 | 3984.49 | `0.032910 -> 0.017634` |

These are small end-to-end training graphs with eight Linear shards, so CUDA is still often launch- and dispatch-bound outside the optimizer update itself.

Latest 2026-06-16 performance-smoke highlights:

| Case | Result |
|---|---:|
| BF16 same-dtype dot/dot2/dot3 backend | `x86-avx512bf16` |
| BF16 dot / dot2 / dot3 | 0.411 / 0.549 / 0.620 us |
| F16 same-dtype dot/dot2/dot3 backend | `x86-avx2-f16c` |
| F16 dot / dot2 / dot3 | 0.382 / 0.405 / 0.448 us |
| I8 same-dtype dot2 / dot3 backend | `x86-avx512bw` |
| I8 dot2 / dot3 | 0.260 / 0.332 us |
| CPU batch matmul backward, BF16xBF16 / I8xI8 / BF16xI8 / F32xI8 | 2019.7 / 391.0 / 325.7 / 274.4 us |
| CUDA dynamic I8 quantize, 1M elements | 99.0 us |
| CUDA lowp sum, 1M elements F32 / F16 / BF16 / I8 | 1374.4 / 319.9 / 303.6 / 394.3 us |
| CUDA I8xI8 matmul, F32 out | 18.5 us, `kernel_err=0.00002` |
| CUDA I8xI8 matmul, typed I8 out | 82.0 us, `quant_err=1.00382` |
| CUDA I8xI8 batch matmul, F32 / typed I8 out | 13.5 / 69.9 us |

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
| `dot_bf16_bf16` | 1.67 us | 0.06 us | 29.84x | `7.02e-4` |
| `dot2_bf16_bf16` | 2.92 us | 0.07 us | 38.87x | `7.02e-4` |
| `dot3_bf16_bf16` | 3.73 us | 0.08 us | 45.50x | `7.02e-4` |
| `dot_f16_f16` | 4.15 us | 0.09 us | 44.11x | `1.114e-3` |
| `dot2_f16_f16` | 5.79 us | 0.10 us | 55.15x | `1.114e-3` |
| `dot3_f16_f16` | 7.49 us | 0.11 us | 65.17x | `1.114e-3` |
| `dot2_i8_i8` | 0.52 us | 0.07 us | 7.72x | exact I32 accumulation |
| `dot3_i8_i8` | 0.69 us | 0.10 us | 6.69x | exact I32 accumulation |
| `tensor_matmul_i8` | 480.66 us | 74.89 us | 6.42x | quantized reference |
| `fused_qkv_i8` | 557.47 us | 74.89 us | 0.13x time | `0` |
| `fused_gate_i8` | 1499.38 us | 132.29 us | 0.09x time | `0` |
| `sgd_bf16` | 2.48 us | 2.18 us | 0.88x time | `0` |
| `adam_bf16` | 14.59 us | 11.51 us | 0.79x time | `0` |
| `adam_i8` | 14.59 us | 15.45 us | 1.06x time | `0` |

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
| `matmul.forward` | 2.094 / 0.053 / 39.66x | 1.184 / 0.032 / 36.99x | 0.644 / 0.025 / 25.96x | 1.946 / 0.097 / 20.12x |
| `batch_matmul.forward` | 0.009 / 0.014 / 0.64x | 0.041 / 0.023 / 1.74x | 0.016 / 0.012 / 1.27x | 0.133 / 0.078 / 1.71x |
| `matmul.backward` | 6.839 / 0.578 / 11.83x | 19.856 / 0.785 / 25.30x | 4.765 / 0.758 / 6.29x | 4.307 / 0.665 / 6.47x |
| `elementwise.mul_add.forward` | 0.232 / 0.160 / 1.45x | 0.166 / 0.156 / 1.06x | 0.304 / 0.148 / 2.05x | 0.392 / 0.335 / 1.17x |
| `elementwise.mul_add.backward` | 2.978 / 1.117 / 2.67x | 2.926 / 1.284 / 2.28x | 2.773 / 0.994 / 2.79x | 3.096 / 1.167 / 2.65x |
| `binary.row_broadcast.forward` | 1.259 / 0.079 / 15.95x | 0.091 / 0.076 / 1.19x | 0.150 / 0.078 / 1.93x | 0.195 / 0.153 / 1.27x |
| `elementwise.mixed_mul.backward` | 2.361 / 0.989 / 2.39x | 2.084 / 3.625 / 0.57x | 1.925 / 0.647 / 2.98x | 3.091 / 0.700 / 4.42x |
| `unary.silu.forward` | 0.285 / 0.078 / 3.68x | 2.717 / 0.075 / 36.37x | 1.077 / 0.080 / 13.46x | 10.602 / 0.202 / 52.49x |
| `unary.silu.backward` | 4.066 / 1.518 / 2.68x | 4.304 / 1.111 / 3.87x | 4.769 / 1.105 / 4.31x | 4.833 / 1.302 / 3.71x |
| `fused_gateup.forward` | 2.454 / 0.039 / 62.60x | 1.421 / 0.130 / 10.95x | 2.253 / 0.112 / 20.06x | 3.429 / 0.111 / 31.03x |
| `fused_qkv.prefill` | 0.897 / 0.078 / 11.55x | 0.918 / 0.181 / 5.07x | 0.955 / 0.085 / 11.18x | 1.043 / 0.130 / 8.00x |
| `softmax.forward` | 0.456 / 0.078 / 5.87x | 3.228 / 0.172 / 18.79x | 1.023 / 0.152 / 6.71x | 5.240 / 0.154 / 33.98x |
| `fused_softmax.forward` | 2.122 / 0.072 / 29.43x | 4.901 / 0.091 / 53.56x | 2.820 / 0.090 / 31.36x | 4.655 / 0.088 / 53.14x |
| `cross_entropy.forward` | 0.542 / 0.075 / 7.19x | 1.131 / 0.197 / 5.76x | 0.992 / 0.221 / 4.49x | 9.869 / 0.200 / 49.44x |
| `mse_loss.forward` | 0.425 / 0.115 / 3.68x | 0.827 / 0.095 / 8.75x | 0.891 / 0.071 / 12.57x | 6.447 / 0.102 / 63.02x |
| `optimizer.sgd.step` | 0.128 / 0.109 / 1.17x | 0.108 / 0.107 / 1.01x | 0.122 / 0.144 / 0.85x | 0.119 / 0.144 / 0.82x |
| `optimizer.adam_f32_state.step` | 0.781 / 0.266 / 2.94x | 0.300 / 0.257 / 1.16x | 0.287 / 0.284 / 1.01x | 0.285 / 0.224 / 1.27x |
| `conv2d.forward` | 0.750 / 0.369 / 2.03x | 0.866 / 0.218 / 3.97x | 0.835 / 0.225 / 3.71x | 0.705 / 0.223 / 3.16x |
| `conv2d.backward` | 2.440 / 1.671 / 1.46x | 3.090 / 1.319 / 2.34x | 2.515 / 1.331 / 1.89x | 2.777 / 1.623 / 1.71x |
| `self_attention.forward` | 0.371 / 0.220 / 1.69x | 0.722 / 0.807 / 0.89x | 0.554 / 0.400 / 1.38x | skipped |
| `llama.prefill_decode` | 1.719 / 1.677 / 1.02x | 2.013 / 2.109 / 0.95x | 1.562 / 1.049 / 1.49x | skipped |
| `llama.train.step` | 3.966 / 2.143 / 1.85x | 3.565 / 4.265 / 0.84x | 3.344 / 4.403 / 0.76x | skipped |

All enabled correctness checks passed. CUDA is strongest on dense, fused, softmax, loss, and larger broadcast work. Single-token QKV decode, tiny broadcast reductions, batched optimizer cases, and some compact attention/training cases remain launch- or dispatch-bound.

### End-to-end Llama prefill/decode snapshot

The TinyLlama rerun used the local `tokenizer.json` and `model.safetensors`, `prompt_tokens=42`, `max_gen=24`, greedy decode, 1 measured run, 0 warmup, and `--stop-on-eos --stop-on-chat-marker`.

| Configuration | Device | Prefill forward | Decode forward | End-to-end decode | Total |
|---|---|---:|---:|---:|---:|
| F32 | CPU | 40.30 tok/s | 11.23 tok/s | 6.31 tok/s | 2379.06 ms |
| F16 | CPU | 136.67 tok/s | 18.05 tok/s | 13.11 tok/s | 1143.85 ms |
| BF16 | CPU | 163.92 tok/s | 18.00 tok/s | 13.61 tok/s | 1101.90 ms |
| I8 weights + BF16 runtime | CPU | 198.42 tok/s | 26.33 tok/s | 19.17 tok/s | 782.40 ms |
| F32 | CUDA | 272.01 tok/s | 45.27 tok/s | 30.82 tok/s | 486.69 ms |
| F16 | CUDA | 155.84 tok/s | 29.12 tok/s | 19.10 tok/s | 785.49 ms |
| BF16 | CUDA | 165.50 tok/s | 29.24 tok/s | 19.54 tok/s | 767.56 ms |
| I8 weights + BF16 runtime | CUDA | 220.54 tok/s | 73.38 tok/s | 37.90 tok/s | 395.82 ms |

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
