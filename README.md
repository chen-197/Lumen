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
  --activation-dtype i8 \
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

The following numbers are a local development snapshot collected on 2026-06-15, not a universal benchmark claim. They were collected on a Windows machine with:

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

Commands:

```bash
cargo test --release --features "cuda,dev-tools,x86-fp-kernels,x86-int8-kernels"
cargo test --release --features "cuda,dev-tools,x86-fp-kernels,x86-int8-kernels" -- --ignored --nocapture --test-threads=1

# Run once for each of f32/f16/bf16/i8
cargo run --release --features "dev-tools cuda x86-fp-kernels x86-int8-kernels" --bin cuda_cpu_bench -- \
  --suite all --size medium --dtype DTYPE --runs 7 --warmup 3 --check
```

Results:

- all-feature regression: `425 passed; 0 failed; 9 ignored`;
- all 9 explicit performance smoke tests: `9 passed; 0 failed`;
- CPU/CUDA forward, backward, F32-gradient, optimizer, and supported Llama training checks passed for F32, F16, BF16, and I8;
- the built-in real-model path checker passed on CPU and CUDA with identical 32-token F32 output and `replacement=0`, `control=0`;
- native I8 Adam state and a fully I8 Llama runtime are intentionally skipped; I8 parameters with F32 Adam state and the standalone training path passed.

Representative same-dtype CPU/CUDA differences:

| Check | F32 max abs | F16 max abs | BF16 max abs | I8 max abs |
|---|---:|---:|---:|---:|
| matrix matmul forward | `3.815e-6` | `7.629e-6` | `3.008e-2` | `1.144e-5` |
| matrix matmul lhs grad | `2.384e-7` | `2.384e-7` | `3.576e-7` | `2.384e-7` |
| matrix matmul rhs grad | `4.768e-7` | `4.768e-7` | `4.768e-7` | `4.768e-7` |

BF16 forward error is visibly larger than F16 because of BF16 mantissa precision and reduction-order differences; gradients remain F32. The BF16 Llama training gradient check has a large relative difference around near-zero values, while its maximum absolute difference is `1.178e-3` and the check passes.

Observed loss trend for the 24-step SGD + momentum path:

| DType | CPU first -> last | CUDA first -> last | Increases during trace |
|---|---:|---:|---:|
| F32 | `9.0 -> 1e-6` | `9.0 -> 1e-6` | 7 |
| F16 | `9.0 -> 2e-6` | `9.0 -> 2e-6` | 6 |
| BF16 | `9.0 -> 0` | `9.0 -> 0` | 3 |
| I8 | `9.0 -> 4e-4` | `9.0 -> 4e-4` | 3 |

The traces are not monotonic, but all have a clear downward trend, which is the intended SGD-path criterion.

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
  --suite all --size medium --dtype DTYPE --runs 7 --warmup 3 --check
```

Each cell is `CPU ms / CUDA ms / CUDA speedup`. These are small-to-medium development shapes, not hardware peak-throughput claims.

#### Dense, fused, normalization, and loss operators

| Operator | F32 | F16 | BF16 | I8 |
|---|---:|---:|---:|---:|
| `matmul.forward` | 2.146 / 0.055 / 38.73x | 1.207 / 0.023 / 52.70x | 0.798 / 0.026 / 31.04x | 2.070 / 0.327 / 6.34x |
| `batch_matmul.forward` | 0.008 / 0.014 / 0.56x | 0.040 / 0.011 / 3.60x | 0.014 / 0.014 / 1.03x | 0.141 / 0.073 / 1.93x |
| `matmul.backward` | 6.712 / 0.663 / 10.13x | 25.320 / 1.162 / 21.79x | 5.270 / 1.943 / 2.71x | 4.619 / 1.250 / 3.70x |
| `fused_gateup.forward` | 2.939 / 0.071 / 41.69x | 1.599 / 0.114 / 14.04x | 2.858 / 0.129 / 22.23x | 3.169 / 0.135 / 23.52x |
| `fused_qkv.decode` | 0.005 / 0.047 / 0.10x | 0.008 / 0.046 / 0.18x | 0.004 / 0.073 / 0.05x | 0.007 / 0.072 / 0.09x |
| `fused_qkv.prefill` | 1.025 / 0.085 / 12.10x | 1.093 / 0.083 / 13.09x | 1.016 / 0.074 / 13.66x | 1.058 / 0.212 / 5.00x |
| `softmax.forward` | 0.497 / 0.078 / 6.39x | 3.107 / 0.153 / 20.27x | 1.050 / 0.154 / 6.79x | 5.918 / 0.156 / 37.89x |
| `softmax.backward` | 3.956 / 1.341 / 2.95x | 3.354 / 1.329 / 2.52x | 3.740 / 1.303 / 2.87x | 4.225 / 1.662 / 2.54x |
| `fused_softmax.forward` | 2.150 / 0.077 / 28.00x | 4.731 / 0.078 / 60.49x | 2.763 / 0.080 / 34.41x | 5.077 / 0.082 / 61.69x |
| `fused_softmax.backward` | 5.606 / 1.329 / 4.22x | 5.760 / 1.181 / 4.88x | 5.644 / 1.248 / 4.52x | 6.390 / 1.314 / 4.86x |
| `embedding.forward` | 0.283 / 0.090 / 3.16x | 0.256 / 0.093 / 2.75x | 0.277 / 0.128 / 2.17x | 0.253 / 0.092 / 2.76x |
| `rms_norm.forward` | 0.279 / 0.032 / 8.87x | 0.319 / 0.036 / 8.95x | 0.277 / 0.038 / 7.31x | 2.207 / 0.113 / 19.44x |
| `rope.forward` | 0.035 / 0.012 / 2.98x | 0.040 / 0.017 / 2.34x | 0.038 / 0.017 / 2.26x | 0.091 / 0.076 / 1.21x |
| `cross_entropy.forward` | 0.504 / 0.077 / 6.55x | 1.182 / 0.226 / 5.24x | 1.170 / 0.205 / 5.71x | 8.425 / 0.232 / 36.33x |
| `cross_entropy.backward` | 0.859 / 0.116 / 7.40x | 1.213 / 0.212 / 5.73x | 1.205 / 0.352 / 3.42x | 5.492 / 0.279 / 19.68x |
| `mse_loss.forward` | 0.345 / 0.123 / 2.80x | 0.902 / 0.077 / 11.64x | 0.870 / 0.070 / 12.48x | 8.216 / 0.072 / 113.32x |
| `mse_loss.backward` | 1.567 / 0.108 / 14.48x | 1.468 / 0.110 / 13.41x | 1.562 / 0.115 / 13.63x | 1.373 / 0.111 / 12.38x |

#### Elementwise and broadcast operators

| Operator | F32 | F16 | BF16 | I8 |
|---|---:|---:|---:|---:|
| `elementwise.mul_add.forward` | 0.218 / 0.168 / 1.29x | 0.195 / 0.146 / 1.33x | 0.386 / 0.156 / 2.48x | 0.417 / 0.378 / 1.10x |
| `elementwise.mul_add.backward` | 3.312 / 1.283 / 2.58x | 2.970 / 1.217 / 2.44x | 2.973 / 1.246 / 2.39x | 3.175 / 1.262 / 2.52x |
| `binary.same_shape.forward` | 0.110 / 0.078 / 1.41x | 0.090 / 0.073 / 1.23x | 0.164 / 0.095 / 1.72x | 0.203 / 0.217 / 0.93x |
| `binary.row_broadcast.forward` | 1.260 / 0.077 / 16.45x | 0.100 / 0.072 / 1.39x | 0.165 / 0.086 / 1.92x | 0.198 / 0.184 / 1.07x |
| `binary.row_scalar.forward` | 0.265 / 0.122 / 2.18x | 2.403 / 0.075 / 31.91x | 0.700 / 0.079 / 8.87x | 9.263 / 0.185 / 49.99x |
| `binary.b1d_1h1.forward` | 0.262 / 0.118 / 2.21x | 2.206 / 0.093 / 23.64x | 0.564 / 0.141 / 4.00x | 7.131 / 0.169 / 42.22x |
| `binary.b1d_1hd.forward` | 0.112 / 0.106 / 1.06x | 2.071 / 0.078 / 26.66x | 0.391 / 0.078 / 4.99x | 6.873 / 0.203 / 33.91x |
| `binary.general_broadcast.forward` | 0.112 / 0.122 / 0.92x | 2.062 / 0.077 / 26.86x | 0.451 / 0.080 / 5.60x | 8.020 / 0.260 / 30.91x |
| `elementwise.mixed_mul.forward` | 0.110 / 0.079 / 1.38x | 0.226 / 0.075 / 3.02x | 0.265 / 0.079 / 3.35x | 0.234 / 0.086 / 2.72x |
| `elementwise.mixed_broadcast_1hd_mul.forward` | 0.003 / 0.020 / 0.13x | 0.003 / 0.009 / 0.30x | 0.003 / 0.009 / 0.32x | 0.005 / 0.011 / 0.42x |
| `elementwise.mixed_row_scalar_mul.forward` | 0.003 / 0.017 / 0.15x | 0.001 / 0.009 / 0.10x | 0.001 / 0.009 / 0.10x | 0.001 / 0.009 / 0.13x |
| `elementwise.mixed_mul.backward` | 2.576 / 0.711 / 3.62x | 2.420 / 1.136 / 2.13x | 2.287 / 1.019 / 2.24x | 2.222 / 1.065 / 2.09x |
| `elementwise.mixed_row_mul.forward` | 1.268 / 0.077 / 16.41x | 0.199 / 0.072 / 2.78x | 0.215 / 0.166 / 1.30x | 0.222 / 0.086 / 2.58x |
| `elementwise.mixed_row_mul.backward` | 2.489 / 1.069 / 2.33x | 0.937 / 1.055 / 0.89x | 1.056 / 1.241 / 0.85x | 0.950 / 1.227 / 0.77x |
| `elementwise.mixed_row_sub.backward` | 2.771 / 1.338 / 2.07x | 1.336 / 1.140 / 1.17x | 1.409 / 1.206 / 1.17x | 1.317 / 1.136 / 1.16x |
| `elementwise.mixed_scalar_sub.backward` | 1.800 / 1.219 / 1.48x | 1.575 / 1.097 / 1.44x | 1.742 / 1.114 / 1.56x | 1.562 / 1.095 / 1.43x |
| `elementwise.mixed_scalar_mul.backward` | 2.324 / 1.111 / 2.09x | 2.226 / 1.143 / 1.95x | 1.509 / 1.084 / 1.39x | 1.563 / 1.192 / 1.31x |
| `elementwise.mixed_broadcast_sub.backward` | 0.074 / 0.290 / 0.25x | 0.081 / 0.140 / 0.58x | 0.069 / 0.275 / 0.25x | 0.097 / 0.146 / 0.66x |
| `elementwise.mixed_broadcast_1hd_sub.backward` | 0.076 / 0.180 / 0.42x | 0.195 / 0.137 / 1.43x | 0.069 / 0.301 / 0.23x | 0.068 / 0.157 / 0.43x |
| `elementwise.mixed_broadcast_mul.backward` | 0.013 / 0.289 / 0.04x | 0.013 / 0.131 / 0.10x | 0.013 / 0.281 / 0.05x | 0.014 / 0.147 / 0.09x |
| `elementwise.mixed_broadcast_1hd_mul.backward` | 0.010 / 0.307 / 0.03x | 0.010 / 0.138 / 0.07x | 0.010 / 0.318 / 0.03x | 0.010 / 0.154 / 0.07x |
| `elementwise.mixed_row_scalar_mul.backward` | 0.011 / 0.122 / 0.09x | 0.015 / 0.116 / 0.13x | 0.010 / 0.216 / 0.05x | 0.010 / 0.210 / 0.05x |
| `unary.silu.forward` | 0.295 / 0.103 / 2.86x | 3.001 / 0.072 / 41.69x | 1.051 / 0.072 / 14.57x | 11.391 / 0.164 / 69.50x |
| `unary.relu.forward` | 0.137 / 0.083 / 1.65x | 2.989 / 0.097 / 30.78x | 0.814 / 0.078 / 10.38x | 11.124 / 0.072 / 154.29x |
| `unary.silu.backward` | 4.485 / 1.590 / 2.82x | 3.898 / 1.486 / 2.62x | 4.344 / 1.375 / 3.16x | 3.865 / 1.247 / 3.10x |

#### Optimizer, CNN, attention, and compact Llama operators

| Operator | F32 | F16 | BF16 | I8 |
|---|---:|---:|---:|---:|
| `optimizer.sgd.step` | 0.140 / 0.109 / 1.29x | 0.120 / 0.116 / 1.03x | 0.141 / 0.105 / 1.34x | 0.131 / 0.124 / 1.06x |
| `optimizer.adam.step` | 0.864 / 0.269 / 3.21x | 2.261 / 2.143 / 1.05x | 2.264 / 2.305 / 0.98x | skipped |
| `optimizer.adam_f32_state.step` | 0.812 / 0.222 / 3.66x | 0.267 / 0.240 / 1.11x | 0.260 / 0.244 / 1.06x | 0.279 / 0.260 / 1.07x |
| `optimizer.sgd_batched.step` | 0.021 / 0.042 / 0.51x | 0.016 / 0.318 / 0.05x | 0.017 / 0.284 / 0.06x | 0.030 / 0.360 / 0.08x |
| `optimizer.adam_f32_state_batched.step` | 0.201 / 0.087 / 2.31x | 0.060 / 0.369 / 0.16x | 0.062 / 0.357 / 0.17x | 0.073 / 0.465 / 0.16x |
| `conv2d.forward` | 0.836 / 0.360 / 2.32x | 0.769 / 0.205 / 3.75x | 0.831 / 0.249 / 3.34x | 0.897 / 0.207 / 4.32x |
| `conv2d.backward` | 2.357 / 1.336 / 1.76x | 2.527 / 1.200 / 2.11x | 2.694 / 1.145 / 2.35x | 2.806 / 1.992 / 1.41x |
| `max_pool2d.forward` | 0.129 / 0.053 / 2.46x | 0.109 / 0.043 / 2.56x | 0.100 / 0.064 / 1.55x | 0.108 / 0.031 / 3.48x |
| `max_pool2d.backward` | 0.305 / 0.299 / 1.02x | 0.287 / 0.294 / 0.97x | 0.257 / 0.282 / 0.91x | 0.296 / 0.572 / 0.52x |
| `self_attention.forward` | 0.314 / 0.238 / 1.32x | 0.694 / 0.439 / 1.58x | 0.464 / 0.783 / 0.59x | skipped |
| `self_attention.backward` | 0.563 / 0.661 / 0.85x | 2.014 / 0.758 / 2.66x | 1.559 / 1.655 / 0.94x | skipped |
| `self_attention_bias.backward` | 0.538 / 0.823 / 0.65x | 1.902 / 1.053 / 1.81x | 1.527 / 2.473 / 0.62x | skipped |
| `llama.infer_last_logits` | 1.187 / 0.699 / 1.70x | 1.464 / 0.731 / 2.00x | 1.234 / 1.387 / 0.89x | skipped |
| `llama.prefill_decode` | 1.517 / 0.865 / 1.75x | 2.156 / 1.122 / 1.92x | 1.673 / 1.100 / 1.52x | skipped |
| `llama.train.backward` | 4.799 / 2.309 / 2.08x | 3.285 / 2.411 / 1.36x | 3.314 / 4.124 / 0.80x | skipped |
| `llama.train.step` | 3.707 / 2.423 / 1.53x | 3.967 / 2.475 / 1.60x | 3.750 / 3.190 / 1.18x | skipped |

All enabled correctness checks passed. CUDA is strongest on dense, fused, softmax, loss, and larger broadcast work. Single-token QKV decode, tiny broadcast reductions, batched optimizer cases, and some compact attention/training cases remain launch- or dispatch-bound.

### End-to-end Llama prefill/decode snapshot

The TinyLlama rerun used `prompt_tokens=43`, `max_gen=64`, greedy decode, 3 measured runs, 1 warmup, and `--stop-on-eos --stop-on-chat-marker`.

| Configuration | Device | Prefill forward | Decode forward | End-to-end decode | Total |
|---|---|---:|---:|---:|---:|
| F32 | CPU | 40.47 tok/s | 10.67 tok/s | 9.06 tok/s | 7065.28 ms |
| F16 | CPU | 135.83 tok/s | 14.30 tok/s | 13.34 tok/s | 4796.53 ms |
| BF16 | CPU | 146.23 tok/s | 15.48 tok/s | 14.44 tok/s | 4432.81 ms |
| I8 weights + BF16 runtime | CPU | 143.73 tok/s | 24.28 tok/s | 21.79 tok/s | 2937.74 ms |
| F32 | CUDA | 998.25 tok/s | 41.72 tok/s | 40.45 tok/s | 1582.36 ms |
| F16 | CUDA | 944.64 tok/s | 48.16 tok/s | 46.38 tok/s | 1379.94 ms |
| BF16 | CUDA | 990.78 tok/s | 48.59 tok/s | 46.88 tok/s | 1365.20 ms |
| I8 weights + BF16 runtime | CUDA | 310.71 tok/s | 64.92 tok/s | 56.69 tok/s | 1129.04 ms |

The separate real-model F32 path checker generated identical fluent CPU/CUDA text for 32 tokens, with `replacement=0` and `control=0`; measured inference throughput was 7.16 tok/s on CPU and 36.20 tok/s on CUDA. The performance rerun shows that real generation is still decode-bound: CUDA decode-forward accounts for roughly 85-97% of measured time. Device-only CUDA hot paths now rely on default-stream ordering and synchronize only at explicit host-observation boundaries. Same-dtype F16/BF16 decode QKV and GateUp use cuBLAS `GemmEx` while preserving low-precision storage. Aligned and sufficiently large batched I8×I8 use signed-I8 cuBLAS GEMM with exact I32 accumulation. Inference-only F16/BF16×I8 prefill uses device-resident row-wise activation quantization before INT8 GEMM, and fused QKV/GateUp reuse the quantized activation. Training keeps direct F16/BF16×I8 forward computation so its F32 backward differentiates the same function. This raised I8+BF16 prefill from 258.52 to 310.71 tok/s while preserving fluent output with `replacement=0` and `control=0`; mixed-I8 prefill is still materially slower than native F16/BF16 GEMM.

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
