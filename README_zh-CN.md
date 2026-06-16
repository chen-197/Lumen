# Lumen

> 一个以 Rust 为主体的轻量级神经网络库，用于研究和构建 dtype-aware CPU/CUDA 执行、动态自动微分、可复用 layer 与 Llama-family 推理路径。

[English README](./README.md)

---

## 项目定位

Lumen 是一个**以 Rust 为主体的轻量级神经网络库 / 深度学习核心**。它把现代 ML stack 中的多层可复用能力集中在同一个仓库中：

- Tensor 核心与动态自动微分；
- 可复用的 Layer、Module、Loss、Optimizer；
- 作为内置模型族保留的 Llama 风格 decoder 模型实现；
- safetensors 权重加载与可选流式加载；
- 参数、激活、KV cache 的运行时 dtype 控制；
- 可选的 on-load 和 offline `i8` 量化；
- 带 x86/ARM kernel 调优空间的 CPU 执行路径；
- 通过 CUDA C++ kernel 和 NVIDIA 库实现的可选 CUDA 加速；
- 用于 CPU kernel、CUDA kernel、训练路径和端到端 Llama prefill/decode 的 benchmark 工具。

这个项目更适合被理解为一个**面向学习与实验的神经网络库**。紧凑的 Llama-family 实现是持续维护的重要端到端模型路径，但不是项目的唯一定位。

它不是生产级 serving 系统，不是完整训练框架，也不是可以直接适配任意 checkpoint 的通用启动器。

Lumen 是 **Rust-first, not Rust-only**。Rust 负责高层 runtime、tensor/autograd 系统、模型代码、loader、tokenizer wrapper、dtype 策略、CPU backend 和 benchmark 工具；可选 CUDA backend 使用 CUDA C++ kernel、cuBLAS/cuDNN，并通过 FFI 边界与 Rust 侧配合。

---

## 状态概览

| 领域 | 当前状态 |
|---|---|
| 自动微分与训练 | 动态自动微分、F32 梯度、SGD/SGD momentum/Adam、CPU 与持续扩展中的 CUDA 训练路径 |
| DType | F32、F16、BF16、I8 存储和运行时路径；I8 使用显式量化 scale |
| CPU kernel | Portable fallback，以及按 feature 开启的 x86/ARM 低精度 kernel |
| CUDA | CUDA-resident tensor、自定义 kernel、cuBLAS、可选 cuDNN、前向和部分反向/训练路径 |
| 模型支持 | 内置 Llama-family decoder 路径，包含 RoPE、GQA、RMSNorm、SwiGLU-style MLP 与 KV cache |
| 验证 | 准确性检查、F32 梯度检查、SGD loss 趋势检查、kernel benchmark 与真实模型文本检查 |

### 精度约定

- 支持原生路径时，参数和激活可以保持 F32、F16、BF16 或 I8 存储。
- 即使前向数据为低精度，反向梯度仍以 **F32** 表示和累加。
- Same-dtype 低精度 kernel 直接读取低精度存储。部分 CPU 路径会使用更宽的累加类型，例如 BF16/F16 累加到 F32、I8 累加到 I32，但不会先把完整输入物化为 F32。
- I8 是量化计算类型，不是浮点训练 dtype；其 scale 必须有限且大于零。
- 已支持的原生路径会与标量、量化或 CPU reference 对照；容差取决于 dtype 与归约顺序。

---

## 当前重点

当前代码库重点是通用 tensor/autograd/layer 行为、dtype-aware CPU 执行路径，以及持续演进的 CUDA 路径。Llama-family 支持仍然作为重要的内置模型路径保留。

比较重要的部分包括：

- 动态自动微分和通用 Tensor ops；
- 可复用神经网络 layer 和序列建模组件，包括 Llama decoder 需要的 RMSNorm、RoPE、GQA、SwiGLU-style MLP、KV-cache decode；
- F32、F16、BF16、I8 的存储、加载和运行时配置，以及 F32 梯度；
- 通过 `cuda` feature 开启的可选 CUDA 执行路径；
- CUDA-resident tensor、KV-cache 更新、decode-oriented kernel、forward path，以及逐步扩展中的 backward / training path；
- x86 backend，例如 AVX-512 BF16、AVX2/F16C，以及 AVX-512BW/AVX2 I8 kernels；
- 可选参数 dtype copy，用于混合精度执行；
- 可选流式权重加载，用于降低峰值内存占用；
- 开发用 benchmark binary，用于 CPU/CUDA 调优和端到端推理测量。

---

## 设计概览

```text
Rust 侧
  ├─ Tensor 表示与动态自动微分图
  ├─ Layers、modules、losses、optimizers
  ├─ 模型实现，包括 Llama-family 支持
  ├─ dtype / precision / quantization 策略
  ├─ safetensors 加载与 tokenizer 集成
  ├─ CPU kernels 与 backend dispatch
  └─ CUDA FFI wrappers

CUDA 侧
  ├─ device memory 分配与复用
  ├─ 自定义 CUDA kernels
  ├─ cuBLAS 矩阵运算
  ├─ 可选 cuDNN primitives
  ├─ KV-cache 与 decode-oriented kernels
  └─ 部分 forward / backward / training kernels
```

Rust 负责框架结构、类型组织、运行时策略和安全边界；CUDA 负责那些更适合直接在 GPU 上执行的低层计算路径。

---

## 仓库结构

```text
src/
├─ autograd.rs              # Tensor + 动态自动微分核心
├─ module.rs                # Module trait / macros
├─ loader.rs                # Safetensors 加载与流式加载
├─ tokenizer.rs             # Tokenizer wrapper
├─ precision.rs             # DType / runtime precision 配置
├─ ops/                     # Tensor ops、CPU kernels、可选 CUDA ops
│  └─ cuda/                 # CUDA/cuDNN/cuBLAS-backed kernels 与模块
├─ layers/                  # 神经网络 layers 与 attention 组件
├─ models/llama.rs          # Llama 模型实现
├─ main.rs                  # 最小本地推理 CLI
└─ bin/
   ├─ quantize_safetensors.rs  # Offline quantization 工具
   ├─ kernel_bench.rs          # 开发用 CPU kernel benchmark
   ├─ prefill_decode_bench.rs  # 端到端 prefill/decode benchmark
   ├─ cuda_cpu_bench.rs        # CPU/CUDA ops、NN、backward 与 path benchmark
   └─ cuda_cpu_bench_path.rs   # cuda_cpu_bench 使用的 path checks
```

---

## 构建

CPU-only release 构建：

```bash
cargo build --release
```

收集本机性能数据时建议开启 native CPU codegen：

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

PowerShell：

```powershell
$env:RUSTFLAGS = "-C target-cpu=native"
cargo build --release
```

CUDA 构建：

```bash
cargo build --release --features cuda
```

开发 benchmark 构建：

```bash
cargo build --release --features dev-tools --bin kernel_bench
cargo build --release --features dev-tools --bin prefill_decode_bench
cargo build --release --features dev-tools --bin cuda_cpu_bench
```

在 x86 机器上做性能测试时，建议显式启用 x86 backend features：

```bash
cargo build --release --features "dev-tools x86-fp-kernels x86-int8-kernels" --bin kernel_bench
cargo build --release --features "dev-tools cuda x86-fp-kernels x86-int8-kernels" --bin prefill_decode_bench
cargo build --release --features "dev-tools cuda x86-fp-kernels x86-int8-kernels" --bin cuda_cpu_bench
```

`x86-fp-kernels` 可在 stable Rust 上构建，包含稳定的 AVX2/F16C、AVX-512F 和 AVX-512 BF16 路径。真正的 AVX-512 FP16 same-dtype 计算依赖 Rust nightly 的 `stdarch_x86_avx512_f16`，需要改用：

```bash
cargo +nightly build --release --features "dev-tools x86-fp-kernels-nightly x86-int8-kernels" --bin kernel_bench
```

CUDA 构建脚本会从环境变量、`nvcc` 和常见系统安装路径中查找 CUDA/cuDNN。CPU-only 构建不需要 CUDA。

### Feature 指南

| Feature | 用途 |
|---|---|
| `cuda` | 构建可选 CUDA C++ backend |
| `dev-tools` | 构建 benchmark 与 path-check binaries |
| `x86-fp-kernels` | 启用 stable x86 BF16/F16/F32 快路径 |
| `x86-fp-kernels-nightly` | 在 `x86-fp-kernels` 基础上启用真 AVX-512 FP16 intrinsics |
| `x86-int8-kernels` | 启用 x86 I8 快路径 |
| `arm64-fp-kernels` | 启用 ARM64 浮点/低精度快路径 |
| `arm64-int8-kernels` | 启用 ARM64 I8 快路径 |

架构相关 CPU features 默认不启用。评估 CPU 或 CUDA 性能时应显式开启对应 feature，避免 CPU fallback 和 helper 工作通过 portable backend 执行。

---

## 最小推理 CLI

```bash
cargo run --release --bin lumen -- \
  --weights path/to/model.safetensors \
  --tokenizer path/to/tokenizer.json
```

常用参数：

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

BF16 示例：

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

I8 weights + BF16 runtime 示例：

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

可以通过下面的环境变量打印 backend 诊断信息：

```bash
LUMEN_SHOW_BACKENDS=1 cargo run --release --bin lumen -- \
  --weights path/to/model.safetensors \
  --tokenizer path/to/tokenizer.json
```

交互命令：

- `/reset`：清空 chat history 和 KV cache；
- `/exit`：退出程序。

---

## Offline quantization

生成 `i8` safetensors checkpoint：

```bash
cargo run --release --bin quantize_safetensors -- \
  --input path/to/model.safetensors \
  --output path/to/model.i8.safetensors \
  --dtype i8
```

手动指定 scale：

```bash
cargo run --release --bin quantize_safetensors -- \
  --input path/to/model.safetensors \
  --output path/to/model.i8.safetensors \
  --dtype i8 \
  --scale 0.02
```

---

## Benchmark 工具

性能测试请使用 `--release`。Debug build 适合查正确性，但不能代表真实性能。

### Kernel benchmark

```bash
cargo run --release --features "dev-tools x86-fp-kernels x86-int8-kernels" --bin kernel_bench -- \
  --iters 400 --samples 8 --hidden 2048 --inter 5632 --vocab 32000
```

输出中的 `dot_bf16_bf16`、`dot2_bf16_bf16`、`dot3_bf16_bf16`，`dot_f16_f16`、`dot2_f16_f16`、`dot3_f16_f16`，以及 `dot2_i8_i8`、`dot3_i8_i8` 行是底层 same-dtype 低精度 dot microbenchmark。BF16 在支持 AVX-512 BF16 时优先使用 `_mm256_dpbf16_ps`；否则 stable `x86-fp-kernels` 使用 AVX2/FMA 直接读取 BF16 存储并以 F32 lane 累加。F16 的 stable 路径使用 AVX2/F16C 直接读取 F16 存储并以 F32 lane 累加；nightly `x86-fp-kernels-nightly` 在运行时支持 AVX-512 FP16 时优先使用 `_mm512_*_ph` 真 FP16 kernel。I8 same-dtype 路径优先使用 AVX-512BW；否则 stable `x86-int8-kernels` 使用 AVX2 将 signed I8 直接扩展到 I16，通过 `_mm256_madd_epi16` 累加到 I32 后再乘 scale。

如果需要测 `_mm512_loadu_ph/_mm512_storeu_ph/_mm512_reduce_add_ph` 等真 AVX-512 FP16 kernel，请使用 nightly feature：

```bash
cargo +nightly run --release --features "dev-tools x86-fp-kernels-nightly x86-int8-kernels" --bin kernel_bench -- \
  --iters 400 --samples 8 --hidden 2048 --inter 5632 --vocab 32000
```

### CPU/CUDA ops 与训练 benchmark

```bash
cargo run --release --features "dev-tools cuda x86-fp-kernels x86-int8-kernels" --bin cuda_cpu_bench -- \
  --suite all --size medium --dtype bf16 --runs 5 --warmup 2 --check
```

常用选项：

- `--suite all|ops|nn|backward|path`
- `--size small|medium|large`
- `--dtype f32|f16|bf16|i8`
- `--case TEXT`：只运行名称中包含 `TEXT` 的 case
- `--check`：运行 CPU/CUDA correctness checks 和 path checks

### 端到端 prefill/decode benchmark

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

真实模型 path check：

```bash
cargo run --release --features "dev-tools cuda x86-fp-kernels x86-int8-kernels" --bin cuda_cpu_bench -- \
  --suite path --check --path-device cuda \
  --weights path/to/model.safetensors \
  --tokenizer path/to/tokenizer.json \
  --max-gen 32 --show-output
```

Path checks 不是纯 microbenchmark，它们主要用于发现算法路径问题：

- 训练路径会运行一个很小的 SGD-like trace，检查 loss 是否符合合理趋势；
- 推理路径可以加载真实 Llama/TinyLlama checkpoint，检查生成文本是否出现明显乱码。

---

## 当前本地性能快照

下面数字是 2026-06-16 刷新的本地开发快照，不是通用 benchmark 结论。测试机器大致配置：

- CPU：AMD Ryzen 9 8945HX with Radeon Graphics
- GPU：NVIDIA GeForce RTX 5070 Laptop GPU，8 GB VRAM
- RAM：32 GB
- NVIDIA driver / runtime CUDA：`610.47` / CUDA `13.3`
- CUDA toolkit：13.0
- cuDNN：9.21.1
- Rust：stable MSVC toolchain，`rustc 1.95.0`

最近一次重新测试同时启用了 CUDA 和 x86 backend features：

```text
backend: float=x86-avx512 bf16_bf16=x86-avx512bf16 f16_f16=x86-avx2-f16c
         int8=x86-avx512bw i8_i8=x86-avx512bw avx512fp16=unavailable-or-stable-build
```

这点很重要：如果 CUDA 端到端测试显示 `backend: portable`，CPU fallback 和 CPU-side helper path 就不是这台机器上的最佳路径，不能作为最终性能基线。

### 全面准确性与训练检查

本轮执行命令：

```bash
cargo fmt --all -- --check
cargo test --release --all-targets --features "cuda x86-fp-kernels x86-int8-kernels"
cargo test --release --lib --features "cuda x86-fp-kernels x86-int8-kernels" -- --ignored --nocapture
cargo test --release --lib --no-default-features --features "x86-fp-kernels x86-int8-kernels"
cargo clippy --release --all-targets --features "cuda x86-fp-kernels x86-int8-kernels" -- -D warnings
cargo clippy --release --lib --no-default-features --features "x86-fp-kernels x86-int8-kernels" -- -D warnings

# 对 f32/f16/bf16/i8 分别执行
cargo run --release --features "dev-tools cuda x86-fp-kernels x86-int8-kernels" --bin cuda_cpu_bench -- \
  --suite all --size medium --dtype DTYPE --runs 3 --warmup 1 --check

# 对 f32/f16/bf16/i8 分别执行
cargo run --release --features "dev-tools cuda x86-fp-kernels x86-int8-kernels" --bin cuda_cpu_bench -- \
  --suite path --check --case path.train --dtype DTYPE --runs 1 --warmup 0

# 对 f32/f16/bf16/i8+bf16 分别在 CPU/CUDA 执行
cargo run --release --features "dev-tools cuda x86-fp-kernels x86-int8-kernels" --bin prefill_decode_bench -- \
  --weights C:\Users\chen-\Downloads\model.safetensors \
  --tokenizer C:\Users\chen-\Downloads\tokenizer.json \
  --device DEVICE --parameter-dtype PARAM_DTYPE --runtime-dtype RUNTIME_DTYPE \
  --activation-dtype ACTIVATION_DTYPE --kv-cache-dtype KV_DTYPE --quantize QUANTIZE \
  --max-gen 24 --max-seq-len 256 --runs 1 --warmup 0 --mode greedy \
  --stop-on-eos --stop-on-chat-marker --system "You are a concise assistant." \
  --prompt "Write one short sentence about neural networks." --show-output
```

结果：

- CUDA all-target 回归：库测试 `440 passed; 0 failed; 9 ignored`，空 main binary 测试目标通过，量化工具 `4 passed; 0 failed`；
- 9 个显式性能烟测单独执行：`9 passed; 0 failed`；
- CPU-only 回归：`251 passed; 0 failed; 6 ignored`；
- CUDA 与 CPU 两套 clippy `-D warnings` 均通过；
- TinyLlama 真实模型推理在 CPU/CUDA 的 F32、F16、BF16、I8 weights + BF16 runtime 全部通过，每组生成文本均为 `replacement=0`、`trailing_replacement=0`、`control=0`；
- F32、F16、BF16、I8 的 CPU/CUDA 前向、反向、F32 梯度、优化器、紧凑 Llama F32/F16/BF16 训练检查，以及 I8 参数训练检查均通过；
- I8 的原生 Adam 状态和完整 I8 Llama runtime 按设计跳过；I8 参数配 F32 optimizer state 以及 I8 + BF16 runtime path checks 通过。

`cuda_cpu_bench --suite all --check` 中代表性 CPU/CUDA 同 dtype 数值差异：

| 检查 | F32 max abs | F16 max abs | BF16 max abs | I8 max abs |
|---|---:|---:|---:|---:|
| matrix matmul forward | `3.815e-6` | `7.629e-6` | `3.008e-2` | `1.144e-5` |
| matrix matmul lhs grad | `2.384e-7` | `2.384e-7` | `3.576e-7` | `2.384e-7` |
| matrix matmul rhs grad | `4.768e-7` | `4.768e-7` | `4.768e-7` | `4.768e-7` |

BF16 前向误差明显高于 F16，这是 BF16 尾数精度与不同归约顺序的预期结果；梯度保持 F32。I8 前向数字是量化路径检查，不表示与浮点计算逐值等价。

训练路径现在对每种 dtype 覆盖 21 条 CPU/CUDA 路径：scalar SGD、MLP、GELU MLP、Dropout MLP、BatchMatMul、gated MLP、Embedding classifier、shape/view chain、shared/tied parameter reuse、gradient accumulation、SGD 批量 optimizer 更新、RMSNorm、RNN、GRU、LSTM、Adam、Adam 批量 optimizer 更新、Conv2D、Conv2D+MaxPool、SelfAttention、紧凑 Transformer block。I8 attention 和 Transformer block 检查使用 I8 参数与 BF16 runtime/KV-cache 数据。

24 步 SGD + momentum 训练路径观察到的 loss 趋势：

| DType | CPU first -> last | CUDA first -> last | 过程中上升次数 |
|---|---:|---:|---:|
| F32 | `9.0 -> 1e-6` | `9.0 -> 1e-6` | 7 |
| F16 | `9.0 -> 2e-6` | `9.0 -> 2e-6` | 6 |
| BF16 | `9.0 -> 0` | `9.0 -> 0` | 3 |
| I8 | `9.0 -> 4e-4` | `9.0 -> 4e-4` | 3 |

这些轨迹不是单调下降，但都有清晰下降趋势，符合随机梯度下降类训练的检查目标。

2026-06-16 本轮 24 步 SGD + momentum 小训练路径计时：

| DType | CPU us/step | CUDA us/step | 说明 |
|---|---:|---:|---|
| F32 | 51.72 | 2197.00 | CUDA 梯度与 momentum state 保持 F32 |
| F16 | 79.85 | 2778.32 | 低精度参数，F32 梯度 |
| BF16 | 51.07 | 2282.88 | 低精度参数，F32 梯度 |
| I8 | 134.49 | 2433.09 | 量化参数，F32 梯度 |

批量 optimizer 路径计时快照。两个 CUDA 路径都会断言实际使用 pointer-batched optimizer kernel，同时梯度和 optimizer state 保持 F32：

| 路径 | DType | CPU us/step | CUDA us/step | Loss first -> last |
|---|---:|---:|---:|---:|
| SGD momentum 批量更新 | F32 | 576.59 | 2992.01 | `0.042612 -> 0.035707` |
| SGD momentum 批量更新 | F16 | 3133.51 | 4016.52 | `0.042624 -> 0.035717` |
| SGD momentum 批量更新 | BF16 | 2022.76 | 1878.86 | `0.042695 -> 0.035773` |
| SGD momentum 批量更新 | I8 | 1940.68 | 4371.20 | `0.042661 -> 0.037453` |
| Adam 批量更新 | F32 | 807.70 | 1610.01 | `0.032873 -> 0.000897` |
| Adam 批量更新 | F16 | 2955.92 | 1607.95 | `0.032875 -> 0.000898` |
| Adam 批量更新 | BF16 | 2394.90 | 1625.08 | `0.032754 -> 0.000915` |
| Adam 批量更新 | I8 | 2052.58 | 3984.49 | `0.032910 -> 0.017634` |

这些是 8 个 Linear shard 组成的小型端到端训练图，因此 CUDA 在 optimizer 更新之外仍常受 kernel launch 和 dispatch 开销主导。

2026-06-16 本轮性能烟测要点：

| Case | Result |
|---|---:|
| BF16 same-dtype dot/dot2/dot3 backend | `x86-avx512bf16` |
| BF16 dot / dot2 / dot3 | 0.411 / 0.549 / 0.620 us |
| F16 same-dtype dot/dot2/dot3 backend | `x86-avx2-f16c` |
| F16 dot / dot2 / dot3 | 0.382 / 0.405 / 0.448 us |
| I8 same-dtype dot2 / dot3 backend | `x86-avx512bw` |
| I8 dot2 / dot3 | 0.260 / 0.332 us |
| CPU batch matmul backward，BF16xBF16 / I8xI8 / BF16xI8 / F32xI8 | 2019.7 / 391.0 / 325.7 / 274.4 us |
| CUDA dynamic I8 quantize，1M elements | 99.0 us |
| CUDA lowp sum，1M elements F32 / F16 / BF16 / I8 | 1374.4 / 319.9 / 303.6 / 394.3 us |
| CUDA I8xI8 matmul，F32 out | 18.5 us，`kernel_err=0.00002` |
| CUDA I8xI8 matmul，typed I8 out | 82.0 us，`quant_err=1.00382` |
| CUDA I8xI8 batch matmul，F32 / typed I8 out | 13.5 / 69.9 us |

### CPU kernel 快照

命令：

```bash
cargo run --release --features "dev-tools x86-fp-kernels x86-int8-kernels" --bin kernel_bench -- \
  --iters 300 --samples 7 --hidden 2048 --inter 5632 --vocab 32000
```

实际启用的 backend：

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

真 AVX-512 FP16 路径仍需 nightly Rust 与 `x86-fp-kernels-nightly`。Cached copy 是否有利取决于具体路径：本轮 cached BF16/F16 tensor matmul 更快，而 cached fused F16 QKV/GateUp 明显慢于 no-copy。

### 详细 CPU/CUDA 算子快照

命令族：

```bash
# 对 f32/f16/bf16/i8 分别执行
cargo run --release --features "dev-tools cuda x86-fp-kernels x86-int8-kernels" --bin cuda_cpu_bench -- \
  --suite all --size medium --dtype DTYPE --runs 3 --warmup 1 --check
```

每个单元格格式为 `CPU ms / CUDA ms / CUDA 加速比`。这些是 small-to-medium 开发用 shape，并不代表硬件峰值吞吐。

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

所有已启用 correctness checks 均通过。CUDA 在 dense、fused、softmax、loss 和较大 broadcast 工作上优势最明显；单 token QKV decode、极小 broadcast reduction、batched optimizer，以及部分紧凑 attention/训练 case 仍受 launch 或 dispatch 开销主导。

### 端到端 Llama prefill/decode 快照

本轮 TinyLlama 使用本地 `tokenizer.json` 与 `model.safetensors`，`prompt_tokens=42`、`max_gen=24`、greedy decode、1 次测量、0 次预热，并开启 `--stop-on-eos --stop-on-chat-marker`。

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

8 组生成样本在这个短 prompt 上都流畅，均报告 `replacement=0`、`trailing_replacement=0`、`control=0`。所有配置生成了同一句：“Neural networks are a powerful tool for analyzing and understanding complex data.” 本轮结果表明真实生成仍明显 decode-bound：decode-forward 主导测量时间，而 I8 weights + BF16 runtime/activation/KV cache 在这次短真实模型运行中拥有最高 end-to-end decode throughput。

---

## 设计限制

`src/main.rs` 和部分 benchmark path 目前有意使用硬编码的 `model_config()`。这样可以让 runtime 更紧凑、更容易检查，但也意味着：

- 加载的 checkpoint 必须匹配预期模型结构；
- 适配不同模型时，可能需要修改 hidden size、layer count、KV-head layout、tokenizer 行为或 prompt formatting；
- 当前 CLI 更像本地 runner，而不是通用推理前端。

Benchmark 工具主要服务于开发和 kernel/runtime 调优。它们适合做 regression tracking 和本地对比，但不是完整包装过的公开 benchmark suite。

CUDA 支持已经可以实际运行，但仍在持续演进。并非每个算子或 dtype 组合都拥有同等优化程度的 CUDA 路径；小工作负载也可能因 kernel launch 和调度开销而慢于 CPU。当前实际目标是单 CUDA device；未来如果要支持多 GPU，还需要把 device index 明确传递到 tensor、module 和 CUDA call 中。

---

## 适合谁使用

Lumen 适合以下场景：

- 学习 Rust tensor / autograd core 如何组织；
- 研究一个没有大型框架包裹的轻量 Llama runtime；
- 实验 dtype 策略、量化、CPU kernels 和 CUDA kernels；
- 在自己的机器上 benchmark 和调优紧凑 Rust inference stack；
- 观察 Rust 高层 runtime 如何与 CUDA 低层 kernel 配合。

它可能不适合以下需求：

- 大规模分布式训练；
- 成熟的生产级 serving；
- 完整多 GPU 部署工具；
- 任意模型家族的 plug-and-play 支持。

---

## License

本仓库使用 [`LICENSE`](./LICENSE) 中包含的许可证发布。
