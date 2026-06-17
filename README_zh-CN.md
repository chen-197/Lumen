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

下面数字是 2026-06-17 刷新的本地开发快照，不是通用 benchmark 结论。测试机器大致配置：

- CPU：AMD Ryzen 9 8945HX with Radeon Graphics
- GPU：NVIDIA GeForce RTX 5070 Laptop GPU，8151 MiB VRAM
- RAM：32 GB
- NVIDIA driver / runtime CUDA：`610.62` / CUDA `13.3`
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
  --suite path --dtype DTYPE --check

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

- CUDA all-target 回归：库测试 `441 passed; 0 failed; 9 ignored`，空 main binary 测试目标通过，量化工具 `4 passed; 0 failed`；
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

训练路径现在对每种 dtype 覆盖 23 条 CPU/CUDA 路径：scalar SGD、MLP、GELU MLP、Dropout MLP、BatchMatMul、gated MLP、residual/gated branch mixing、row-broadcast affine parameters、Embedding classifier、shape/view chain、shared/tied parameter reuse、gradient accumulation、SGD 批量 optimizer 更新、RMSNorm、RNN、GRU、LSTM、Adam、Adam 批量 optimizer 更新、Conv2D、Conv2D+MaxPool、SelfAttention、紧凑 Transformer block。I8 attention 和 Transformer block 检查使用 I8 参数与 BF16 runtime/KV-cache 数据。

24 步 SGD + momentum 训练路径观察到的 loss 趋势：

| DType | CPU first -> last | CUDA first -> last | 过程中上升次数 |
|---|---:|---:|---:|
| F32 | `9.0 -> 1e-6` | `9.0 -> 1e-6` | 7 |
| F16 | `9.0 -> 2e-6` | `9.0 -> 2e-6` | 6 |
| BF16 | `9.0 -> 0` | `9.0 -> 0` | 3 |
| I8 | `9.0 -> 4e-4` | `9.0 -> 4e-4` | 3 |

这些轨迹不是单调下降，但都有清晰下降趋势，符合随机梯度下降类训练的检查目标。

2026-06-17 本轮 24 步 SGD + momentum 小训练路径计时：

| DType | CPU us/step | CUDA us/step | 说明 |
|---|---:|---:|---|
| F32 | 93.35 | 2405.87 | CUDA 梯度与 momentum state 保持 F32 |
| F16 | 114.42 | 2399.96 | 低精度参数，F32 梯度 |
| BF16 | 136.10 | 2577.54 | 低精度参数，F32 梯度 |
| I8 | 86.48 | 2515.04 | 量化参数，F32 梯度 |

批量 optimizer 路径计时快照。两个 CUDA 路径都会断言实际使用 pointer-batched optimizer kernel，同时梯度和 optimizer state 保持 F32：

| 路径 | DType | CPU us/step | CUDA us/step | Loss first -> last |
|---|---:|---:|---:|---:|
| SGD momentum 批量更新 | F32 | 719.18 | 2081.82 | `0.042612 -> 0.035707` |
| SGD momentum 批量更新 | F16 | 3292.39 | 1805.36 | `0.042624 -> 0.035717` |
| SGD momentum 批量更新 | BF16 | 2774.70 | 2048.36 | `0.042695 -> 0.035773` |
| SGD momentum 批量更新 | I8 | 2388.58 | 2838.08 | `0.042661 -> 0.037453` |
| Adam 批量更新 | F32 | 980.41 | 1301.71 | `0.032873 -> 0.000897` |
| Adam 批量更新 | F16 | 3457.87 | 1466.73 | `0.032875 -> 0.000898` |
| Adam 批量更新 | BF16 | 2910.04 | 1728.02 | `0.032754 -> 0.000915` |
| Adam 批量更新 | I8 | 2559.43 | 3499.50 | `0.032910 -> 0.017634` |

这些是 8 个 Linear shard 组成的小型端到端训练图，因此 CUDA 在 optimizer 更新之外仍常受 kernel launch 和 dispatch 开销主导。

2026-06-17 本轮性能烟测要点：

| Case | Result |
|---|---:|
| BF16 same-dtype dot/dot2/dot3 backend | `x86-avx512bf16` |
| BF16 dot / dot2 / dot3 | 0.308 / 0.450 / 0.630 us |
| F16 same-dtype dot/dot2/dot3 backend | `x86-avx2-f16c` |
| F16 dot / dot2 / dot3 | 0.393 / 0.441 / 0.460 us |
| I8 same-dtype dot2 / dot3 backend | `x86-avx512bw` |
| I8 dot2 / dot3 | 0.347 / 0.351 us |
| CPU batch matmul backward，BF16xBF16 / I8xI8 / BF16xI8 / F32xI8 | 2335.0 / 400.9 / 476.1 / 808.2 us |
| CUDA dynamic I8 quantize，1M elements | 87.1 us |
| CUDA lowp sum，1M elements F32 / F16 / BF16 / I8 | 240.3 / 162.0 / 134.2 / 140.4 us |
| CUDA I8xI8 matmul，F32 out | 22.4 us，`kernel_err=0.00002` |
| CUDA I8xI8 matmul，typed I8 out | 117.9 us，`quant_err=1.00382` |
| CUDA I8xI8 batch matmul，F32 / typed I8 out | 14.3 / 86.4 us |

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

所有已启用 correctness checks 均通过。CUDA 在 dense、fused、softmax、loss 和较大 broadcast 工作上优势最明显；单 token QKV decode、极小 broadcast reduction、batched optimizer，以及部分紧凑 attention/训练 case 仍受 launch 或 dispatch 开销主导。

### 端到端 Llama prefill/decode 快照

本轮 TinyLlama 使用本地 `tokenizer.json` 与 `model.safetensors`，`prompt_tokens=42`、`max_gen=24`、greedy decode、1 次测量、0 次预热，并开启 `--stop-on-eos --stop-on-chat-marker`。

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
