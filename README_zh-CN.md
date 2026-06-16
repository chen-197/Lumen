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

执行命令：

```bash
cargo test --release --all-targets --features "cuda x86-fp-kernels x86-int8-kernels"
cargo test --release --lib --features "cuda x86-fp-kernels x86-int8-kernels" -- --ignored --nocapture
cargo test --release --lib --no-default-features --features "x86-fp-kernels x86-int8-kernels"
cargo clippy --release --all-targets --features "cuda x86-fp-kernels x86-int8-kernels" -- -D warnings
cargo clippy --release --all-targets --no-default-features --features "x86-fp-kernels x86-int8-kernels" -- -D warnings

# 对 f32/f16/bf16/i8 分别执行
cargo run --release --features "dev-tools cuda x86-fp-kernels x86-int8-kernels" --bin cuda_cpu_bench -- \
  --suite all --size medium --dtype DTYPE --runs 7 --warmup 3 --check
```

结果：

- CUDA all-target 回归：库测试 `436 passed; 0 failed; 9 ignored`，量化工具 `4 passed; 0 failed`；
- 9 个显式性能烟测单独执行：`9 passed; 0 failed`；
- CPU-only 回归：`249 passed; 0 failed; 6 ignored`；
- CUDA 与 CPU 两套 clippy `-D warnings` 均通过；
- TinyLlama 真实模型推理在 CPU/CUDA 的 F32、F16、BF16、I8 weights + BF16 runtime 全部通过，每组生成文本均为 `replacement=0`、`control=0`；
- F32、F16、BF16、I8 的 CPU/CUDA 前向、反向、F32 梯度、优化器、紧凑 Llama F32/F16/BF16 训练检查，以及独立 I8 参数训练检查均通过；
- I8 的原生 Adam 状态和完整 I8 Llama runtime 按设计跳过；I8 参数配 F32 optimizer state 及独立训练路径通过。

代表性 CPU/CUDA 同 dtype 数值差异：

| 检查 | F32 max abs | F16 max abs | BF16 max abs | I8 max abs |
|---|---:|---:|---:|---:|
| matrix matmul forward | `3.815e-6` | `7.629e-6` | `3.008e-2` | `1.144e-5` |
| matrix matmul lhs grad | `2.384e-7` | `2.384e-7` | `3.576e-7` | `2.384e-7` |
| matrix matmul rhs grad | `4.768e-7` | `4.768e-7` | `4.768e-7` | `4.768e-7` |

BF16 前向误差明显高于 F16，这是 BF16 尾数精度与不同归约顺序的预期结果；梯度保持 F32。BF16 Llama 训练梯度在接近零的位置会出现较大的相对误差，但最大绝对差为 `1.178e-3`，检查通过。

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
| F32 | 125.76 | 2646.00 | CUDA 梯度与 momentum state 保持 F32 |
| F16 | 134.58 | 2443.51 | 低精度参数，F32 梯度 |
| BF16 | 47.66 | 2533.73 | 低精度参数，F32 梯度 |
| I8 | 119.11 | 2287.18 | 量化参数，F32 梯度 |

紧凑 Llama 训练检查在 F32/F16/BF16 的 CPU 与 CUDA 上也通过。I8 紧凑 Llama 训练 case 按设计跳过，因为该 bench 构造器要求浮点 runtime 与 KV-cache dtype；I8 参数训练由 `path.train` 覆盖。

2026-06-16 本轮性能烟测要点：

| Case | Result |
|---|---:|
| BF16 same-dtype dot/dot2/dot3 backend | `x86-avx512bf16` |
| BF16 dot / dot2 / dot3 | 0.305 / 0.348 / 0.426 us |
| F16 same-dtype dot/dot2/dot3 backend | `x86-avx2-f16c` |
| F16 dot / dot2 / dot3 | 0.402 / 0.428 / 0.463 us |
| I8 same-dtype dot2 / dot3 backend | `x86-avx512bw` |
| I8 dot2 / dot3 | 0.237 / 0.340 us |
| CUDA dynamic I8 quantize，1M elements | 239.2 us |
| CUDA I8xI8 matmul，F32 out | 24.6 us，`kernel_err=0.00002` |
| CUDA I8xI8 matmul，typed I8 out | 120.3 us，`quant_err=1.00382` |
| CUDA I8xI8 batch matmul，F32 / typed I8 out | 13.4 / 90.1 us |

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
  --suite all --size medium --dtype DTYPE --runs 7 --warmup 3 --check
```

每个单元格格式为 `CPU ms / CUDA ms / CUDA 加速比`。这些是 small-to-medium 开发用 shape，并不代表硬件峰值吞吐。

#### Dense、fused、归一化与 loss 算子

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

#### Elementwise 与 broadcast 算子

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

#### Optimizer、CNN、attention 与紧凑 Llama 算子

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

所有已启用 correctness checks 均通过。CUDA 在 dense、fused、softmax、loss 和较大 broadcast 工作上优势最明显；单 token QKV decode、极小 broadcast reduction、batched optimizer，以及部分紧凑 attention/训练 case 仍受 launch 或 dispatch 开销主导。

### 端到端 Llama prefill/decode 快照

本轮 TinyLlama 使用本地 `tokenizer.json` 与 `model.safetensors`，`prompt_tokens=48`、`max_gen=24`、greedy decode、1 次测量、0 次预热，并开启 `--stop-on-eos --stop-on-chat-marker`。

| Configuration | Device | Prefill forward | Decode forward | End-to-end decode | Total |
|---|---|---:|---:|---:|---:|
| F32 | CPU | 40.38 tok/s | 9.76 tok/s | 6.57 tok/s | 3650.59 ms |
| F16 | CPU | 128.27 tok/s | 15.12 tok/s | 12.22 tok/s | 1963.40 ms |
| BF16 | CPU | 127.66 tok/s | 15.20 tok/s | 12.27 tok/s | 1956.53 ms |
| I8 weights + BF16 runtime | CPU | 178.60 tok/s | 22.69 tok/s | 18.07 tok/s | 1327.90 ms |
| F32 | CUDA | 302.18 tok/s | 42.20 tok/s | 32.92 tok/s | 729.12 ms |
| F16 | CUDA | 159.93 tok/s | 32.56 tok/s | 22.79 tok/s | 1052.92 ms |
| BF16 | CUDA | 163.08 tok/s | 29.39 tok/s | 21.53 tok/s | 1114.71 ms |
| I8 weights + BF16 runtime | CUDA | 222.98 tok/s | 67.49 tok/s | 41.93 tok/s | 572.42 ms |

8 组生成样本在这个短 prompt 上都流畅，均报告 `replacement=0`、`trailing_replacement=0`、`control=0`。F32/F16/BF16 生成了相同的开头，说明 Transformer 架构会存储中间结果；I8+BF16 生成了略有差异但仍连贯的句子，说明 Transformer-based 模型通过存储中间结果提升性能。本轮结果表明真实生成仍明显 decode-bound：decode-forward 主导 CUDA 测量时间。Device-only CUDA 热路径现在依靠默认 stream 保序，只在显式 Host 观察边界同步；同 dtype F16/BF16 decode QKV 和 GateUp 使用 cuBLAS `GemmEx` 计算并保持低精度存储。对齐且计算量足够大的 batched I8×I8 使用 signed-I8 cuBLAS GEMM 并精确累加到 I32；仅推理使用的 F16/BF16×I8 prefill 在设备端逐行动态量化激活后执行 INT8 GEMM，fused QKV/GateUp 会复用量化后的激活。训练路径保留直接 F16/BF16×I8 前向，使 F32 backward 对应同一个前向函数。Mixed-I8 prefill 仍明显慢于原生 F16/BF16 GEMM，但 I8+BF16 在本轮短真实模型推理中拥有最高 end-to-end decode throughput。

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
