# Lumen

> 一个使用 Rust 编写的轻量级深度学习核心，包含动态自动微分、模块化神经网络组件，以及面向 CPU 的 Llama 推理路径。

[English README](./README_EN.md) · [仓库首页 README](./README.md)

---

## 项目简介

Lumen 是一个用 Rust 编写的紧凑型深度学习项目。当前这个发布版保留了**核心库代码**与**一个最小可运行的 Llama 推理示例**，并移除了开发阶段使用的测试、基准、计时和扫参代码。

这个仓库适合两类用途：

- 作为一个 **学习型 DL Core**，用于理解 Rust 中的张量、自动微分、层、模块和优化器；
- 作为一个 **小型 LLM 推理骨架**，展示 Llama 风格模型、safetensors 权重加载、tokenizer 接入，以及基于 KV cache 的增量解码。

当前这个补丁分支还额外包含了一套 **实验性的 pure BF16 推理加载路径**。它目前更适合作为**务实的测试版 / 过渡方案**来看待，而不是已经定型的最终设计；但既然已经实打实接入了可运行示例，那 README 里也应该正经写清楚。

> `src/main.rs` **只是一个简单示例程序**。它的作用是演示如何把库串起来跑通本地推理，而**不是**完整的生产级 CLI、服务框架，或通用模型启动器。

---

## 技术特点

- **动态自动微分引擎**：基于张量构建计算图并执行反向传播
- **`Module` 风格抽象**：方便组织可训练模块与网络结构
- **层 / 算子分层设计**：上层建模和底层计算实现解耦，便于演进
- **Llama 系列解码器实现**，包括：
  - RMSNorm
  - RoPE
  - 因果自注意力
  - GQA（`num_key_value_heads`）
  - SwiGLU 风格 MLP
  - 基于 **KV Cache** 的增量解码
- **面向 CPU 的推理热路径优化**，包括：
  - 基于 Rayon 的并行计算
  - 面向 decode 场景的 row-major 并行 matvec 路径
  - 面向多 token prefill 的 BF16 优化路径（实验性）
  - MLP 中推理态 fused gate/up/SiLU 路径
  - `release` 配置启用 `lto`、`panic = "abort"`、`strip`
- 通过 **`safetensors` + `memmap2`** 进行高效权重加载
- 为 Llama 大矩阵提供 **实验性的 pure BF16 直接推理加载路径**
- 提供带 KV cache 复用、采样参数和 `/reset` 命令的交互式示例入口
- 通过 Hugging Face **`tokenizers`** 接入 `tokenizer.json`

---

## 实验性 Pure BF16 推理路径

当前仓库包含一套 **实验性的 pure BF16 推理路径**，主要面向本地测试、调优和继续迭代。它的大体思路是：

- 保持项目整体结构尽量轻量；
- 让部分大矩阵走 BF16 定向的推理加载路径；
- 在维持 CPU-first 设计的前提下，尽量减少实际推理中的内存流量。

在当前这个补丁分支里，这套 BF16 路径主要和下面这些点绑定：

- 示例入口中的 `set_pure_bf16_infer_loading(true)`；
- `ModelLoader::load_llama_weights_direct(...)` 这条直接加载路径；
- loader 侧对大矩阵的识别，包括 embedding、LM head、注意力投影层、MLP 投影层等；
- 围绕 CPU 上多 token prefill 的 BF16 定向优化。

需要说明的是，这套实现**暂时还不应该被描述成稳定、最终、对所有场景都最优的方案**。更准确地说，它是一个**已经具备实际价值的测试版 / 过渡实现**，后续仍然可以继续在 kernel、layout 和 loader 规则上演进。

## 仓库结构

```text
src/
├─ autograd.rs          # Tensor 与动态自动微分核心
├─ module.rs            # Module trait / 宏
├─ ops/                 # 张量算子与底层 kernel
├─ layers/              # 神经网络层与注意力组件
├─ models/llama.rs      # Llama 模型实现
├─ loader.rs            # Safetensors 权重加载
├─ tokenizer.rs         # Hugging Face tokenizer 封装
├─ kv_cache.rs          # 旧版 / 简化 KV cache 实现
├─ optim.rs             # 优化器
├─ loss.rs              # 损失函数
├─ init.rs              # 参数初始化
└─ main.rs              # 本地推理最小示例入口
```

---

## 构建

建议使用 release 模式构建：

```bash
cargo build --release
```

为了更好利用本机 CPU 指令集，可以额外启用：

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

PowerShell：

```powershell
$env:RUSTFLAGS = "-C target-cpu=native"
cargo build --release
```

---

## 最小示例运行方式

示例程序需要显式提供权重文件和 tokenizer：

```bash
cargo run --release -- \
  --weights path/to/model.safetensors \
  --tokenizer path/to/tokenizer.json
```

可选参数：

- `--system`
- `--temperature`
- `--top-p`
- `--repetition-penalty`
- `--recent-window`
- `--max-gen`

当前示例入口会在加载权重前显式启用实验性 BF16 加载路径：

```rust
set_pure_bf16_infer_loading(true);
ModelLoader::load_llama_weights_direct(&args.weights, &mut model)?;
```

示例聊天循环支持的命令：

- `/reset` —— 清空对话状态与 KV cache
- `/exit` —— 退出程序

---

## 关于 `main.rs` 的重要说明

`src/main.rs` 使用了**硬编码的 `model_config()`** 和一个非常轻量的 CLI 流程。这是有意为之：它更容易读，也更适合作为示例。

但这也意味着：

- `model_config()` 中的模型结构必须与你加载的权重匹配；
- tokenizer 的词表与特殊 token 需要和 `vocab_size`、提示词模板兼容；
- 当前 BF16 加载路径带有明显补丁分支特征，并依赖 loader 正确识别目标大矩阵；
- 若要适配其他 checkpoint，通常需要调整维度、层数、注意力头配置、特殊 token、prompt template，必要时还要调整 BF16 loader 的匹配规则。

也就是说，**`main.rs` 是一个简单集成示例，不是通用启动器。**

---

## 这个项目的价值

很多 Rust 机器学习项目只做到张量层、MLP 层，或者停留在玩具示例；而 Lumen 把下面这些部分串在了一起：

- 张量与 autograd 基础能力
- 可复用神经网络模块
- Llama 解码器结构
- checkpoint 权重加载
- tokenizer 桥接
- 有状态自回归解码

因此，它不仅适合阅读，也适合作为一个小型 Rust 原生 DL / LLM runtime 的起点。

---

## License

GPL v3.0
