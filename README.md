# 💡 Lumen: Illuminating Deep Learning in Pure Rust

**Lumen** 是一个轻量级、高性能的深度学习训练与推理框架，完全使用 Rust 编写。

它的设计理念借鉴 PyTorch，提供动态计算图（Dynamic Computational Graph）和模块化的 API 设计。**Lumen** 不仅支持从零构建 CNN、RNN 和 MLP 进行训练，其内置的高性能算子和显存优化策略更使其能够高效运行现代 LLM（如 Llama 架构），在 CPU 上实现优秀的推理速度。

---

## ✨ 核心特性 (Key Features)

### 🧠 核心引擎

* **Dynamic Autograd**: 实现了动态自动微分引擎，支持标量与张量级别的反向传播（Define-by-Run）。
* **PyTorch-like API**: 采用 `Module` trait 设计，层（Layer）与模型（Model）的组合方式与 PyTorch 直觉一致。
* **Optimizers & Loss**: 内置 SGD、Adam 等优化器及 CrossEntropy、MSE 等损失函数，支持完整的训练闭环。

### 🚀 优秀性能 (High Performance)

* **Static KV Cache**: 针对自回归模型（如 Llama）实现了**静态 KV Cache 预分配**策略，推理过程中 **0 动态内存分配**，彻底消除内存碎片。
* **Decoding Optimization**: 针对 `Batch=1` 的 Decoding 阶段，实现了手写并行矩阵-向量乘法（Vector-Matrix Multiplication），突破 BLAS 库在小矩阵上的性能瓶颈。
* **Zero-Copy Design**: 广泛使用内部可变性模式（Interior Mutability）和视图切片，最大程度减少张量数据的内存拷贝。
* **Rayon Parallelism**: 对 GQA Attention、Softmax、RMSNorm 和 Convolution 进行了细粒度的多线程并行优化。

### 🏗️ 丰富的模型支持

* **Transformers**: 完整支持 **Llama** 架构，包含 **RoPE**（旋转位置编码）、**GQA**（分组查询注意力）和 **SwiGLU**等。
* **RNN Family**: 原生支持 **RNN**、**LSTM**、**GRU**，适用于序列建模。
* **CNN Support**: 支持卷积操作（Im2Col + GEMM），适用于计算机视觉任务。

---

## 🛠️ 快速上手 (Getting Started)

**Lumen** 包含了一个极速的 Llama 推理实现。

#### 准备工作

1. 下载 **TinyLlama-1.1B-Chat-v1.0** (或其他 Llama 模型) 的 `model.safetensors` 和 `tokenizer.json`。
2. 在 `src/main.rs` 中配置路径：
```rust
let tokenizer_path = r"path/to/tokenizer.json";
let weight_path = r"path/to/model.safetensors";

```

3. (可选) 在 `main.rs` 或 `config` 中调整模型参数（如 `rope_theta`, `hidden_size`）。

#### 运行

为了获得最佳性能，**必须**使用 Release 模式并开启 CPU 原生指令集。

**Linux / macOS:**

```bash
# 开启 AVX2/AVX-512 优化
RUSTFLAGS="-C target-cpu=native" cargo run --release

```

**Windows (PowerShell):**

```powershell
$env:RUSTFLAGS = "-C target-cpu=native"
cargo run --release

```

---

## ⚡ 性能优化细节

Lumen 在 CPU 推理上的高性能源于以下细节：

1. **Static Memory Allocation**: 这里的 `LlamaKVCache` 并非简单的 `Vec`，而是基于预先计算的 `max_seq_len` 分配的一整块连续内存。推理时仅移动指针，无任何 `malloc` 开销。
2. **Vector-Matrix Speculation**: 标准 BLAS 库在矩阵维度 `M=1` (即 Decoding 阶段) 时往往无法利用多核优势。Lumen 在 `ops/matmul.rs` 中检测到 `M=1` 时，会自动切换到基于 Rayon 的手动切块并行模式，大幅提升 CPU 利用率。
3. **Parallel GQA**: 多头注意力（Multi-Head Attention）的计算在 Head 维度被完全并行化。

---

## 📄 License

GPL v3.0 License.
