# Lumen

一个 Rust 神经网络库，提供动态自动微分、通用神经网络层、支持多种精度的
CPU/CUDA 执行，以及内置 Llama 和稠密 Qwen3 解码器。

[English](./README.md) · [更新日志](./CHANGELOG.md) · [许可证](./LICENSE)

[功能](#功能) · [快速开始](#快速开始) · [库 API 与用法](#库-api-与用法) · [精度与训练](#精度与训练) ·
[CPU 调度](#cpu-调度) · [模型推理](#模型推理) · [基准测试](#基准测试) ·
[测试](#测试) · [仓库结构](#仓库结构)

## 功能

- Tensor 运算与动态自动微分，使用 F32 梯度。
- Linear、Embedding、Dropout、Conv2D、MaxPool2D、RNN、GRU 和 LSTM。
- LayerNorm、RMSNorm，以及支持 MHA/GQA/MQA 和交叉注意力的通用注意力算子。
- SGD、momentum 和 Adam，可为低精度训练启用 F32 主权重。
- F32、F16、BF16、I8 存储，safetensors 加载、流式读取与离线 I8 量化。
- 便携 CPU 实现、可选 x86/ARM 内核和由调用方控制的 Rayon 调度。
- 可选 CUDA 内核、cuBLAS 和 cuDNN，支持设备常驻张量与梯度路径。
- 带 RoPE、GQA、greedy/采样解码和 KV cache 的 Llama 与稠密 Qwen3 推理。

Tensor 和层 API 独立于内置模型。CLI 从权重配套的 `config.json` 读取架构和网络尺寸。

## 快速开始

### 从源码构建

```bash
git clone https://github.com/chen-197/Lumen.git
cd Lumen
cargo build --release
```

使用较新的 stable Rust 工具链（edition 2024）。CPU 构建不依赖 CUDA，也可以通过 Git 引入库：

```toml
[dependencies]
lumen = { git = "https://github.com/chen-197/Lumen.git" }
ndarray = "0.15"
rayon = "1.7"
```

### 训练一个小模型

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

`matmul` 的语义是 `A[..., K] @ B[N, K]ᵀ`，输出为 `[..., N]`。
参数更新后应重新执行前向并构建计算图，并在下一批次前调用 `zero_grad()`。
多次反向传播会累积叶子梯度。
需要自定义上游梯度时，使用 `backward_with_grad(seed)`。

### 通用注意力与归一化

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

每个维度按下面的顺序排列：

| 张量 | 维度顺序 | 上例的实际形状 |
| --- | --- | --- |
| Q（查询） | `[每批序列数, 查询头数, 查询序列长度, 每头查询特征数]` | `[2, 4, 3, 5]` |
| K（键） | `[每批序列数, 键值头数, 键值序列长度, 每头键特征数]` | `[2, 2, 7, 5]` |
| V（值） | `[每批序列数, 键值头数, 键值序列长度, 每头值特征数]` | `[2, 2, 7, 6]` |
| 注意力输出 | `[每批序列数, 查询头数, 查询序列长度, 每头值特征数]` | `[2, 4, 3, 6]` |

上例同时处理 2 条序列，每条序列用 3 个查询位置关注 7 个键值位置。
4 个查询头共享 2 个键值头，即每两个查询头共用一组 K/V 头。
查询和键的每头特征数必须相同；值的每头特征数可以不同，并决定输出的最后一维。
查询头数必须是键值头数的整数倍。

布尔遮罩中 `true` 表示允许关注。遮罩广播到
`[每批序列数, 查询头数, 查询序列长度, 键值序列长度]`；
上例 `[2, 1, 1, 7]` 的遮罩为每条序列指定 7 个键位的可见性，供所有查询头和查询位置共享。
加性遮罩在分数缩放后加入；完全被遮挡的行返回零上下文和零分数梯度。
`AttentionOptions` 提供因果遮罩与查询位置偏移；投影、位置编码和缓存更新由调用方负责。

## 库 API 与用法

层的 `forward`、`parameters` 等方法需要导入 `lumen::module::Module`；
优化器的 `step`、`zero_grad` 需要导入 `lumen::optim::Optimizer`。
下列示例使用快速开始中的依赖。完整签名可运行 `cargo doc --no-deps --open` 查看；
这里按使用场景列出主要公开接口。

[Tensor/求导](#tensor-与自动求导) · [算子](#算子与形状约定) · [训练/保存](#组合网络选择损失和保存参数) ·
[自定义网络](#自定义-module) · [网络层](#可用网络层) · [设备](#cpucuda-放置)

| 模块 | 提供的功能 |
| --- | --- |
| [`autograd`](./src/autograd/) | `Tensor`、`Device`、计算图、梯度、存储与设备迁移 |
| [`ops`](./src/ops/) | 逐元素算术、形状操作、矩阵乘、卷积、池化与融合算子 |
| [`layers`](./src/layers/) | 通用层、激活函数、归一化与注意力 |
| [`module`](./src/module.rs) | `Module`、`Sequential`、`sequential!`、检查点与参数转换 |
| [`loss`](./src/loss.rs)、[`optim`](./src/optim/) | MSE、稠密目标交叉熵、SGD/momentum、Adam |
| [`precision`](./src/precision.rs)、[`init`](./src/init.rs) | 精度/量化策略、参数初始化与作用域配置 |
| [`models`](./src/models/)、[`loader`](./src/loader.rs)、[`tokenizer`](./src/tokenizer.rs) | 解码器配置、权重加载、分词与对话格式 |
| [`cpu`](./src/cpu/) | 调用方线程池和任务预算 |

### Tensor 与自动求导

| API | 用法 |
| --- | --- |
| `Tensor::new(array)`、`from_data(array)` | F32 叶子张量；在 `no_grad` 外默认跟踪梯度 |
| `new_with_dtype(array, dtype)` | 显式指定 F32/F16/BF16/I8 存储 |
| `from_array_no_grad(array)` | 从拥有所有权的 `ndarray::ArrayD<f32>` 创建常量或输入 |
| `from_data_no_grad(shared)` | 从 `ndarray::ArcArray<f32, IxDyn>` 创建常量 |
| `from_data_with_grad_flag(array, flag)` | 显式选择叶子是否接收梯度 |
| `parameter(array)`、`parameter_with_dtype(array, dtype)` | 可训练参数；前者遵循默认参数策略 |
| `parameter_with_quantization(array, policy)` | 使用显式量化策略创建参数 |
| `shape_vec()`、`ndim()`、`len()`、`is_empty()`、`dtype()`、`device()`、`requires_grad()` | 查询元信息 |
| `data()`、`data_arc()`、`data_ref()`、`data_mut()` | F32 数据的独立副本、共享数组、只读借用及可写借用 |
| `backward()` | 使用全 1 上游梯度；非标量输出相当于对元素之和求导 |
| `backward_with_grad(seed)` | 自定义向量-Jacobian 乘积，seed 形状必须与输出完全相同 |
| `grad()`、`grad_arc()`、`grad_ref()`、`zero_grad()` | 读取或清除 F32 梯度 |
| `detach()`、`no_grad(\|\| ...)`、`NoGradGuard::enter()` | 独立的不跟踪副本，或在作用域内关闭构图 |
| `cast_inplace(dtype)`、`quantize_inplace_with_quantization(policy)`、`dequantize_inplace(dtype)` | 转换存储；`quantization_scale()` 查询 I8 scale |
| `export_raw()`、`import_raw(shape, dtype, raw)` | 导出/恢复原生类型数据和 I8 scale |

`Tensor::clone()` 共享句柄，参数更新和梯度也共享；`detach()` 会复制存储。
F32 数据读取会物化数据、可能发生设备传输，也可能把原 Tensor 的存储提升为 F32；
这尤其影响激活和未允许保留类型副本的参数。只想检查数值且保留原存储时，使用
`x.detach().data()` 读取独立副本，或用 `export_raw()` 导出原生类型数据。
持有 `data_ref`/`data_mut` 借用期间，应先释放借用再执行会访问或修改同一 Tensor 的操作。
叶子梯度会累积，每个 batch 前需要清零。保留的计算图使用原先前向值，
参数更新后必须重新前向构图。

```rust
use lumen::autograd::Tensor;
use lumen::ops::{matmul::{batch_matmul, matmul}, shape::cat};
use ndarray::{ArrayD, IxDyn, arr1, arr2};

fn main() {
    let x = Tensor::new(arr2(&[[1.0, 2.0], [3.0, 4.0]]).into_dyn());
    let y = &x * &x;
    y.backward_with_grad(ArrayD::ones(IxDyn(&[2, 2])));
    assert_eq!(x.grad().unwrap(), arr2(&[[2.0, 4.0], [6.0, 8.0]]).into_dyn());
    x.zero_grad();

    let bias = Tensor::from_array_no_grad(arr1(&[0.5, 1.0]).into_dyn());
    let shifted = &x + &bias;
    let flat = shifted.reshape(vec![1, -1]);
    let joined = cat(&[flat.clone(), flat], 0);
    assert_eq!(joined.shape_vec(), vec![2, 4]);

    let weight = Tensor::parameter(arr2(&[[1.0, 0.0], [0.0, 1.0]]).into_dyn());
    assert_eq!(matmul(&x, &weight).shape_vec(), vec![2, 2]);

    let a = Tensor::from_array_no_grad(ArrayD::ones(IxDyn(&[1, 2, 3, 4])));
    let b = Tensor::from_array_no_grad(ArrayD::ones(IxDyn(&[1, 2, 4, 5])));
    assert_eq!(batch_matmul(&a, &b).shape_vec(), vec![1, 2, 3, 5]);
}
```

### 算子与形状约定

| API | 输入输出约定 |
| --- | --- |
| `&a + &b`、`&a - &b`、`&a * &b` | 支持兼容广播的逐元素运算，操作数均为 Tensor |
| `x.sum()` / `ops::arithmetic::sum(&x)` | 对全部元素求和，得到标量 |
| `x.reshape(vec![...])` / `ops::shape::reshape` | 保持元素数；已知维度乘积非零时可用一个 `-1` 推导维度 |
| `x.permute(axes)`、`x.transpose(i, j)` | 重排维度 |
| `ops::shape::cat(&tensors, axis)` | 沿指定轴拼接 |
| `ops::shape::slice_last_dim(&x, start, end)` | 对最后一维取 `[start, end)` |
| `ops::matmul::matmul(&a, &weights)` | `[..., K] × [N, K]ᵀ → [..., N]`，权重按输出通道排列 |
| `ops::matmul::batch_matmul(&a, &b)` | `[B,H,M,K] × [B,H,K,N] → [B,H,M,N]`，右侧已按乘法次序排列 |
| `ops::convolution::{conv2d, max_pool2d}` | NCHW 卷积和池化，也可使用更方便的层构造器 |
| `ops::fused::fused_softmax(&x, scale, causal)` | 缩放 softmax，可选因果 mask |

有跟踪梯度的输入参与时，Tensor 算子构建计算图；推理使用 `no_grad`。
支持的空矩阵轴和空归约保留其逻辑形状，算子入口会检查形状与设备要求。
需要裸数组投影、MatVec、argmax 时可查看 `ops::matmul`；
`ops::fused` 的 `fused_gate_up_silu_infer*`、`fused_qkv_*` 系列是专用推理入口。

### 组合网络、选择损失和保存参数

`Sequential::new(Vec<Box<dyn Module>>)` 和 `sequential![...]` 串联普通单输入层。
Sequential 的 `train_mode()` / `eval_mode()` 会向子层传播，包括切换 Dropout；
单独调用 `no_grad` **不会**关闭 Dropout。普通模块的模式切换也会改变调用线程的
推理/构图模式。推理时同时设置模型评估模式，并使用 `no_grad` 作用域。

| API | 约定 |
| --- | --- |
| `MSELoss::apply(&prediction, &target)` | 形状相同且非空，返回所有元素的均方误差 |
| `CrossEntropyLoss::apply(&logits, &targets)` | `[B,C]` 原始 logits 和同形状稠密 one-hot/软目标，按 batch 求平均 |
| `SGD::new_with_dtype(params, lr, state_dtype)` | SGD；`.with_momentum(value)` 开启动量 |
| `Adam::new_with_dtype(params, lr, state_dtype)` | Adam，默认 betas 为 `(0.9, 0.999)`、epsilon 为 `1e-8` |
| `Optimizer::{zero_grad, step, params}` | 清梯度、更新参数和查询参数句柄 |
| `Module::{save, load}` | Lumen 二进制参数检查点，需相同网络结构与参数顺序 |
| `Module::{cast_parameters, quantize_parameters, apply_parameter_quantization}` | 分别用于存储转换、I8 量化及按策略量化 |

交叉熵接收**稠密目标**，不是整数类别索引向量；直接传 logits，不需要先做 Softmax。
优化器的 `new(...)` 构造器使用默认 runtime dtype 作为状态类型；
需要明确的 F32 状态时使用 `new_with_dtype(..., DType::F32)`。

```rust
use lumen::autograd::{Tensor, no_grad};
use lumen::layers::{Dropout, Linear, SiLU};
use lumen::loss::CrossEntropyLoss;
use lumen::module::Module;
use lumen::optim::{Adam, Optimizer};
use lumen::precision::DType;
use ndarray::arr2;

fn main() -> std::io::Result<()> {
    let mut model = lumen::sequential![
        Linear::new(2, 8), SiLU::new(), Dropout::new(0.1), Linear::new(8, 2),
    ];
    let input = Tensor::from_array_no_grad(arr2(&[[1.0, 0.0], [0.0, 1.0]]).into_dyn());
    let targets = Tensor::from_array_no_grad(arr2(&[[1.0, 0.0], [0.0, 1.0]]).into_dyn());
    model.train_mode();
    let mut optimizer = Adam::new_with_dtype(model.parameters(), 0.01, DType::F32);
    for _ in 0..100 {
        optimizer.zero_grad();
        let logits = model.forward(input.clone());
        let loss = CrossEntropyLoss::apply(&logits, &targets);
        loss.backward();
        optimizer.step();
    }
    model.eval_mode();
    let logits = no_grad(|| model.forward(input));
    assert_eq!(logits.shape_vec(), vec![2, 2]);
    model.save("network.bin")?;
    model.load("network.bin")?;
    Ok(())
}
```

检查点包含参数形状、类型、数据及 I8 scale，不保存网络定义、优化器状态、RNG 或 KV。
恢复时先重建相同结构，再加载参数，随后新建优化器。
此二进制格式与 safetensors 不同，后者由 `ModelLoader` 加载。

### 自定义 Module

实现 `forward` 和 `parameters`，后者返回前向实际使用的可训练 Tensor 句柄，
优化器才能更新对应参数。自定义权重共享时，每个共享参数只返回一次。
如果自定义容器持有 Dropout 等有状态子层，还需实现模式的递归传播。

```rust
use lumen::autograd::Tensor;
use lumen::layers::{Gelu, Linear};
use lumen::module::Module;
use ndarray::arr2;

struct Mlp { hidden: Linear, output: Linear }

impl Module for Mlp {
    fn forward(&self, input: Tensor) -> Tensor {
        self.output.forward(Gelu::new().forward(self.hidden.forward(input)))
    }
    fn parameters(&self) -> Vec<Tensor> {
        let mut params = self.hidden.parameters();
        params.extend(self.output.parameters());
        params
    }
}

fn main() {
    let model = Mlp { hidden: Linear::new(2, 8), output: Linear::new(8, 1) };
    let input = Tensor::from_array_no_grad(arr2(&[[1.0, 2.0]]).into_dyn());
    model.forward(input).sum().backward();
    assert!(model.parameters().iter().all(|p| p.grad().is_some()));
}
```

### 可用网络层

多数带参数的层提供 `new_with_dtype(...)`；Linear 另有
`new_no_bias(...)` 和 `new_no_bias_with_dtype(...)`。

| 层 / 构造器 | 形状或行为 |
| --- | --- |
| `Linear::new(input, output)` | `[..., input] → [..., output]`，可选 bias |
| `Embedding::new(vocab, dim)` | `ids[...] → [..., dim]`，调用 `forward(&ids)` |
| `Dropout::new(p)`、`new_with_seed(p, seed)` | inverted dropout，`0 ≤ p < 1`；固定 seed 的每次调用重现相同 mask |
| `Flatten::new()` | 保留第一维/batch，其余轴展平 |
| `activation::{ReLU, Sigmoid, Tanh, SiLU, Gelu}::new()` | 逐元素激活 |
| `activation::Softmax::new(axis)` | 沿指定轴归一化 |
| `Conv2D::new(in_channels, out_channels, kernel, stride, padding)` | NCHW 方形卷积核 |
| `MaxPool2D::new(kernel, stride)` | NCHW 最大池化 |
| `RNN::new(input, hidden)`、`GRU::new(input, hidden)` | `forward_step(&x, &h) → h_next`，输入分别为 `[B,input]`、`[B,hidden]` |
| `LSTM::new(input, hidden)` | `forward_step(&x, &h, &c) → (h_next, c_next)` |
| `LayerNorm::new(shape, eps)` | 归一化末尾形状；`with_options` 选择 affine/bias/参数类型 |
| `RMSNorm::new(dim, eps)` | 归一化最后一维，`new_with_dtype` 选择参数存储 |
| `scaled_dot_product_attention(...)` | 已投影 Q/K/V 的 MHA/GQA/MQA/交叉注意力，示例见快速开始 |
| `SelfAttention::new(embed, heads, kv_heads, max_seq, theta, causal)` | `forward(x, Option<KVCache>) → (output, cache)`；`new_with_options` 可指定独立 head_dim 和 Q/K norm |
| `RotaryEmbedding::new(dim, max_seq, theta)` | `forward(&x, offset)`，输入 `[每批序列数, 注意力头数, 序列长度, 每头特征数]`，每头特征数为偶数 |
| `KVCache` | 逐层共享缓存句柄；预训练模型用 `init_kv_caches` 创建并显式 reset |

ReLU/Sigmoid/Tanh/Softmax 从 `lumen::layers::activation` 导入，
SiLU、Gelu 也在 `lumen::layers` 重导出。
循环层必须使用 `forward_step`：其 `Module::forward` 不是序列执行器，
不能直接放入 Sequential。由调用方逐步遍历序列；需要截断反向传播时，
在分段之间对隐藏状态调用 `detach()`。
Embedding/模型 token ID 使用表示整数的 F32 Tensor，不要把 ID 量化成 I8。

```rust
use lumen::autograd::Tensor;
use lumen::layers::{Conv2D, Embedding, Flatten, GRU, LSTM, MaxPool2D, RNN};
use lumen::module::Module;
use ndarray::{ArrayD, IxDyn, arr2};

fn main() {
    let image = Tensor::from_array_no_grad(ArrayD::zeros(IxDyn(&[2, 3, 8, 8])));
    let features = Conv2D::new(3, 4, 3, 1, 1).forward(image);
    let pooled = MaxPool2D::new(2, 2).forward(features);
    assert_eq!(Flatten::new().forward(pooled).shape_vec(), vec![2, 64]);

    let ids = Tensor::from_array_no_grad(arr2(&[[1.0, 2.0]]).into_dyn());
    assert_eq!(Embedding::new(16, 5).forward(&ids).shape_vec(), vec![1, 2, 5]);

    let x = Tensor::from_array_no_grad(ArrayD::zeros(IxDyn(&[2, 5])));
    let h = Tensor::from_array_no_grad(ArrayD::zeros(IxDyn(&[2, 3])));
    let c = h.detach();
    assert_eq!(RNN::new(5, 3).forward_step(&x, &h).shape_vec(), vec![2, 3]);
    assert_eq!(GRU::new(5, 3).forward_step(&x, &h).shape_vec(), vec![2, 3]);
    let (next_h, next_c) = LSTM::new(5, 3).forward_step(&x, &h, &c);
    assert_eq!(next_h.shape_vec(), next_c.shape_vec());
}
```

### CPU/CUDA 放置

| API | 行为 |
| --- | --- |
| `tensor.to_device(Device::Cpu / Device::Cuda)` | 返回独立 Tensor/存储，保留叶子的梯度标志，不保留原反向图 |
| `to_cpu()`、`to_cuda()`、`cpu()`、`cuda()` | Tensor 设备副本的简写 |
| `to_device_inplace(device)`、`to_cpu_inplace()`、`to_cuda_inplace()` | 迁移现有 Tensor 句柄 |
| `module.to_device(device)`、`to_cpu()`、`to_cuda()` | 迁移模块参数 |
| `ops::cuda::is_available()` | 检查当前构建及机器是否提供 CUDA |
| `set_strict_device_execution_scoped(true)` | 拒绝需要不受支持的 CPU 回退的操作 |

在构图和创建优化器状态前迁移参数与输入，操作数必须位于同一设备。
`data()`、`grad()` 等主机读取可能同步 CUDA 并传回数据。
CUDA 当前面向单设备，不同算子/类型的覆盖与 CPU 存在差异。

```rust
use lumen::autograd::{Device, Tensor, no_grad, set_strict_device_execution_scoped};
use lumen::layers::Linear;
use lumen::module::Module;
use lumen::ops::cuda;
use ndarray::arr2;

fn main() {
    if !cuda::is_available() {
        println!("CUDA is unavailable; build with --features cuda on a CUDA host.");
        return;
    }
    let layer = Linear::new(2, 1);
    layer.to_device(Device::Cuda);
    let x = Tensor::from_array_no_grad(arr2(&[[1.0, 2.0]]).into_dyn())
        .to_device(Device::Cuda);
    let _strict = set_strict_device_execution_scoped(true);
    let y = no_grad(|| layer.forward(x));
    assert_eq!(y.device(), Device::Cuda);
    println!("{:?}", y.to_cpu().data());
}
```

## 构建选项

```bash
# CUDA 后端
cargo build --release --features cuda

# x86 内核与基准工具
cargo build --release --features "dev-tools x86-fp-kernels x86-int8-kernels"

# 为当前机器生成优化代码
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

PowerShell 中先设置 `$env:RUSTFLAGS = "-C target-cpu=native"`。
CUDA 构建需要 CUDA toolkit 和兼容的 NVIDIA 驱动；`build.rs` 从环境变量、
`nvcc` 与系统安装路径查找 CUDA/cuDNN。

| Feature | 用途 |
|---|---|
| `cuda` | CUDA C++ 后端、cuBLAS 与可选 cuDNN |
| `dev-tools` | 基准与诊断程序 |
| `cpu-profiling` | 算子计时插桩，包含 `dev-tools` |
| `x86-fp-kernels` | stable x86 F32/F16/BF16 内核 |
| `x86-fp-kernels-nightly` | 真 AVX-512 FP16 intrinsics，需要 nightly |
| `x86-int8-kernels` | x86 I8 内核 |
| `arm64-fp-kernels` | ARM64 浮点内核 |
| `arm64-int8-kernels` | ARM64 I8 内核 |

架构内核按 feature 开启，再由运行时检测选择受支持的实现。
真 AVX-512 FP16 可使用
`cargo +nightly build --release --features x86-fp-kernels-nightly` 构建。
正式性能比较应使用未插桩版本。

## 精度与训练

- 受支持的路径允许参数、激活和 KV 使用不同精度。KV 使用浮点存储；
  I8 权重通常配合 BF16 runtime/KV。
- 梯度为 F32。构建自动微分图的算子输出 F32；同精度低精度矩阵推理保留存储类型。
  F32/F16 激活与 BF16 权重相乘时保留激活声明的值。
- 普通及批量矩阵乘支持空维度和零长度归约；冻结输入不接收算子梯度。
  Flatten 和形状操作保留空输出形状及其对应的输入梯度形状。
- `no_grad` 关闭建图；`detach()` 关闭跟踪并拥有独立存储。
  自定义反向种子使用 `backward_with_grad`，不要用 `add_grad` 预填。
- `backward_with_cuda_grad` 接受设备常驻的 F32 CUDA 种子。通过 `grad()`、
  `grad_arc()` 或 `grad_ref()` 读取 CUDA 梯度时，会生成并缓存主机表示；
  保持梯度计算在设备上可避免这次传输。
- CPU 优化器更新后保留梯度，成功的 CUDA 更新会消耗已应用的梯度。
  两种后端均应在每批次前调用 `zero_grad()`。SGD 和 Adam 会先校验全部梯度形状，
  再修改参数或优化器状态；形状不匹配会触发 panic。
- Module 检查点保存参数，不序列化优化器 moments、主权重残差或 RNG 状态。

低精度训练可选 F32 优化器状态，并在 SGD 或 Adam 上调用
`.with_fp32_master_weights()`；I8 参数范围需要随训练变化时，可使用
`.with_dynamic_i8_master_weights()` 逐步校准 scale。这些选项需要在训练前设置，
并增加 F32 主权重存储。外部替换参数存储、形状、类型、scale 或设备后，应重建优化器。

I8 激活量化需要显式设置 `--activation-quantization dynamic-i8`，默认为 `off`。
比较输出或性能时应使用相同策略。

### 在库中配置精度与量化

| API | 范围 |
| --- | --- |
| `precision_guard(PrecisionConfig { ... })`、`with_precision_config(config, f)` | 参数/runtime 默认精度，以及是否允许保留参数类型副本 |
| `runtime_component_dtypes_guard(Some(activation), Some(kv))`、`with_runtime_component_dtypes(...)` | 分别指定默认激活/KV 类型；`None` 不改变该组件 |
| `ParameterQuantization::new(DType::I8).with_scale(scale)` | 显式逐张量 I8 存储和手动 scale；省略 `with_scale` 则自动计算 |
| `parameter_quantization_guard(policy)`、`with_parameter_quantization(policy, f)` | 为遵循该策略的构造器设置默认量化 |
| `tensor_init(shape, InitType::...)`、`tensor_init_with_dtype(...)` | XavierUniform、KaimingNormal、Zeros 或 Ones 参数初始化 |
| `with_parameter_init_mode(ParameterInitMode::Placeholder, f)` | 即将加载权重时跳过随机参数初始化 |

默认设置影响使用该策略的构造器/执行路径，不会自动转换已存在的模型；
显式类型构造器优先。作用域 guard 在销毁时恢复配置，应覆盖模型构建和执行过程。

```rust
use lumen::autograd::Tensor;
use lumen::layers::Linear;
use lumen::loss::MSELoss;
use lumen::module::Module;
use lumen::optim::{Adam, Optimizer};
use lumen::precision::{DType, ParameterQuantization, PrecisionConfig, precision_guard};
use ndarray::arr2;

fn main() {
    let _precision = precision_guard(PrecisionConfig {
        parameter_dtype: DType::BF16,
        runtime_dtype: DType::F32,
        allow_parameter_dtype_copies: false,
    });
    let layer = Linear::new(2, 1);
    let mut optimizer = Adam::new_with_dtype(layer.parameters(), 0.01, DType::F32)
        .with_fp32_master_weights();
    let x = Tensor::from_array_no_grad(arr2(&[[1.0, 2.0]]).into_dyn());
    let target = Tensor::from_array_no_grad(arr2(&[[1.0]]).into_dyn());
    optimizer.zero_grad();
    MSELoss::apply(&layer.forward(x), &target).backward();
    optimizer.step();
    assert_eq!(layer.weight.dtype(), DType::BF16);

    let quantized = Tensor::parameter_with_quantization(
        arr2(&[[0.5, -0.5]]).into_dyn(),
        ParameterQuantization::new(DType::I8).with_scale(0.01),
    );
    assert_eq!(quantized.dtype(), DType::I8);
    assert_eq!(quantized.quantization_scale(), Some(0.01));
}
```

## CPU 调度

```rust
use lumen::autograd::{Tensor, no_grad};
use lumen::cpu::{CpuInferenceContext, CpuKernelContext};
use lumen::ops::matmul::matmul;
use ndarray::arr2;
use std::{num::NonZeroUsize, sync::Arc};

fn main() {
    let pool = Arc::new(rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap());
    let context = CpuKernelContext::with_pool(pool)
        .with_task_limit(NonZeroUsize::new(2).unwrap());
    let input = Tensor::from_array_no_grad(arr2(&[[1.0, 2.0]]).into_dyn());
    let weight = Tensor::parameter(arr2(&[[0.5, 0.5]]).into_dyn());
    context.scope(|| matmul(&input, &weight).sum().backward());
    assert!(weight.grad().is_some());

    let inference = CpuInferenceContext::new(context)
        .with_prefill_task_limit(NonZeroUsize::new(2).unwrap())
        .with_decode_task_limit(NonZeroUsize::new(1).unwrap());
    let prefill = inference.prefill(|| no_grad(|| matmul(&input, &weight)));
    let decode = inference.decode(|| no_grad(|| matmul(&input, &weight)));
    assert_eq!(prefill.data(), decode.data());
}
```

在计算循环外创建线程池，并让多个调用方复用。`scope` 保持 Tensor 和
自动微分状态位于调用线程，仅将数组计算派发到池。预算约束单个已接入内核，
不是进程线程总数；小任务和嵌套数组计算串行执行。库不配置全局线程池、不绑定
线程，也不根据某颗 CPU 的型号固定调度。

MatVec、普通/批量矩阵乘前向与反向、融合投影、argmax、prefill、推理注意力及
原生低精度优化器状态内核使用这些上下文。**前向和反向都应进入作用域**；
仅在其中构建计算图，不会将策略绑定到之后的反向遍历。
`CpuInferenceContext` 可为共享池配置独立的 prefill/decode 预算。
Tensor 句柄使用 `Rc<RefCell<_>>`，不是 `Send`/`Sync`；不同调用方各自持有
模型和 Tensor 句柄，共享的是数组计算线程池。
CPU BF16 prefill 通过临时权重分块复用多行输入，长 prompt 同样使用分块路径。
支持相应指令的 CPU 使用
BF16 点积和 F32 累加；混合 F32/F16 激活保留原精度，配合精确权重展开及
AVX2/FMA 或 AVX-512 计算。其他 CPU 使用通用回退路径，公开 API 保持一致。
no-grad、精度、初始化和设备选择 guard 同样不是 `Send`/`Sync`。
应在执行计算的线程内创建和销毁这些作用域，以正确恢复线程局部设置。

## 模型推理

```bash
cargo run --release --bin lumen -- \
  --weights path/to/model.safetensors \
  --tokenizer path/to/tokenizer.json
```

GPU 推理增加 `--features cuda` 和 `--device cuda`。完整精度、缓存、采样、流式读取
及 CPU 任务选项可通过 `cargo run --release --bin lumen -- --help` 查看。
参数、激活和 KV 类型必须受所选路径支持。将 `config.json` 放在权重旁，或通过
`--config PATH` 指定。支持单个 safetensors 文件和 HF safetensors 分片索引；
`--weights` 也可以传入模型目录，由程序查找权重或索引。

### Qwen3

```bash
cargo run --release --bin lumen -- \
  --weights path/to/Qwen3-0.6B/model.safetensors \
  --tokenizer path/to/Qwen3-0.6B/tokenizer.json \
  --max-seq-len 1024
```

稠密 Qwen3 使用独立头维度、逐头 Q/K RMSNorm，并按配置共享 embedding 与输出权重。
CLI 使用纯文本 ChatML，默认关闭思考模式，增加 `--enable-thinking` 可开启。
停止条件结合模型配置、生成配置及 tokenizer 标记。`prefill_decode_bench` 读取同样的
模型配置，支持自由生成和固定 token 数值诊断。
MoE、滑动窗口、缩放/部分 RoPE 及带 bias 的注意力配置会明确报错。
CLI 不解释任意 Jinja 模板、工具调用或多模态输入。
当前 CLI 对话格式覆盖 TinyLlama 和 Qwen3，其他对话模板需要相应的格式化实现。

### 在库中加载预训练模型

使用 `CausalLMConfig` 从 `config.json` 构建受支持的架构。
`Qwen3Config`/`Qwen3Model` 提供稠密 Qwen3 的具名封装，
`LlamaConfig`/`LlamaModel` 也可显式构造自定义模型。
通用配置构造器返回共享的解码器实现。

| API | 用途 |
| --- | --- |
| `CausalLMConfig::{from_file, validate, build}` | 读取/验证架构，并指定上下文容量与精度构建模型 |
| `ModelLoader::load_weights_with_options(path, &parameters, options)` | 严格加载 safetensors 文件、模型目录或 HF 分片索引 |
| `WeightLoadOptions::{stream_from_disk, float_source_quantization}` | 流式读取参数载荷，或加载时量化浮点源权重 |
| `model.parameters()` | 内置解码器去重后的可训练句柄 |
| `model.named_parameters()`、`checkpoint_parameters()` | 检查参数名；规范检查点名称省略共享输出权重的别名 |
| `ModelTokenizer::{from_file, encode, decode, decode_complete_prefix}` | 分词、解码及保留待完成 ID 的完整 UTF-8 显示前缀 |
| `ChatFormat::{first_turn, next_turn}` | TinyLlama / Qwen3 纯文本对话格式 |
| `init_kv_caches(batch)`、`init_kv_caches_with_dtype(batch, dtype)` | 分配逐层推理缓存 |
| `forward_last_argmax(ids, &mut caches, pos)` | batch=1 的 greedy 下一 token，避免物化完整 logits |
| `forward_last_argmax_batch(...)`、`forward_last_logits(...)` | 批量 token 选择，或最后位置的 `[B,1,V]` logits |
| `model.forward(ids, &mut caches, pos)` | 带缓存的 `[B,S,V]` 推理 logits |
| `forward_train(ids)` / `Module::forward` | 可微的全序列 logits，不使用推理缓存 |
| `reset_kv_caches(&mut caches)`、`cast_kv_caches(&mut caches, dtype)` | 开始新序列或转换浮点缓存类型 |

下面是 CPU 示例：先对 prompt 做一次 prefill，再逐 token decode。
程序第一个参数为模型目录，使用 256-token 上下文容量，最多生成 32 个 token；
输入与生成预算之和应位于容量内。Placeholder 模式跳过参数随机初始化，
推理前必须加载全部所需权重。

```rust
use lumen::autograd::{Tensor, no_grad};
use lumen::init::{ParameterInitMode, with_parameter_init_mode};
use lumen::loader::{ModelLoader, WeightLoadOptions};
use lumen::models::CausalLMConfig;
use lumen::precision::DType;
use lumen::tokenizer::{ChatFormat, ModelTokenizer};
use ndarray::Array2;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let directory = PathBuf::from(std::env::args().nth(1).expect("usage: example MODEL_DIR"));
    let config = CausalLMConfig::from_file(directory.join("config.json"))?;
    let model = with_parameter_init_mode(ParameterInitMode::Placeholder, || {
        config.build(256, DType::BF16, DType::BF16, DType::BF16)
    })?;
    ModelLoader::load_weights_with_options(
        &directory, &model.checkpoint_parameters(), WeightLoadOptions::default(),
    )?;
    let tokenizer = ModelTokenizer::from_file(directory.join("tokenizer.json"))?;
    let format = if config.model_type == "qwen3" {
        ChatFormat::Qwen3 { enable_thinking: false }
    } else { ChatFormat::TinyLlama };
    let text = format.first_turn("You are a helpful assistant.", "What is the capital of France?");
    let prompt = tokenizer.encode(&text, false)?;
    let input = |ids: &[usize]| Tensor::from_array_no_grad(
        Array2::from_shape_vec((1, ids.len()), ids.iter().map(|&id| id as f32).collect())
            .unwrap().into_dyn(),
    );
    let mut stop = config.eos_token_id.as_ref().map(|ids| ids.values()).unwrap_or_default();
    stop.extend(tokenizer.eos_id());
    stop.extend(tokenizer.eot_id());
    let mut caches = model.init_kv_caches(1);
    let mut next_input = input(&prompt);
    let mut generated = Vec::new();
    no_grad(|| {
        for _ in 0..32 {
            let token = model.forward_last_argmax(next_input, &mut caches, 0);
            if stop.contains(&token) { break; }
            generated.push(token);
            next_input = input(&[token]);
        }
    });
    println!("{}", tokenizer.decode(&generated, true));
    model.reset_kv_caches(&mut caches);
    Ok(())
}
```

`pos` 为兼容参数，实际位置由缓存长度决定。独立新 prompt 前先 reset；
续写时只传新 token，把完整前缀再次传入已有缓存会重复追加。
该示例使用 greedy 和模型/tokenizer 停止 ID；应用可通过 `forward_last_logits`
实现自己的采样及额外生成配置停止策略。流式展示可用 `decode_complete_prefix`
等待完整 UTF-8 字符，同时保留原始 token ID。
训练时使用新建的 `forward_train` 计算图，将序列 logits 和稠密目标整理成 `[B*S,V]`
传给 `CrossEntropyLoss`，不要复用推理 KV。

需要流式加载时传入 `WeightLoadOptions { stream_from_disk: true, ..Default::default() }`。
加载时量化可指定 `float_source_quantization: ParameterQuantization::new(DType::I8)`，
并以 I8 参数和浮点 activation/KV 构建模型。

Loader 会在加载参数载荷前检查所需名称、元信息、形状和 I8 配套 scale。
旧的 `load_llama_weights*` 入口继续保留；配置驱动的模型使用通用严格 loader。

### 离线量化

```bash
cargo run --release --bin quantize_safetensors -- \
  --input path/to/model.safetensors \
  --output path/to/model.i8.safetensors \
  --dtype i8
```

`--scale` 可指定手动 I8 scale；loader 也支持加载时量化与流式 safetensors 读取。

转换器采用逐张量对称量化，范围为 `[-127, 127]`，半整数按远离零方向舍入。
手动 scale 必须为有限正数，支持正的次正规值。每个 I8 张量都附带标量 F32
`<tensor-name>.scale` 条目，加载量化权重时需要对应的 scale。

## 基准测试

基准工具需要 `dev-tools`，测量性能请使用 `--release`。

| 程序 | 用途 |
|---|---|
| `kernel_bench` | CPU 内核微基准 |
| `generic_prefill_bench` | 矩阵乘、批量矩阵乘与通用注意力 |
| `cpu_backward_bench` | 普通及批量矩阵乘反向 |
| `cpu_projection_bench` | 投影、QKV 与门控 MLP |
| `cpu_decode_bench` | 解码工作集与词表 argmax |
| `cpu_scaling_bench` | 线程池、任务预算与并发调用 |
| `cuda_cpu_bench` | CPU/CUDA 算子与训练路径 |
| `prefill_decode_bench` | 真实权重 prefill/decode、回放与采样 |
| `manual_state_check` | 公开 API、检查点与状态检查 |

```bash
cargo run --release --features "dev-tools x86-fp-kernels x86-int8-kernels" \
  --bin generic_prefill_bench -- --threads 4 --tasks 2 --warmup 5 --samples 7 --iters 5

cargo run --release --features "dev-tools cuda x86-fp-kernels x86-int8-kernels" \
  --bin prefill_decode_bench -- --help
```

`--samples-output` 保存模型计时样本，`--replay-token-ids` 用于固定输入的解码测量。
自由生成文本应单独检查，不能用回放代替质量测试。
`cpu-profiling` 提供诊断计时，最终性能比较应排除插桩开销。

运行代表性 CPU 性能矩阵时，先构建工具，再将 release 输出目录交给 Python runner：

```bash
cargo build --release --locked --features dev-tools --bins
python dev-tools/check_performance.py --threads 8 --plan
python dev-tools/check_performance.py --threads 8 \
  --binary-directory path/to/release --output-directory target/performance-new
```

线程数大于一时，默认选择 21 个进程，每项先预热再计时。
`--select argmax` 可缩小范围，`--extended` 扩展任务预算和精度/请求组合。
每次使用新的输出目录；计时应串行执行，避免与构建、独立参考检查争抢资源。
输出及运行库要求见[基准 runner 文档](./dev-tools/README.md#bounded-performance-runs)。

### 通用算子实测（2026-10-29）

Ryzen 9 8945HX，8 workers，单算子任务上限 8；构建条件同下方模型测量。
每项预热 5 组、正式测量 7 组，每组 5 次调用，表中为单次调用中位数。
前向为 no-grad，包含输出分配；反向使用已构建的计算图，计入清梯度、seed 复制和 VJP，
不含前向与优化器 step。原始样本见[测量明细](./dev-tools/benchmarks/results/2026-10-29.json)。

| 操作 | 输入 / 权重形状或参数 | 类型 | 布局 | 中位数 ms |
| --- | --- | --- | --- | ---: |
| matmul 前向 | A=[129,1024], W=[1536,1024] | F32 | 跨步 | 2.06412 |
| matmul 前向 | A=[129,1024], W=[1536,1024] | BF16 | 跨步 | 0.99518 |
| GQA 前向 | 2 条序列；6 个查询头 / 2 个键值头；查询长度 129 / 键值长度 191；每头查询/键特征 65、值特征 33；因果遮罩，查询偏移 31，最后一条序列全遮挡 | F32 | 跨步 | 0.31784 |
| matmul 反向 VJP | A=[193,257], W=[257,257] | F32 | 跨步 | 1.39080 |
| batch_matmul 反向 VJP | A=[2,3,65,129], B=[2,3,129,257] | F32 | 连续 | 0.64416 |

### 模型实测（2026-10-29）

环境：Windows/x86-64、Ryzen 9 8945HX、RTX 5070 Laptop（8 GB），CPU 使用八个计算线程。
Rust 1.98.1 release 构建使用 `target-cpu=native`，启用
`dev-tools,cuda,x86-fp-kernels,x86-int8-kernels`。
[测量明细](./dev-tools/benchmarks/results/2026-10-29.json) 包含被测代码指纹、构建参数、
权重哈希、提示、回放 ID、逐轮预热/正式样本、范围和中位数。

参考为 PyTorch 2.14.0+cu130 / Transformers 5.17.0、SDPA + DynamicCache，
八个 CPU 计算线程、一个 interop 线程，关闭 TF32、低精度归约和图编译。
每进程预热两轮、正式测量五轮，串行执行，不与编译或参考检查争抢资源。
双方使用相同模型、浮点存储精度、提示和固定 64-token decode 回放。
浮点行的权重/激活/KV 类型对应一致。I8 使用 BF16 runtime/激活/KV，关闭动态激活 I8，
不据此计算 PyTorch I8 倍速。

表中为中位数，不计模型加载和外部输入构建，包含 forward、argmax、主机 token 完成及
最后一次 decode forward。Lumen 使用预分配 KV，PyTorch 使用 DynamicCache。
各行每个引擎五个正式样本；TinyLlama CUDA BF16、Qwen3 CUDA F32/F16/BF16 因任一
引擎首轮 prefill 极差/中位数超过 25%，追加一次逆序配对，各有十个正式样本。
所有样本均保留。

#### TinyLlama-1.1B (46 token 提示)

| 设备 / 权重 | Lumen prefill ms | PyTorch prefill ms | Lumen decode token/s | PyTorch decode token/s |
| --- | ---: | ---: | ---: | ---: |
| cpu / F32 | 291.398 | 275.976 | 11.51 | 8.65 |
| cpu / F16 | 291.338 | 1162.755 | 21.01 | 18.80 |
| cpu / BF16 | 248.226 | 97.590 | 21.11 | 16.93 |
| cpu / I8 (BF16 runtime/KV) | 256.487 | — | 35.90 | — |
| cuda / F32 | 31.670 | 22.482 | 49.55 | 60.44 |
| cuda / F16 | 19.992 | 16.739 | 85.82 | 63.34 |
| cuda / BF16 | 19.677 | 19.381 | 88.43 | 54.55 |
| cuda / I8 (BF16 runtime/KV) | 38.883 | — | 148.03 | — |

#### Qwen3-0.6B (36 token 提示)

| 设备 / 权重 | Lumen prefill ms | PyTorch prefill ms | Lumen decode token/s | PyTorch decode token/s |
| --- | ---: | ---: | ---: | ---: |
| cpu / F32 | 163.534 | 131.526 | 17.86 | 11.83 |
| cpu / F16 | 194.189 | 488.894 | 28.33 | 24.27 |
| cpu / BF16 | 145.108 | 64.487 | 27.90 | 22.74 |
| cpu / I8 (BF16 runtime/KV) | 145.368 | — | 41.54 | — |
| cuda / F32 | 21.696 | 26.651 | 56.37 | 34.23 |
| cuda / F16 | 32.385 | 30.793 | 65.64 | 22.22 |
| cuda / BF16 | 31.120 | 30.202 | 68.36 | 24.85 |
| cuda / I8 (BF16 runtime/KV) | 32.170 | — | 92.84 | — |

#### CPU BF16 长输入

| 模型 | 输入 token | Lumen prefill ms | PyTorch prefill ms |
| --- | ---: | ---: | ---: |
| TinyLlama-1.1B | 360 | 1502.552 | 646.172 |
| Qwen3-0.6B | 350 | 939.386 | 370.206 |

长输入使用 BF16 权重/激活/KV、2048 上下文容量，每个引擎预热两次、测量五次。
仅用一次固定 decode forward 完成回放，此处只解释 prefill；短输入上下文容量为 512。
固定回放是计时控制，不是自由生成质量测试。

TinyLlama/Qwen3 的 CPU BF16 短输入 prefill 耗时分别约为 PyTorch 的 2.54/2.25 倍；
TinyLlama CUDA F32 decode 为 49.55 对 60.44 token/s。
GPU 样本存在波动，范围保存在明细中。结果依赖形状、精度、缓存策略和硬件，
不代表编译图、StaticCache 或专用推理引擎，也不能用不同日期或代码快照直接推断版本加速。

## 测试

```bash
cargo test --release --locked --all-targets --features dev-tools
cargo test --release --locked --doc --features dev-tools
cargo fmt --all --check
```

独立参考需要安装 PyTorch 的 Python 环境；真实权重检查还需要 Transformers 和匹配的
模型/tokenizer 文件。PowerShell 脚本使用 PowerShell 7，按所需范围选择入口：

```powershell
# CPU 独立数值参考
./dev-tools/check_operator_accuracy.ps1 -Python python

# portable + native CPU 发布检查
./dev-tools/check_release.ps1 -NativeCpu -Python python

# CPU/CUDA 发布检查，包含长时测试与真实权重 KV 参考
./dev-tools/check_release.ps1 -Cuda -Endurance -Python python -ModelDirectory "path/to/TinyLlama-1.1B-Chat-v1.0"
```

无 CUDA 时，使用 `check_release.ps1 -NativeCpu` 覆盖 portable 和 native CPU。
长时测试只在优化配置上运行一套；旧调度器对照和穷举性能组合按需选择。
测试选择策略见 [`tests/README.md`](./tests/README.md#selecting-a-suite)，
精简性能入口见 [`dev-tools/README.md`](./dev-tools/README.md#bounded-performance-runs)。

默认测试不需要下载模型。测试覆盖标量 FP64/PyTorch 参考、混合精度与跨步张量、
I8 scale/舍入/饱和边界、空输出及梯度、VJP、冻结输入、计算图复用、
梯度累积、检查点、优化器状态、KV 路径和所有权生命周期。
文档编译测试验证执行与精度 guard 的线程限制。

长时测试对小网络运行 1000 次更新，检查数值有限及损失整体下降，允许中间反弹。
真实权重 KV 检查在选定缓存位置对照完整词表 logits；自由生成质量与计时回放分别检查。
小模型能力有限，回答质量可能不高。因此，生成测试使用简单英文问题，
主要判断句子是否流畅、是否存在显著乱码；切题程度和异常重复作为补充观察。
排查推理结果差异时，与相同浮点精度的 PyTorch 对照。
长时测试需显式选择，缺少真实模型资产时会报告跳过。
CI 配置见[便携 CPU 工作流](./.github/workflows/operator-accuracy.yml)。

## 仓库结构

主要目录与入口：

```text
Lumen/
├── Cargo.toml
├── Cargo.lock
├── build.rs
├── CHANGELOG.md
├── .github/workflows/operator-accuracy.yml
├── src/
│   ├── lib.rs
│   ├── main.rs
│   ├── bin/quantize_safetensors.rs
│   ├── autograd/
│   ├── cpu/
│   ├── arch.rs
│   ├── precision.rs
│   ├── init.rs
│   ├── module.rs
│   ├── loss.rs
│   ├── optim/
│   ├── loader.rs
│   ├── loader/pretrained.rs
│   ├── tokenizer.rs
│   ├── layers/
│   │   ├── activation.rs
│   │   ├── basic/
│   │   ├── conv/
│   │   ├── rnn/
│   │   ├── norm/
│   │   └── attention/
│   ├── ops/
│   │   ├── arithmetic.rs
│   │   ├── shape.rs
│   │   ├── matmul/
│   │   └── cuda/
│   └── models/
│       ├── llama.rs
│       ├── pretrained.rs
│       └── qwen3.rs
├── tests/
│   ├── unit/
│   ├── validation/
│   └── profiling/
└── dev-tools/
    ├── README.md
    ├── bin/
    ├── benchmarks/
    │   └── results/2026-10-29.json
    ├── check_release.ps1
    ├── check_operator_accuracy.ps1
    ├── check_performance.py
    └── common.ps1
```

`src/` 只放库与正式命令行程序的实现。`cpu/mod.rs` 负责 CPU 调度和执行上下文，
`cpu/profiling.rs` 是它的可选性能采样子模块；CPU 数值算子仍归入 `ops/`。

`autograd/` 按 Tensor 构造、存储、量化、视图、设备迁移、梯度执行和优化器更新拆分。
`ops/matmul/` 按点积、MatVec、融合投影、prefill、反向计算和 CUDA 路由分组。
`ops/cuda/api/` 放算子接口封装，私有 `imp/` 目录放参数校验、内存与调度及独立的 FFI 声明。
Tensor 和算子的公共访问路径由模块入口保留。

[`tests/`](./tests/README.md) 按单元测试、跨模块验证和显式性能采样测试分组。
需要访问私有实现的单元测试通过 `#[cfg(test)]` 和 `#[path]` 挂接到原模块，
无需扩大公共 API。[`dev-tools/`](./dev-tools/README.md) 集中开发程序、benchmark
辅助模块、检查脚本和独立 Python 参考。Cargo target 名称与 `cargo run --bin ...`
用法不变。使用 `cargo doc --open` 生成 API 文档。

## 限制

- 内置 checkpoint 推理支持文档列出的 Llama 和稠密 Qwen3 配置；其他模型族需要相应实现和权重映射。
- CUDA 面向单个设备，不同后端的算子/精度覆盖有所不同。
- 发布验证覆盖 Windows/x86-64 CPU 和 CUDA GPU；ARM64 与仅 nightly 可用的
  AVX-512 FP16 路径需要单独的平台验证。
- 低精度归约和 argmax 舍入可能改变生成 token；模型知识与指令遵循取决于加载的权重。
- 本项目提供神经网络组件和训练核心，不包含分布式训练器或 serving 系统。

## 许可证

见 [LICENSE](./LICENSE)。
