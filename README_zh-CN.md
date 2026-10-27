# Lumen

一个 Rust 神经网络库，提供动态自动微分、通用神经网络层、支持多种精度的
CPU/CUDA 执行，以及内置 Llama 解码器。

[English](./README.md) · [更新日志](./CHANGELOG.md) · [许可证](./LICENSE)

[功能](#功能) · [快速开始](#快速开始) · [精度与训练](#精度与训练) ·
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
- 带 RoPE、GQA、greedy/采样解码和 KV cache 的 Llama 推理。

Tensor 和层 API 独立于内置模型。CLI 加载的权重需要匹配所配置的 Llama 结构。

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

Q/K/V 形状分别为 `[B,Hq,Q,D]`、`[B,Hkv,K,D]`、`[B,Hkv,K,Dv]`。
布尔遮罩中 `true` 表示允许关注，广播目标为 `[B,Hq,Q,K]`；加性遮罩在缩放后加入。
完全被遮挡的行返回零上下文和零分数梯度。`AttentionOptions` 提供因果遮罩与 query
偏移；投影、位置编码和缓存更新由调用方负责。

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

## CPU 调度

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
参数、激活和 KV 类型必须受所选路径支持；适配其他模型结构需要匹配的实现、配置和 tokenizer。

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

### TinyLlama 1.1B 对照

测量日期为 2026-10-28，环境为 Windows/x86-64、Ryzen 9 8945HX 和
RTX 5070 Laptop（8 GB），CPU 使用八个计算线程。Rust 1.98.1 release 构建使用
`target-cpu=native`，启用 CUDA 及 x86 浮点/I8 内核。
参考为 PyTorch 2.14.0+cu130 / Transformers 5.17.0、
SDPA + DynamicCache，关闭 TF32 和 reduced-precision reduction，不使用图编译。
双方使用 46-token 提示和固定 64-token 回放，每进程预热两轮、测量五轮。
CUDA F16/BF16 因任一引擎首轮 prefill 的极差超过中位数的 25%，
追加逆序采样，形成 Lumen/PyTorch/PyTorch/Lumen 平衡顺序。
这两组每个引擎各十个正式样本，其他行各五个；全部样本保留，下表为中位数。
不计模型加载和外部输入构建，包含 forward、argmax、主机 token 完成及最后一次 decode forward；
Lumen 预分配 KV，因此双方缓存实现不同。

| 设备 / 权重 | Lumen prefill ms | PyTorch prefill ms | Lumen decode token/s | PyTorch decode token/s |
|---|---:|---:|---:|---:|
| cpu / F32 | 312.332 | 273.558 | 11.46 | 8.52 |
| cpu / F16 | 269.304 | 1160.605 | 20.98 | 18.62 |
| cpu / BF16 | 444.203 | 102.229 | 21.10 | 16.70 |
| cpu / I8 (BF16 runtime/KV) | 264.231 | — | 35.16 | — |
| cuda / F32 | 31.008 | 22.979 | 49.46 | 58.73 |
| cuda / F16 | 19.883 | 17.697 | 90.88 | 52.91 |
| cuda / BF16 | 20.115 | 17.982 | 90.82 | 49.88 |
| cuda / I8 (BF16 runtime/KV) | 39.530 | — | 149.21 | — |

I8 配合 BF16 activation/runtime/KV，关闭动态激活 I8，不据此计算 PyTorch I8 倍速。
该环境下 CPU BF16 prefill 耗时约为 PyTorch 的 4.35 倍，CUDA F32 解码吞吐约低 15.8%。
结果取决于形状、精度、缓存策略和硬件，
GPU 参考计时也存在波动；该对照不涵盖编译图、StaticCache 或专用推理引擎。

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
小模型生成使用简单英文问题，检查切题程度、流畅度、乱码及异常重复。
排查差异时与相同浮点精度的 PyTorch 对照；句子流畅也可能包含事实错误。
长时测试需显式选择，缺少真实模型资产时会报告跳过。
CI 配置见[便携 CPU 工作流](./.github/workflows/operator-accuracy.yml)。

## 仓库结构

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

- 内置 checkpoint 推理面向所配置的 Llama 结构，CLI 不负责适配任意模型族。
- CUDA 面向单个设备，不同后端的算子/精度覆盖有所不同。
- 发布验证覆盖 Windows/x86-64 CPU 和 CUDA GPU；ARM64 与仅 nightly 可用的
  AVX-512 FP16 路径需要单独的平台验证。
- 低精度归约和 argmax 舍入可能改变生成 token；模型知识与指令遵循取决于加载的权重。
- 本项目提供神经网络组件和训练核心，不包含分布式训练器或 serving 系统。

## 许可证

见 [LICENSE](./LICENSE)。
