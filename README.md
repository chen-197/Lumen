# Lumen

A Rust neural-network library with dynamic autograd, reusable layers, dtype-aware
CPU/CUDA execution, and built-in Llama and dense Qwen3 decoders.

[中文文档](./README_zh-CN.md) · [Changelog](./CHANGELOG.md) · [License](./LICENSE)

[Features](#features) · [Quick start](#quick-start) · [Library API](#library-api) · [Precision](#precision-and-training) ·
[CPU scheduling](#cpu-scheduling) · [Inference](#model-inference) · [Benchmarks](#benchmarks) ·
[Testing](#testing) · [Repository layout](#repository-layout)

## Features

- Tensor operations and dynamic automatic differentiation with F32 gradients.
- Linear, Embedding, Dropout, Conv2D, MaxPool2D, RNN, GRU and LSTM layers.
- LayerNorm, RMSNorm and scaled dot-product attention for MHA/GQA/MQA and cross-attention.
- SGD, momentum and Adam, with optional F32 master weights for low-precision training.
- F32, F16, BF16 and I8 storage, safetensors loading, streaming and offline I8 quantization.
- Portable CPU execution, optional x86/ARM kernels, and caller-controlled Rayon scheduling.
- Optional CUDA kernels, cuBLAS and cuDNN, including resident tensors and gradient paths.
- Llama and dense Qwen3 inference with RoPE, grouped KV heads, greedy/sampling decode and KV caches.

The Tensor/layer APIs are independent of the built-in models. The CLI reads the
architecture and dimensions from the checkpoint's `config.json`.

## Quick start

### Build from source

```bash
git clone https://github.com/chen-197/Lumen.git
cd Lumen
cargo build --release
```

Use a recent stable Rust toolchain (edition 2024). CUDA is optional for CPU builds.
To use the library directly from Git:

```toml
[dependencies]
lumen = { git = "https://github.com/chen-197/Lumen.git" }
ndarray = "0.15"
rayon = "1.7"
```

### Train a small model

```rust
use lumen::autograd::Tensor;
use lumen::loss::MSELoss;
use lumen::ops::matmul::matmul_transposed_rhs;
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
        let prediction = matmul_transposed_rhs(&input, &weight);
        let loss = MSELoss::apply(&prediction, &target);
        loss.backward();
        optimizer.step();
    }

    let loss = MSELoss::apply(&matmul_transposed_rhs(&input, &weight), &target);
    let value = *loss.data().first().unwrap();
    println!("loss = {value:.6}");
    assert!(value < 1e-6);
}
```

`matmul(A, B)` computes the standard product `A[..., K] @ B[K, N] → [..., N]`.
`matmul_transposed_rhs(A, W)` computes `A[..., K] @ W[N, K]ᵀ → [..., N]` for output-major linear weights.

Migrate earlier `matmul(x, weight)` calls to `matmul_transposed_rhs(x, weight)`.
Square weights also require migration: both layouts have valid shapes but generally produce different results.
`Linear` already uses the explicit transposed-RHS API internally.
Create a fresh forward graph after each parameter update and call `zero_grad()`
before the next batch. Leaf gradients accumulate across backward calls;
`backward_with_grad(seed)` accepts a custom seed.

### General attention and normalization

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

The dimensions are ordered as follows:

| Tensor | Dimension order | Shape in the example |
| --- | --- | --- |
| Q (queries) | `[batch size, query heads, query sequence length, features per query head]` | `[2, 4, 3, 5]` |
| K (keys) | `[batch size, key/value heads, key/value sequence length, features per key head]` | `[2, 2, 7, 5]` |
| V (values) | `[batch size, key/value heads, key/value sequence length, features per value head]` | `[2, 2, 7, 6]` |
| Attention output | `[batch size, query heads, query sequence length, features per value head]` | `[2, 4, 3, 6]` |

The example processes two sequences together. Each has three query positions
attending to seven key/value positions. Four query heads share two key/value
heads, so each pair of query heads uses one K/V head. Queries and keys must have
the same per-head feature count. Values may have a different feature count,
which becomes the output's last dimension. The query head count must be a
multiple of the key/value head count.

Boolean masks use `true` to permit attention and broadcast to
`[batch size, query heads, query sequence length, key/value sequence length]`.
The example's `[2, 1, 1, 7]` mask specifies seven key positions per sequence,
shared across all query heads and query positions. Additive masks are applied
after score scaling. Fully masked rows return zero context and score gradients.
`AttentionOptions` provides causal masking and a query-position offset;
projection, positional encoding and cache updates remain the caller's responsibility.

## Library API

Import `lumen::module::Module` for layer `forward`/`parameters` methods and
`lumen::optim::Optimizer` for `step`/`zero_grad`. The examples below use the
dependencies from Quick start. Generate the full API reference with
`cargo doc --no-deps --open`; the tables describe the main application-facing APIs.

[Tensors/autograd](#tensors-and-automatic-differentiation) · [Operators](#operators-and-shapes) ·
[Training/checkpoints](#compose-a-network-choose-a-loss-and-save-parameters) ·
[Custom modules](#define-your-own-module) · [Layers](#available-layers) · [Devices](#cpu-and-cuda-placement)

| Module | What it provides |
| --- | --- |
| [`device`](./src/device.rs) | `Device`, CPU/CUDA placement and transfers, strict execution policy |
| [`autograd`](./src/autograd/) | `Tensor`, graph construction, gradients and storage |
| [`ops`](./src/ops/) | Arithmetic, shape operations, matrix products, convolution, pooling and fused kernels |
| [`layers`](./src/layers/) | Reusable layers, activation functions, normalization and attention |
| [`module`](./src/module.rs) | `Module`, `Sequential`, `sequential!`, checkpoints and parameter conversion |
| [`loss`](./src/loss.rs), [`optim`](./src/optim/) | MSE, dense-target cross entropy, SGD/momentum and Adam |
| [`precision`](./src/precision.rs), [`init`](./src/init.rs) | Dtype/quantization policies, parameter initialization and scoped settings |
| [`models`](./src/models/), [`loader`](./src/loader.rs), [`tokenizer`](./src/tokenizer.rs) | Decoder configurations, checkpoint loading, tokenization and chat formatting |
| [`cpu`](./src/cpu/) | Caller-owned worker pools and task budgets |

### Tensors and automatic differentiation

| API | Usage |
| --- | --- |
| `Tensor::new(array)`, `from_data(array)` | F32 leaf; tracks gradients unless created inside `no_grad` |
| `new_with_dtype(array, dtype)` | Leaf with explicit F32/F16/BF16/I8 storage |
| `from_array_no_grad(array)` | Constant/input from an owned `ndarray::ArrayD<f32>` |
| `from_data_no_grad(shared)` | Constant from an `ndarray::ArcArray<f32, IxDyn>` |
| `from_data_with_grad_flag(array, flag)` | Explicitly choose whether the leaf receives gradients |
| `parameter(array)`, `parameter_with_dtype(array, dtype)` | Trainable parameter; the first follows the default parameter policy |
| `parameter_with_quantization(array, policy)` | Trainable parameter with an explicit quantization policy |
| `shape_vec()`, `ndim()`, `len()`, `is_empty()`, `dtype()`, `device()`, `requires_grad()` | Inspect metadata |
| `data()`, `data_arc()`, `data_ref()`, `data_mut()` | Owned copy, shared F32 array, borrowed F32 view, mutable F32 view |
| `backward()` | Backpropagate an all-ones seed; for a non-scalar this differentiates the sum |
| `backward_with_grad(seed)` | Vector-Jacobian product; seed shape must exactly match the output |
| `grad()`, `grad_arc()`, `grad_ref()`, `zero_grad()` | Read or clear F32 gradients |
| `detach()`, `no_grad(\|\| ...)`, `NoGradGuard::enter()` | Independent untracked copy, or scoped graph suppression |
| `cast_inplace(dtype)`, `quantize_inplace_with_quantization(policy)`, `dequantize_inplace(dtype)` | Convert tensor storage; `quantization_scale()` reads an I8 scale |
| `export_raw()`, `import_raw(shape, dtype, raw)` | Export/restore typed tensor data and I8 scales |

`Tensor::clone()` shares a handle, including parameter updates and gradients.
`detach()` copies storage. F32 data reads materialize/transfer values and may also
promote the original Tensor storage to F32, especially activations and parameters
without retained dtype copies. To inspect without changing the original storage,
use `x.detach().data()` on an independent copy, or `export_raw()` for native data. Drop `data_ref`/`data_mut` borrow guards before other
operations that borrow or mutate the same Tensor. Leaf gradients accumulate;
clear them between batches. A retained graph reuses its original forward values,
so run a fresh forward after updating parameters.

```rust
use lumen::autograd::Tensor;
use lumen::ops::{matmul::{batch_matmul, matmul, matmul_transposed_rhs}, shape::cat};
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

    let rhs = Tensor::from_array_no_grad(arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).into_dyn());
    assert_eq!(matmul(&x, &rhs).shape_vec(), vec![2, 3]);

    let weight = Tensor::parameter(arr2(&[[1.0, 0.0], [0.0, 1.0]]).into_dyn());
    assert_eq!(matmul_transposed_rhs(&x, &weight).shape_vec(), vec![2, 2]);

    let a = Tensor::from_array_no_grad(ArrayD::ones(IxDyn(&[1, 2, 3, 4])));
    let b = Tensor::from_array_no_grad(ArrayD::ones(IxDyn(&[1, 2, 4, 5])));
    assert_eq!(batch_matmul(&a, &b).shape_vec(), vec![1, 2, 3, 5]);
}
```

### Operators and shapes

| API | Input/output convention |
| --- | --- |
| `&a + &b`, `&a - &b`, `&a * &b` | Elementwise operations with compatible broadcasting; operands are Tensors |
| `x.sum()` / `ops::arithmetic::sum(&x)` | Sum all elements to a scalar |
| `x.reshape(vec![...])` / `ops::shape::reshape` | Preserve element count; one `-1` can infer a dimension when the known product is nonzero |
| `x.permute(axes)`, `x.transpose(i, j)` | Reorder axes |
| `ops::shape::cat(&tensors, axis)` | Concatenate along an axis |
| `ops::shape::slice_last_dim(&x, start, end)` | Slice the final axis over `[start, end)` |
| `ops::matmul::matmul(&a, &b)` | Standard product: `[..., K] × [K, N] → [..., N]`; RHS is 2D and LHS leading dimensions are preserved |
| `ops::matmul::matmul_transposed_rhs(&a, &weights)` | `[..., K] × [N, K]ᵀ → [..., N]`; weights are stored output-major |
| `ops::matmul::batch_matmul(&a, &b)` | `[B,H,M,K] × [B,H,K,N] → [B,H,M,N]`; RHS is already in multiplication order |
| `ops::convolution::{conv2d, max_pool2d}` | NCHW convolution and pooling; layer wrappers provide simpler constructors |
| `ops::fused::fused_softmax(&x, scale, causal)` | Scaled softmax, optionally with a causal mask |

Tensor operators build a graph when tracked inputs participate. Use `no_grad`
for inference; supported empty matrix axes and reductions retain their logical
shapes. Shape and device requirements are checked at the operation boundary.
For raw-slice projection/MatVec/argmax APIs, see `ops::matmul`; the
`fused_gate_up_silu_infer*` and `fused_qkv_*` families in `ops::fused` are
specialized inference entry points.

### Compose a network, choose a loss, and save parameters

`Sequential::new(Vec<Box<dyn Module>>)` and `sequential![...]` chain ordinary
single-input layers. `train_mode()` / `eval_mode()` propagate through a
Sequential, including Dropout; `no_grad` alone does **not** turn Dropout off.
For ordinary modules, mode switching also changes inference/graph mode on the
calling thread. Wrap inference in `no_grad` and set the module's evaluation mode.

| API | Contract |
| --- | --- |
| `MSELoss::apply(&prediction, &target)` | Same nonempty shape; mean squared error over all elements |
| `CrossEntropyLoss::apply(&logits, &targets)` | `[B,C]` raw logits and same-shaped dense one-hot/soft targets; batch-mean loss |
| `SGD::new_with_dtype(params, lr, state_dtype)` | SGD; `.with_momentum(value)` enables momentum |
| `Adam::new_with_dtype(params, lr, state_dtype)` | Adam with default betas `(0.9, 0.999)` and epsilon `1e-8` |
| `Optimizer::{zero_grad, step, params}` | Clear accumulated gradients, apply updates, inspect parameter handles |
| `Module::{save, load}` | Lumen binary parameter checkpoints; restore into a matching architecture/parameter order |
| `Module::{cast_parameters, quantize_parameters, apply_parameter_quantization}` | Storage conversion, I8 quantization, or policy-driven quantization, respectively |

Cross entropy accepts **dense targets**, not an integer class-index vector.
Pass logits directly without a preceding Softmax. `new(...)` optimizer
constructors use the default runtime dtype for optimizer state; use
`new_with_dtype(..., DType::F32)` when you want explicit F32 state.

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

Checkpoints include parameter shapes, dtypes, values and I8 scales. They do not
save the network definition, optimizer state, RNG or KV caches. Recreate the same
architecture, load parameters, then create a new optimizer. The binary format is
separate from safetensors, which is handled by `ModelLoader`.

### Define your own Module

Implement `forward` and `parameters`; return the original trainable Tensor handles
so optimizers update the values used by forward. For custom weight sharing, return
each shared parameter once. Override mode propagation when a custom container owns
Dropout or other stateful children.

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

### Available layers

Most parameterized layers provide `new_with_dtype(...)`; `Linear` also provides
`new_no_bias(...)` and `new_no_bias_with_dtype(...)`.

| Layer / constructor | Shape or behavior |
| --- | --- |
| `Linear::new(input, output)` | `[..., input] → [..., output]`; optional bias |
| `Embedding::new(vocab, dim)` | `ids[...] → [..., dim]`; `forward(&ids)` |
| `Dropout::new(p)`, `new_with_seed(p, seed)` | Inverted dropout, `0 ≤ p < 1`; seeded calls reproduce the same mask |
| `Flatten::new()` | Keep the first/batch axis and flatten the remaining axes |
| `activation::{ReLU, Sigmoid, Tanh, SiLU, Gelu}::new()` | Elementwise activations |
| `activation::Softmax::new(axis)` | Normalize along one axis |
| `Conv2D::new(in_channels, out_channels, kernel, stride, padding)` | Square-kernel NCHW convolution |
| `MaxPool2D::new(kernel, stride)` | NCHW max pooling |
| `RNN::new(input, hidden)`, `GRU::new(input, hidden)` | `forward_step(&x, &h) → h_next`, with `[B,input]` and `[B,hidden]` |
| `LSTM::new(input, hidden)` | `forward_step(&x, &h, &c) → (h_next, c_next)` |
| `LayerNorm::new(shape, eps)` | Normalize a trailing shape; `with_options` selects affine/bias/parameter dtype |
| `RMSNorm::new(dim, eps)` | Normalize the last dimension; `new_with_dtype` selects parameter storage |
| `scaled_dot_product_attention(...)` | Already projected MHA/GQA/MQA/cross-attention; see Quick start |
| `SelfAttention::new(embed, heads, kv_heads, max_seq, theta, causal)` | `forward(x, Option<KVCache>) → (output, cache)`; `new_with_options` supports explicit head_dim and Q/K norm |
| `RotaryEmbedding::new(dim, max_seq, theta)` | `forward(&x, offset)`, with `[batch size, attention heads, sequence length, features per head]` input; features per head must be even |
| `KVCache` | Shared per-layer cache handle; built-in models provide `init_kv_caches` and explicit reset |

Import ReLU/Sigmoid/Tanh/Softmax from `lumen::layers::activation`; SiLU and Gelu
are also re-exported from `lumen::layers`. Recurrent layers require
`forward_step`: their `Module::forward` is not a sequence runner and cannot be
placed directly in Sequential. Iterate over sequence steps yourself; detach hidden
states between chunks when truncated backpropagation is intended.
Embedding/model token IDs are integer-valued F32 Tensors, not I8-quantized IDs.

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

### CPU and CUDA placement

| API | Behavior |
| --- | --- |
| `tensor.to_device(Device::Cpu / Device::Cuda)` | Independent Tensor/storage copy; retains the leaf's gradient flag, not its backward graph |
| `to_cpu()`, `to_cuda()`, `cpu()`, `cuda()` | Tensor copy conveniences |
| `to_device_inplace(device)`, `to_cpu_inplace()`, `to_cuda_inplace()` | Migrate the existing Tensor handle |
| `module.to_device(device)`, `to_cpu()`, `to_cuda()` | Move the module's parameters |
| `ops::cuda::is_available()` | Check whether this build and host provide CUDA |
| `set_strict_device_execution_scoped(true)` | Reject operations requiring an unsupported CPU fallback |

Move parameters and inputs before building the graph and creating optimizer state;
operands must reside on the same device. Host inspection such as `data()`/`grad()`
can synchronize CUDA and copy values back. CUDA targets a single device;
operator/dtype coverage differs from CPU.

```rust
use lumen::device::{Device, set_strict_device_execution_scoped};
use lumen::autograd::{Tensor, no_grad};
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

## Build options

```bash
# CUDA backend
cargo build --release --features cuda

# x86 kernels and benchmark tools
cargo build --release --features "dev-tools x86-fp-kernels x86-int8-kernels"

# Native code generation for a machine-specific performance build
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

In PowerShell, set `$env:RUSTFLAGS = "-C target-cpu=native"` before building.
CUDA builds need a CUDA toolkit and compatible NVIDIA driver. `build.rs` discovers
CUDA/cuDNN through environment variables, `nvcc` and system installation paths.

| Feature | Purpose |
|---|---|
| `cuda` | CUDA C++ backend, cuBLAS and optional cuDNN |
| `dev-tools` | Benchmark and diagnostic binaries |
| `cpu-profiling` | Kernel timing instrumentation; includes `dev-tools` |
| `x86-fp-kernels` | Stable x86 F32/F16/BF16 kernels |
| `x86-fp-kernels-nightly` | Native AVX-512 FP16 intrinsics; requires nightly |
| `x86-int8-kernels` | x86 I8 kernels |
| `arm64-fp-kernels` | ARM64 floating-point kernels |
| `arm64-int8-kernels` | ARM64 I8 kernels |

Architecture features are opt-in; runtime detection selects supported kernels.
Native AVX-512 FP16 can be built with
`cargo +nightly build --release --features x86-fp-kernels-nightly`.
Use a non-instrumented build for performance comparisons.

## Precision and training

- Parameters, activations and KV storage can use different dtypes on supported paths.
  KV caches use floating-point storage; I8 weights commonly pair with BF16 runtime/KV.
- Gradients are F32. Operations tracked by autograd produce F32 outputs; same-dtype
  lowp matrix inference preserves its storage dtype. Mixed F32/F16 activation ×
  BF16 weight paths preserve the declared activation values.
- Matmul and batch matmul support empty axes and zero-length reductions. Frozen
  inputs do not receive operator gradients. Flatten and shape operations preserve
  empty output shapes and their corresponding input gradient shapes.
- `no_grad` disables graph construction. `detach()` disables tracking and owns
  independent storage. Use `backward_with_grad`, rather than preloading a seed with `add_grad`.
- `backward_with_cuda_grad` accepts a resident F32 CUDA seed. Reading a CUDA
  gradient through `grad()`, `grad_arc()` or `grad_ref()` materializes a cached
  host representation. Keep gradient computation on device to avoid this transfer.
- CPU optimizer steps retain gradients; successful CUDA updates consume the
  applied gradients. Call `zero_grad()` before each batch on either backend.
  SGD and Adam validate every gradient shape before changing any parameter or
  optimizer state; a shape mismatch panics.
- Module checkpoints contain parameters. Optimizer moments, master-weight residuals
  and RNG state are not serialized.

For low-precision training, choose F32 optimizer state and call
`.with_fp32_master_weights()` on SGD or Adam. For trainable I8 ranges,
`.with_dynamic_i8_master_weights()` recalibrates scales each step. These options
must be chosen before training and add F32 master storage. Rebuild the optimizer
after replacing parameter storage, shapes, dtypes, scales or devices.

I8 activation quantization is explicit: `--activation-quantization dynamic-i8`.
The default is `off`; use the same policy when comparing outputs or performance.

### Configure precision and quantization in library code

| API | Scope |
| --- | --- |
| `precision_guard(PrecisionConfig { ... })`, `with_precision_config(config, f)` | Parameter/runtime defaults and permission to retain parameter dtype copies |
| `runtime_component_dtypes_guard(Some(activation), Some(kv))`, `with_runtime_component_dtypes(...)` | Separate default activation/KV storage; `None` leaves a component unchanged |
| `ParameterQuantization::new(DType::I8).with_scale(scale)` | Explicit per-tensor I8 storage with a manual scale; omit `with_scale` for automatic scale |
| `parameter_quantization_guard(policy)`, `with_parameter_quantization(policy, f)` | Default quantization for constructors following that policy |
| `tensor_init(shape, InitType::...)`, `tensor_init_with_dtype(...)` | XavierUniform, KaimingNormal, Zeros or Ones parameter initialization |
| `with_parameter_init_mode(ParameterInitMode::Placeholder, f)` | Skip random initialization when immediately loading pretrained parameters |

Defaults affect constructors/routes that consult them; they do not recast an
existing model. Explicit dtype constructors take precedence. Scoped guards
restore settings on drop; set policies around model construction and execution.

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

## CPU scheduling

```rust
use lumen::autograd::{Tensor, no_grad};
use lumen::cpu::{CpuInferenceContext, CpuKernelContext};
use lumen::ops::matmul::matmul_transposed_rhs;
use ndarray::arr2;
use std::{num::NonZeroUsize, sync::Arc};

fn main() {
    let pool = Arc::new(rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap());
    let context = CpuKernelContext::with_pool(pool)
        .with_task_limit(NonZeroUsize::new(2).unwrap());
    let input = Tensor::from_array_no_grad(arr2(&[[1.0, 2.0]]).into_dyn());
    let weight = Tensor::parameter(arr2(&[[0.5, 0.5]]).into_dyn());
    context.scope(|| matmul_transposed_rhs(&input, &weight).sum().backward());
    assert!(weight.grad().is_some());

    let inference = CpuInferenceContext::new(context)
        .with_prefill_task_limit(NonZeroUsize::new(2).unwrap())
        .with_decode_task_limit(NonZeroUsize::new(1).unwrap());
    let prefill = inference.prefill(|| no_grad(|| matmul_transposed_rhs(&input, &weight)));
    let decode = inference.decode(|| no_grad(|| matmul_transposed_rhs(&input, &weight)));
    assert_eq!(prefill.data(), decode.data());
}
```

Create pools outside compute loops and share them between callers. `scope` keeps
Tensors and autograd state on the calling thread; numerical slices enter the pool.
The task budget is per integrated kernel, not a process-wide thread limit.
Explicitly capped indexed loops divide tiles evenly across the selected task
groups without temporary lists of views, avoiding lost groups from ceiling-width
partitioning.
Small jobs and nested numerical work execute serially. The library does not set
a global pool, pin threads or infer policy from a particular CPU model.

MatVec, matmul/batch-matmul forward and backward, fused projections, argmax,
prefill, inference attention, RMSNorm forward rows and native lowp optimizer-state kernels observe
these contexts. Enter the scope around **backward as well as forward**; creating
a graph inside it does not bind a later traversal to that policy.
LayerNorm forward also uses these contexts. CPU LayerNorm inference computes
rows without retaining backward intermediates and preserves input/parameter
storage precision; centered statistics use F64 and outputs remain F32.
RMSNorm input-gradient rows and weight-gradient reductions also observe these
contexts. Within-row arithmetic is unchanged; grouping weight-gradient sums can
change floating-point rounding. LayerNorm backward also observes these contexts:
small work uses a fused serial pass, while larger input/parameter gradients share
grouped row work and F64 accumulation. Frozen inputs and parameters skip their
gradient arrays and computation. Grouping parameter sums may change rounding.
`CpuInferenceContext` provides separate prefill/decode budgets over a shared pool.
Tensor handles use `Rc<RefCell<_>>` and are not `Send`/`Sync`; separate callers
own their model/Tensor handles while sharing numerical worker pools.
CPU BF16 prefill reuses temporary weight panels across input rows, including
long prompts. Supported CPUs use BF16 dot products with F32 accumulation.
BF16 K-pair weight packing uses an integer AVX2 transpose for complete panels,
with SSE2/scalar fallbacks for smaller panels and other CPUs.
Native BF16 × BF16 matrix-product panels write directly into the final output,
avoiding a padded full-result buffer and its rearrangement. Adjacent panels
share a task within the caller's pool capacity and task budget.
Narrow BF16 output projections can also reuse native panels when the input size
and caller's task capacity amortize packing; smaller parallel products retain
the row kernels. Single-row decoding keeps its existing path.
Mixed F32/F16 activations
retain their precision through exact weight expansion and AVX2/FMA or AVX-512.
Other CPUs use the portable fallback without changing the public API.
Standard `matmul` also packs F16/BF16/I8 RHS blocks directly from their original
strides on supported x86 CPUs, avoiding a complete F32 RHS copy or transpose.
BF16 input/RHS pairs can use AVX-512 BF16 with F32 accumulation; other native
panels expand weights during F32 computation. Vectors and a few input rows can
stream row-major F32/F16/BF16/I8 RHS columns directly with AVX2/FMA or AVX-512,
without packing. Small work stays on the calling thread; larger work respects
the caller's pool and task budget. Narrow multi-row F32 products, strided RHS
and unsupported small shapes retain general GEMM. CPU matrix-product backward
skips gradients for frozen inputs.
No-grad, precision, initialization and device-selection guards are also
`!Send`/`!Sync`. Create and drop these scopes on the thread that executes the
computation so their thread-local settings are restored correctly.

## Model inference

```bash
cargo run --release --bin lumen -- \
  --weights path/to/model.safetensors \
  --tokenizer path/to/tokenizer.json
```

Add `--features cuda` and `--device cuda` for GPU inference. Use
`cargo run --release --bin lumen -- --help` for precision, cache, sampling,
streaming and CPU task options. Parameter, activation and KV dtypes must be
supported by the selected path. Keep `config.json` beside the weights, or provide
`--config PATH`. Both a single safetensors file and a HF safetensors shard index
are supported; a directory passed to `--weights` is resolved automatically.

### Qwen3

```bash
cargo run --release --bin lumen -- \
  --weights path/to/Qwen3-0.6B/model.safetensors \
  --tokenizer path/to/Qwen3-0.6B/tokenizer.json \
  --max-seq-len 1024
```

Dense Qwen3 uses explicit head dimensions, per-head Q/K RMSNorm and tied
embeddings when configured. The CLI uses text-only ChatML with thinking disabled
by default; `--enable-thinking` enables it. Model/configuration and generation
stop IDs are used alongside tokenizer markers. `prefill_decode_bench` reads the
same configuration and supports both generation and forced-token diagnostics.
Unsupported MoE, sliding-window, scaled/partial RoPE and biased-attention
configurations return an error. Arbitrary Jinja templates, tool calls and
multimodal inputs are not interpreted by the CLI.
The CLI chat formatters currently cover TinyLlama and Qwen3; other conversation
templates require a matching formatter.

### Load a pretrained model from the library

Use `CausalLMConfig` to construct the supported architecture from `config.json`.
`Qwen3Config`/`Qwen3Model` provide a named wrapper for dense Qwen3;
`LlamaConfig`/`LlamaModel` also support explicit construction for custom models.
The common configuration builder returns the shared decoder implementation.

| API | Purpose |
| --- | --- |
| `CausalLMConfig::{from_file, validate, build}` | Read/validate architecture and build with explicit context capacity and dtypes |
| `ModelLoader::load_weights_with_options(path, &parameters, options)` | Strict safetensors file, model directory or HF shard-index loading |
| `WeightLoadOptions::{stream_from_disk, float_source_quantization}` | Stream parameter payloads or quantize floating source weights during loading |
| `model.parameters()` | Unique trainable handles for the built-in decoder |
| `model.named_parameters()`, `checkpoint_parameters()` | Inspect names; canonical checkpoint names omit a tied output-weight alias |
| `ModelTokenizer::{from_file, encode, decode, decode_complete_prefix}` | Tokenization, decoding and a complete UTF-8 display prefix with pending IDs |
| `ChatFormat::{first_turn, next_turn}` | TinyLlama / Qwen3 text-only conversation formatting |
| `init_kv_caches(batch)`, `init_kv_caches_with_dtype(batch, dtype)` | Allocate per-layer inference caches |
| `forward_last_argmax(ids, &mut caches, pos)` | Greedy next token for batch size one; avoids materializing full logits |
| `forward_last_argmax_batch(...)`, `forward_last_logits(...)` | Batch token choices or last-position logits `[B,1,V]` |
| `model.forward(ids, &mut caches, pos)` | Cached inference logits `[B,S,V]` |
| `forward_train(ids)` / `Module::forward` | Differentiable full-sequence logits, without inference caches |
| `reset_kv_caches(&mut caches)`, `cast_kv_caches(&mut caches, dtype)` | Start a new sequence or convert floating cache storage |

This CPU example performs one prompt prefill, then feeds one generated token per
decode step. It accepts a model directory as its first program argument, uses a
256-token context capacity, and generates at most 32 tokens. Use a prompt plus
output budget that fits the capacity. Placeholder initialization skips random
parameter initialization; load every required weight before inference.

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

`pos` is retained for API compatibility; cache lengths determine positions.
For a new independent prompt, reset the caches first. For continuation, feed only
the new tokens: replaying the full prefix into a populated cache appends it again.
The example uses greedy decoding and model/tokenizer stop IDs; applications can
use `forward_last_logits` for custom sampling and additional generation-config
stop policies. `decode_complete_prefix` helps streaming clients wait for complete
UTF-8 characters without discarding original token IDs.
For training, create a fresh `forward_train` graph and reshape sequence logits
and dense targets to `[B*S,V]` for `CrossEntropyLoss`; do not reuse inference KV.

For streaming, pass `WeightLoadOptions { stream_from_disk: true, ..Default::default() }`.
For on-load I8 quantization, set
`float_source_quantization: ParameterQuantization::new(DType::I8)` and build the
model with I8 parameters and floating activation/KV storage.

The loader checks required names, metadata, shapes and I8 companion scales before
loading payloads. Its legacy `load_llama_weights*` entry points remain available;
use the strict common loader for configuration-driven models.

### Offline quantization

```bash
cargo run --release --bin quantize_safetensors -- \
  --input path/to/model.safetensors \
  --output path/to/model.i8.safetensors \
  --dtype i8
```

`--scale` optionally supplies a manual I8 scale. The loader also supports
on-load quantization and streamed safetensors loading.

The converter uses symmetric per-tensor quantization to `[-127, 127]`, with
half-integer ties rounded away from zero. Manual scales must be finite and
positive; positive subnormal scales are supported. Each I8 tensor is accompanied
by a scalar F32 `<tensor-name>.scale` entry required when loading its weights.

## Benchmarks

Benchmark tools require `dev-tools` and should run with `--release`.

| Binary | Purpose |
|---|---|
| `kernel_bench` | CPU kernel microbenchmarks |
| `generic_prefill_bench` | Matmul, batch matmul and general attention |
| `cpu_backward_bench` | Matmul and batch-matmul gradients |
| `cpu_projection_bench` | Projection, QKV and gated MLP kernels |
| `cpu_decode_bench` | Decode working sets and vocabulary argmax |
| `cpu_scaling_bench` | Pool size, task budgets, concurrent callers, and normalization |
| `cuda_cpu_bench` | CPU/CUDA operators and training paths |
| `prefill_decode_bench` | Real-weight prefill/decode, replay and sampling |
| `manual_state_check` | Public API, checkpoint and state checks |

```bash
cargo run --release --features "dev-tools x86-fp-kernels x86-int8-kernels" \
  --bin generic_prefill_bench -- --threads 4 --tasks 2 --warmup 5 --samples 7 --iters 5

cargo run --release --features "dev-tools cuda x86-fp-kernels x86-int8-kernels" \
  --bin prefill_decode_bench -- --help
```

`--samples-output` saves model timing samples; `--replay-token-ids` provides
fixed-input decode measurements. Check free-generated text separately from replay.
`cpu-profiling` adds diagnostic timing regions; its overhead should be excluded
from final performance comparisons.

For a representative CPU suite, build the tools once and run the Python runner
against the resulting release directory:

```bash
cargo build --release --locked --features dev-tools --bins
python dev-tools/check_performance.py --threads 8 --plan
python dev-tools/check_performance.py --threads 8 \
  --binary-directory path/to/release --output-directory target/performance-new
```

The default selection runs 21 processes at more than one worker, with warmup
before timing. `--select argmax` narrows the selection; `--extended` enables
additional task budgets and precision/request combinations. Use a new output
directory for each run and run timing serially, without competing builds or
reference checks. See the [benchmark runner documentation](./dev-tools/README.md#bounded-performance-runs)
for output and runtime-library requirements.

### General operator measurements (2026-11-04)

Windows x86-64, Ryzen 9 8945HX, eight workers. Matrix multiplication and attention
use a task limit of eight, five warmup samples, and seven measured samples with
five calls each. Forward includes output allocation under no-grad. Backward reuses
a prebuilt graph and measures gradient clearing, upstream seed copying, and the VJP.
`matmul` computes `A @ B` and `matmul_transposed_rhs` computes `A @ Wᵀ`.
The data records matrix strides in the measured storage dtype.
[Samples and build metadata](./dev-tools/benchmarks/results/2026-11-04.json)

| Operation | Shapes or parameters | Dtype | Layout | Median ms |
| --- | --- | --- | --- | ---: |
| `matmul` forward | `A=[129,1024]`, `B=[1024,1536]` | F32 | strided | 2.85434 |
| `matmul` forward | `A=[129,1024]`, `B=[1024,1536]` | BF16 | strided | 0.91118 |
| `matmul_transposed_rhs` forward | `A=[129,1024]`, `W=[1536,1024]` | F32 | strided | 2.45120 |
| `matmul_transposed_rhs` forward | `A=[129,1024]`, `W=[1536,1024]` | BF16 | strided | 2.76784 |
| GQA forward | 2 sequences, 6 query heads, 2 key-value heads, query length 129, key-value length 191, 65 query/key and 33 value features per head, causal mask, query offset 31, final sequence fully masked | F32 | strided | 0.33592 |
| `matmul_transposed_rhs` backward VJP | `A=[193,257]`, `W=[257,257]` | F32 | strided | 1.43494 |
| `batch_matmul` backward VJP | `A=[2,3,65,129]`, `B=[2,3,129,257]` | F32 | contiguous | 0.67626 |

Normalization uses automatic task selection, five warmup calls, and seven
measured samples with 30 calls each. Shapes are `[rows, normalized features]`.
Backward measures the complete VJP with gradient clearing and seed copying.

| Operation | Shape | F32 median µs | BF16 median µs |
| --- | --- | ---: | ---: |
| `rms_norm` | `[65,513]` | 56.593 | 51.633 |
| `rms_norm_backward` | `[65,513]` | 96.607 | 116.770 |
| `rms_norm_backward` | `[512,2048]` | 1876.883 | — |
| `layer_norm` | `[65,513]` | 66.163 | 79.343 |
| `layer_norm_backward` | `[65,513]` | 230.740 | 226.287 |
| `layer_norm_backward` | `[512,2048]` | 4398.540 | — |

### Model measurements (2026-11-04)

Windows x86-64, Ryzen 9 8945HX, RTX 5070 Laptop GPU (8 GB), eight CPU workers.
Lumen used Rust 1.98.1 with `--release`, `-C target-cpu=native`, and
`dev-tools,cuda,x86-fp-kernels,x86-int8-kernels`. PyTorch 2.14.0+cu130 with
Transformers 5.17.0 used SDPA and DynamicCache, eight CPU threads and one
interop thread, with TF32, reduced-precision reductions and compilation disabled.
Lumen used a preallocated KV cache with capacity 1024.
[Source fingerprint, model assets, and samples](./dev-tools/benchmarks/results/2026-11-04.json)

Each row reports medians from five runs after two warmups per engine. Both
engines use the same prompt, 64 fixed decode steps, and matching floating-point
weight, activation, and KV dtypes. Forward timing includes argmax and excludes
model loading and input construction. I8 rows use I8 weights with BF16
activations and KV, and report Lumen timings only.

#### TinyLlama-1.1B (46-token prompt)

| Device / weights | Lumen prefill ms | PyTorch prefill ms | Lumen decode steps/s | PyTorch decode steps/s |
| --- | ---: | ---: | ---: | ---: |
| cpu / F32 | 441.550 | 410.303 | 11.51 | 7.62 |
| cpu / F16 | 267.890 | 1767.148 | 20.27 | 16.06 |
| cpu / BF16 | 245.308 | 158.047 | 18.96 | 14.87 |
| cpu / I8 | 315.466 | — | 32.70 | — |
| cuda / F32 | 37.014 | 28.952 | 48.31 | 50.66 |
| cuda / F16 | 25.185 | 23.569 | 76.54 | 45.83 |
| cuda / BF16 | 25.929 | 23.275 | 75.37 | 44.98 |
| cuda / I8 | 51.057 | — | 127.01 | — |

#### Qwen3-0.6B (36-token prompt)

| Device / weights | Lumen prefill ms | PyTorch prefill ms | Lumen decode steps/s | PyTorch decode steps/s |
| --- | ---: | ---: | ---: | ---: |
| cpu / F32 | 216.199 | 232.537 | 17.65 | 10.35 |
| cpu / F16 | 242.245 | 613.837 | 27.24 | 20.69 |
| cpu / BF16 | 131.692 | 94.021 | 26.09 | 18.45 |
| cpu / I8 | 213.505 | — | 34.48 | — |
| cuda / F32 | 33.181 | 35.572 | 44.67 | 29.02 |
| cuda / F16 | 56.987 | 39.096 | 50.68 | 26.48 |
| cuda / BF16 | 55.199 | 38.331 | 51.43 | 26.96 |
| cuda / I8 | 54.523 | — | 75.00 | — |

The earlier dataset includes CPU BF16 long-prompt prefill measurements.
[2026-10-29 measurements](./dev-tools/benchmarks/results/2026-10-29.json)

## Testing

```bash
cargo test --release --locked --all-targets --features dev-tools
cargo test --release --locked --doc --features dev-tools
cargo fmt --all --check
```

Independent references require a Python environment with PyTorch. Real-weight
checks also require Transformers and matching model/tokenizer files. Run the
PowerShell scripts with PowerShell 7. Choose the runner for the required scope:

```powershell
# Independent CPU numerical references
./dev-tools/check_operator_accuracy.ps1 -Python python

# Release checks on portable and native CPU
./dev-tools/check_release.ps1 -NativeCpu -Python python

# CPU/CUDA release checks, including endurance and real-weight KV references
./dev-tools/check_release.ps1 -Cuda -Endurance -Python python -ModelDirectory "path/to/TinyLlama-1.1B-Chat-v1.0"
```

Use `check_release.ps1 -NativeCpu` for portable and native CPU coverage without
CUDA. Run endurance once on the optimized configuration; legacy scheduler
ablations and exhaustive performance grids are opt-in investigations. See
[suite selection](./tests/README.md#selecting-a-suite) and the
[representative performance runner](./dev-tools/README.md#bounded-performance-runs).

Default tests do not require model downloads. Tests cover scalar FP64/PyTorch
references, mixed and strided tensors, I8 scale/rounding/saturation boundaries,
empty outputs and gradients, VJPs,
frozen inputs, retained graphs, gradient accumulation, checkpoints, optimizer
state, KV routes and ownership lifetimes. Documentation tests enforce the
thread restrictions of execution/precision guards.

Endurance checks run small networks for 1000 updates and require finite values
and an overall downward loss trend, allowing intermediate rebounds. Real-weight
KV checks compare the full vocabulary logits at selected cache positions;
free-generation quality is checked separately from timing replay. Small models
have limited capabilities, so answer quality may be poor. Use simple English
questions and judge primarily whether sentences are fluent and free of significant
text corruption; relevance and abnormal repetition are additional observations.
Compare against PyTorch at matching floating precision when investigating
differences in inference results.
Long tests are explicitly selected, and missing real-model assets are reported as skipped.
See the [portable CPU workflow](./.github/workflows/operator-accuracy.yml) for CI.

The 2026-11-04 validation passed 685 native CPU/CUDA and 408 portable library tests.
Each configuration passed 17 tool tests and 7 doctests. All 72 training runs
showed an overall loss decrease across 1,000 updates. Real-weight logits for
TinyLlama and Qwen3 passed PyTorch comparisons, and simple English generations
showed no significant text corruption. [Validation and measurement data](./dev-tools/benchmarks/results/2026-11-04.json)

## Repository layout

Main directories and entry points:

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
│   ├── device.rs
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
│   │   ├── cpu_gemm.rs
│   │   ├── cpu_gemm/
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
    │   └── results/
    ├── check_release.ps1
    ├── check_operator_accuracy.ps1
    ├── check_performance.py
    └── common.ps1
```

`src/` contains library and production CLI implementation. `cpu/mod.rs` owns
CPU scheduling and execution contexts; `cpu/profiling.rs` is its optional
instrumentation submodule. `ops/cpu_gemm.rs` provides the CPU matrix kernel
entry points, and `ops/cpu_gemm/` contains packing and compute implementations.

`device.rs` owns device types, strict execution policy, and Tensor placement and synchronization; earlier `autograd::Device` imports remain compatible.
`autograd/` separates Tensor construction, storage, quantization, views, gradient execution, and optimizer updates. `ops/matmul/` groups dot
products, MatVec, fused projections, prefill, backward, and CUDA routing.
`ops/cuda/api/` contains operator-facing wrappers; the private `imp/` directory
contains validation, allocation/dispatch, and isolated FFI declarations.
Public Tensor and operator paths remain available through their module entries.

[`tests/`](./tests/README.md) groups unit tests, cross-module validation, and
explicit profiling tests. Private unit tests are attached to their original
modules with `#[cfg(test)]` and `#[path]`, without widening the public API.
[`dev-tools/`](./dev-tools/README.md) contains developer binaries, benchmark
helpers, check runners, and independent Python references. Cargo target names
and `cargo run --bin ...` commands are unchanged. Generate API documentation
with `cargo doc --open`.

## Limitations

- Built-in checkpoint inference supports the documented Llama and dense Qwen3
  configurations; other model families require an implementation and weight mapping.
- CUDA targets a single device, and operator/dtype coverage varies by backend.
- Release validation covers Windows/x86-64 CPU and a CUDA GPU. ARM64 and the
  nightly-only AVX-512 FP16 path require separate platform validation.
- Low-precision reduction and argmax rounding can change generated token choices.
  Model knowledge and instruction following depend on the loaded weights.
- The library provides neural-network components and a training core, rather
  than a distributed trainer or serving system.

## License

See [LICENSE](./LICENSE).
