# Lumen

> A lightweight deep learning core written in Rust, featuring dynamic autograd, modular neural network components, and a CPU-oriented Llama inference path.

[中文 README](./README_zh-CN.md) · [Repository README](./README.md)

---

## Overview

Lumen is a compact Rust deep learning project. This release keeps the **core library** and a **minimal runnable Llama inference example**, while removing development-time test, benchmark, timing, and sweep code.

The repository is useful in two ways:

- as a **learning-oriented DL core** for understanding tensors, autograd, layers, modules, and optimizers in Rust;
- as a **small LLM inference skeleton** showing a Llama-style model, safetensors loading, tokenizer integration, and KV-cache-based incremental decoding.

This patched branch also includes an **experimental pure-BF16 inference-loading path**. It should currently be viewed as a **pragmatic test / transitional implementation**, not as a finalized design, but it is already part of the runnable example and deserves proper documentation.

> `src/main.rs` is **just a simple example program**. Its purpose is to demonstrate how the library pieces are wired together for local inference. It is **not** intended to be a full production CLI, serving stack, or universal model runner.

---

## Technical Highlights

- **Dynamic autograd engine** built around tensor-based graph construction
- **`Module`-style abstraction** for organizing trainable components
- **Separated layer / op design** for cleaner modeling and easier kernel evolution
- **Llama-family decoder implementation**, including:
  - RMSNorm
  - RoPE
  - causal self-attention
  - GQA via `num_key_value_heads`
  - SwiGLU-style MLP
  - incremental decoding with **KV cache**
- **CPU-oriented inference hot-path optimization**, including:
  - Rayon-backed parallel execution
  - row-major parallel matvec path for decode
  - optimized BF16 prefill path for multi-token prefill (experimental)
  - fused infer-time gate/up/SiLU path in the MLP
  - release profile tuned with `lto`, `panic = "abort"`, and `strip`
- **Efficient checkpoint loading** through `safetensors` + `memmap2`
- **Experimental pure-BF16 direct infer loading** for large Llama projection / embedding matrices
- **Interactive example chat entry** with KV-cache reuse, reset support, and sampling controls
- **Tokenizer integration** via Hugging Face `tokenizers`

---

## Experimental Pure-BF16 Inference Path

This repository currently ships with an **experimental pure-BF16 inference path** meant for local testing and iteration. The overall idea is to:

- keep the project architecture lightweight;
- load selected large matrices through a BF16-oriented infer-time path;
- preserve the existing CPU-first design while reducing memory traffic in practical inference workloads.

In the current patched build, this BF16 path is mainly tied to:

- `set_pure_bf16_infer_loading(true)` in the example entry;
- `ModelLoader::load_llama_weights_direct(...)` for direct checkpoint loading;
- loader-side identification of large matrices such as embeddings, LM head, attention projections, and MLP projections;
- dedicated prefill-side optimization work targeting multi-token BF16 prefill on CPU.

This should **not** be read as a stable, final, or universally optimal implementation yet. It is better described as a **useful testing / stopgap solution** that already works as part of the current branch, while still leaving room for future kernel, layout, and loader redesign.

## Repository Layout

```text
src/
├─ autograd.rs          # Tensor + dynamic autograd core
├─ module.rs            # Module trait/macros
├─ ops/                 # Tensor ops and low-level kernels
├─ layers/              # NN layers and attention components
├─ models/llama.rs      # Llama model implementation
├─ loader.rs            # Safetensors weight loading
├─ tokenizer.rs         # Hugging Face tokenizer wrapper
├─ kv_cache.rs          # Legacy/simple KV cache implementation
├─ optim.rs             # Optimizers
├─ loss.rs              # Loss functions
├─ init.rs              # Parameter initialization helpers
└─ main.rs              # Minimal local inference example
```

---

## Build

Use release mode for practical performance:

```bash
cargo build --release
```

To better utilize your local CPU:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

PowerShell:

```powershell
$env:RUSTFLAGS = "-C target-cpu=native"
cargo build --release
```

---

## Running the Minimal Example

The example program expects explicit weight and tokenizer paths:

```bash
cargo run --release -- \
  --weights path/to/model.safetensors \
  --tokenizer path/to/tokenizer.json
```

Optional arguments:

- `--system`
- `--temperature`
- `--top-p`
- `--repetition-penalty`
- `--recent-window`
- `--max-gen`

The current example entry explicitly enables the experimental BF16 loading path before loading weights:

```rust
set_pure_bf16_infer_loading(true);
ModelLoader::load_llama_weights_direct(&args.weights, &mut model)?;
```

Interactive commands in the example chat loop:

- `/reset` — clear conversation state and KV cache
- `/exit` — quit the program

---

## Important Note on `main.rs`

`src/main.rs` uses a **hard-coded `model_config()`** and a very lightweight CLI flow. This is intentional: it keeps the example easy to inspect and modify.

But it also means:

- the architecture in `model_config()` must match the loaded checkpoint;
- the tokenizer vocabulary and special tokens must be compatible with `vocab_size` and prompt formatting;
- the current BF16 loading path is patch-oriented and depends on the loader correctly recognizing the intended large matrices;
- adapting to other checkpoints may require changing dimensions, layer counts, attention layout, special tokens, prompt-template logic, or BF16 loader matching rules.

So, **`main.rs` should be understood as a simple integration example, not a universal launcher**.

---

## Why This Project Is Interesting

Many Rust ML repositories stop at tensors or MLP demos. Lumen goes further by connecting multiple layers of the stack in one codebase:

- tensor + autograd fundamentals,
- reusable NN modules,
- a Llama decoder implementation,
- checkpoint loading,
- tokenizer bridging,
- stateful autoregressive decoding.

That makes it a solid starting point for studying how a small Rust-native DL / LLM runtime can be assembled end to end.

---

## License

GPL v3.0
