# Lumen

> A lightweight Rust deep learning core with dynamic autograd, modular neural network building blocks, and a CPU-oriented Llama inference path.

[中文 README](./README_zh-CN.md) · [English README](./README_EN.md)

---

## Overview

Lumen is a compact deep learning project written in Rust. This release version keeps the **core library** and a **minimal runnable Llama inference example**, while removing test, benchmark, timing, and sweep code used during development.

The repository is useful in two roles:

- as a **learning-oriented DL core** for understanding tensors, autograd, layers, modules, and optimizers in Rust;
- as a **small LLM inference skeleton** centered on a Llama-style architecture, safetensors loading, tokenization, and KV cache based incremental decoding.

This patched branch also includes an **experimental pure-BF16 inference-loading path**. It is currently best understood as a **pragmatic test build / temporary solution** rather than a finalized design, but it is already integrated into the runnable example and documented here on purpose.

> `src/main.rs` is intentionally a **simple example program**. It demonstrates how to wire the library together for local inference, but it is **not** intended to represent a complete production CLI, serving stack, or general model runner.

---

## Technical Highlights

- **Dynamic autograd engine** with tensor-based computation graph construction
- **`Module`-style neural network abstraction** for composing trainable components
- **Layer / op separation** for cleaner modeling and lower-level kernel evolution
- **Llama-family decoder implementation** with:
  - RMSNorm
  - RoPE
  - causal self-attention
  - GQA (`num_key_value_heads`)
  - SwiGLU-style MLP
  - incremental decoding with **KV cache**
- **CPU-oriented inference hot path optimizations**, including:
  - Rayon-backed parallel computation
  - row-major parallel matvec path for decode
  - optimized BF16-prefill path for multi-token prefill (experimental)
  - fused infer-time gate/up/SiLU path in MLP
  - release profile tuned with `lto`, `panic = "abort"`, and `strip`
- **Efficient weight loading** from `safetensors` via `memmap2`
- **Experimental pure-BF16 direct infer loading** for large Llama projection / embedding matrices
- **Interactive example chat entry** with KV-cache reuse, reset support, and sampling controls
- **Tokenizer integration** through Hugging Face `tokenizers`

---

## Experimental Pure-BF16 Inference Path

This repository currently ships with an **experimental pure-BF16 inference path** intended for local testing and iteration. The key idea is:

- keep the project architecture lightweight;
- load selected large matrices through a BF16-oriented infer-time path;
- preserve the existing CPU-first design while reducing memory traffic in practical inference scenarios.

In the current patched build, the BF16 path is especially tied to the following pieces:

- `set_pure_bf16_infer_loading(true)` in the example entry;
- `ModelLoader::load_llama_weights_direct(...)` for direct weight loading;
- loader-side identification of large matrices such as embeddings, LM head, attention projections, and MLP projections;
- dedicated prefill-side optimization work aimed at improving multi-token BF16 prefill behavior on CPU.

This is **not yet presented as a stable, final, or universally optimal implementation**. It should be treated as a **testing / transitional solution** with real practical value, but still subject to further kernel, layout, and loader evolution.

## Repository Layout

```text
src/
├─ autograd.rs          # Tensor + dynamic autograd core
├─ module.rs            # Module trait/macros
├─ ops/                 # Tensor ops and kernels
├─ layers/              # NN layers and attention building blocks
├─ models/llama.rs      # Llama model implementation
├─ loader.rs            # Safetensors weight loading
├─ tokenizer.rs         # Hugging Face tokenizer wrapper
├─ kv_cache.rs          # Legacy/simple KV cache implementation
├─ optim.rs             # Optimizers
├─ loss.rs              # Loss functions
├─ init.rs              # Parameter initialization helpers
└─ main.rs              # Minimal example entry for local inference
```

---

## Build

Use release mode for practical performance:

```bash
cargo build --release
```

For better CPU utilization on your local machine:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

PowerShell:

```powershell
$env:RUSTFLAGS = "-C target-cpu=native"
cargo build --release
```

---

## Minimal Example Usage

The example program accepts model weights and a tokenizer file:

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

The current example entry enables the experimental BF16 loading path before model loading:

```rust
set_pure_bf16_infer_loading(true);
ModelLoader::load_llama_weights_direct(&args.weights, &mut model)?;
```

Interactive commands in the example chat loop:

- `/reset` — clear conversation state and KV cache
- `/exit` — quit the program

---

## Important Note About `main.rs`

`src/main.rs` uses a **hard-coded `model_config()`** and a very lightweight CLI flow. This is deliberate: it keeps the example easy to read.

But it also means:

- the model architecture in `model_config()` must match the loaded weights;
- tokenizer vocabulary must be compatible with the configured `vocab_size` and prompt format;
- the current BF16 loading path is patch-oriented and depends on the loader recognizing the intended large matrices correctly;
- adapting to other checkpoints may require updating dimensions, layer counts, attention layout, special tokens, prompt template logic, or BF16-loader matching rules.

In other words, **`main.rs` is a simple integration example, not a universal launcher**.

---

## Why This Project Stands Out

Unlike many toy Rust ML repos that stop at tensors or MLP demos, Lumen connects several layers of the stack together in one codebase:

- tensor + autograd fundamentals,
- reusable NN modules,
- Llama decoder architecture,
- checkpoint loading,
- tokenizer bridging,
- stateful autoregressive decoding.

That makes it a good foundation for studying how a small Rust-native DL/LLM runtime can be assembled end to end.

---

## License

GPL v3.0
