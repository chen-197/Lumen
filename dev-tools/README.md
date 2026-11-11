# Development tools

This directory contains benchmark programs, explicit diagnostics, and independent
Python references. Library implementation lives in `src/`; Rust test bodies live
in [`tests/`](../tests/README.md).

## Layout

- `benchmarks/`: warmed kernel, scheduling, backward, and model benchmarks.
  Multi-file programs keep `main.rs` and their helper modules in one directory.
- `benchmarks/results/`: dated measurement data with source fingerprints, build
  metadata and individual samples; README performance tables use these files.
- `bin/`: explicit developer diagnostics such as `manual_state_check`.
- `check_*.ps1`: accuracy, state, endurance, and release check runners.
- `check_*.py`: independent numerical and training references.
- `common.ps1`: shared Cargo output-directory discovery.

Cargo registers these Rust programs in the root `Cargo.toml`. Enable the
`dev-tools` feature and use their existing target names:

```sh
cargo run --release --features dev-tools --bin cpu_backward_bench -- --threads 4 --tasks 4
cargo run --release --features dev-tools --bin cpu_scaling_bench -- --kind rms_norm --dtype bf16 --m 16 --k 128 --threads 4 --tasks 2
cargo run --release --features dev-tools --bin manual_state_check -- --device cpu
```

The production inference CLI and weight conversion CLI remain in `src/`.
The RMSNorm scaling case uses `--m` rows of `--k` values and supports the same
shared-pool/task/request controls. Its input and weight both use `--dtype`,
including I8; `--n` and attention head options do not affect this case.
`--kind rms_norm_backward` retains one training graph and measures gradient
reset, upstream-seed cloning and the complete input/weight VJP. It uses the same
row, dtype, pool and request arguments. Both input-gradient rows and
weight-gradient reduction observe the policy. Set
`LUMEN_BENCH_DUMP=1` to include represented inputs, seed and gradients after timing.
`--kind layer_norm` measures affine LayerNorm with weight and bias, using the
same row/last-dimension arguments. It accepts F32/F16/BF16 storage and produces
F32 outputs. Per-request records include input/parameter storage dtypes after
the run, so implicit storage promotion is visible in older baselines.
`--kind layer_norm_backward` retains an affine LayerNorm graph and measures
gradient reset, seed cloning and the full input/weight/bias VJP. It accepts
F32/F16/BF16 storage and the same row, dimension, pool and request controls;
`LUMEN_BENCH_DUMP=1` exports inputs, parameters, seed and gradients outside timing.
The `cpu-profiling` feature enables library instrumentation used by benchmarks;
it depends on `dev-tools` and is disabled by default.

`matmul_transposed_rhs.total` includes `matmul_transposed_rhs.kernel`, while `rms_norm` measures the full
normalization call. These regions are inclusive and must not be summed as
independent shares of total time.

## Check runners

Use PowerShell 7 and a Python environment with PyTorch installed. Real-weight
endurance references also require Transformers and compatible local model
assets. Select the Python executable explicitly when needed:

```powershell
./dev-tools/check_operator_accuracy.ps1 -Python python
./dev-tools/check_manual_state.ps1 -Python python
./dev-tools/check_operator_accuracy.ps1 -Cuda -NativeCpu -Python python
./dev-tools/check_endurance.ps1 -Python python -ModelDirectory "path/to/model"
./dev-tools/check_release.ps1 -Cuda -Python python
./dev-tools/check_release.ps1 -NativeCpu -Endurance -Python python -ModelDirectory "path/to/model"
```

Paths are resolved from the repository and script locations. Runners accept
`-OutputDirectory`; relative output paths are relative to the repository root.
Build output discovery follows Cargo metadata, including `CARGO_TARGET_DIR` and
Cargo configuration. Model locations and Python executables are caller inputs;
no user profile, downloaded model, or Python installation path is hard-coded.
Temporary logs, samples, and reference results belong under `target/` or an
explicit output directory. Reviewed datasets published with documentation belong
in `benchmarks/results/`; include source/build identities and remove machine-specific paths.

Endurance tests are opt-in. The runner reports real-weight tests as skipped when
`-ModelDirectory` is absent. Timing comparisons should run serially with warmup,
without other builds, tests, or model processes competing for resources.

The accuracy runner includes `check_qwen3.py`, an independent FP64 reference for
Q/K normalization, attention heads wider than `hidden_size / num_heads`, tied
embeddings, full forward logits and parameter gradients. Like the other ordinary
numerical tests, it uses small fixtures and requires no downloaded weights.

Matrix benchmarks distinguish `matmul` (`A @ B`) from `matmul_transposed_rhs`
(`A @ Wᵀ`). The projection/scaling benchmarks select the latter with
`--kind matmul_transposed_rhs`; `generic_prefill_bench --op` accepts either name.
For a custom matrix shape, use `--op matmul --shape 1,1024,1536 --layout 0`
(`A=[1,1024]`, `B=[1024,1536]`); layout 0 is row-major and layout 1 is strided.
Custom shapes use case 0; the default benchmark cases are unchanged.
Native F16/BF16/I8 layouts are constructed after precision conversion; matrix
records include `input_strides` and `rhs_strides`. Older low-precision layout
labels described the array before casting and do not establish native strides.
The standard product has corresponding `matmul.total` / `matmul.kernel` profiling scopes.
Historical result files retain their original operator labels; the 2026-10-29
`matmul` rows measured the transposed-RHS operation.

`prefill_decode_bench` reads an adjacent `config.json` or an explicit `--config`.
It supports the documented Llama/dense Qwen3 configurations. Qwen3 text prompts
use ChatML with thinking disabled unless `--enable-thinking` is supplied.

## Bounded performance runs

Build the benchmark binaries once, then time them serially. `--binary-directory`
must also contain required runtime libraries. Choose the worker count explicitly;
the runner does not assume a particular CPU or CUDA installation.

```powershell
cargo build --release --locked --features dev-tools --bins
python ./dev-tools/check_performance.py --threads 8 --plan
python ./dev-tools/check_performance.py --threads 8 --binary-directory "path/to/release" --output-directory "target/performance-new"
```

The default matrix contains 21 processes at more than one worker: forward and
backward at one/full worker budgets, four projection/decoding precisions,
single/multiple requests, and tail shapes. Programs warm up before measurement;
the runner preserves raw samples, executable hashes, commands and exit status in
a new directory. It does not build binaries during measurement or declare a
correctness pass from a successful timing run. Hardware, build flags and runtime
library versions should accompany shared timing results.

Use `--select argmax` for a component subset, or `--extended` for intermediate
task budgets and full precision/request combinations (109 processes at eight
workers). Neither suite invokes old scheduler model ablations or downloads
weights. Whole-model timing and free-generation quality checks remain separate.
See [test selection](../tests/README.md#selecting-a-suite) for the release policy
and conditions that warrant expanding a matrix.
