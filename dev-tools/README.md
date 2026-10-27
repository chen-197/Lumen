# Development tools

This directory contains benchmark programs, explicit diagnostics, and independent
Python references. Library implementation lives in `src/`; Rust test bodies live
in [`tests/`](../tests/README.md).

## Layout

- `benchmarks/`: warmed kernel, scheduling, backward, and model benchmarks.
  Multi-file programs keep `main.rs` and their helper modules in one directory.
- `bin/`: explicit developer diagnostics such as `manual_state_check`.
- `check_*.ps1`: accuracy, state, endurance, and release check runners.
- `check_*.py`: independent numerical and training references.
- `common.ps1`: shared Cargo output-directory discovery.

Cargo registers these Rust programs in the root `Cargo.toml`. Enable the
`dev-tools` feature and use their existing target names:

```sh
cargo run --release --features dev-tools --bin cpu_backward_bench -- --threads 4 --tasks 4
cargo run --release --features dev-tools --bin manual_state_check -- --device cpu
```

The production inference CLI and weight conversion CLI remain in `src/`.
The `cpu-profiling` feature enables library instrumentation used by benchmarks;
it depends on `dev-tools` and is disabled by default.

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
Generated logs, samples, and reference results belong under `target/` or an
explicit output directory, rather than the source tree.

Endurance tests are opt-in. The runner reports real-weight tests as skipped when
`-ModelDirectory` is absent. Timing comparisons should run serially with warmup,
without other builds, tests, or model processes competing for resources.

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
