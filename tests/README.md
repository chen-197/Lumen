# Tests

- `unit/`: operator, tensor, optimizer, layer, scheduling, and CLI unit tests,
  arranged by the implementation they exercise.
- `validation/`: cross-module numerical, autograd, training, and endurance checks.
- `profiling/`: explicitly selected developer profiling tests.

These files are attached to their original Rust modules with `#[cfg(test)]` and
`#[path = "..."]`. This keeps private-kernel tests within the module they test,
preserves test names and feature guards, and avoids exposing internals through
the public API. They are compiled by the library and binary test harnesses;
there is no additional duplicate integration-test harness.

```sh
cargo test --release --all-targets
cargo test --release --all-targets --features dev-tools
cargo test --release --doc --features dev-tools
cargo test --release --lib --features dev-tools -- --list --ignored
```

Default tests do not require downloaded model weights. Long tests and profiling
are ignored by default and selected explicitly by the runners in
[`dev-tools/`](../dev-tools/README.md). See the main README for independent
PyTorch checks and CUDA build requirements.

Documentation tests include compile-fail checks that execution/precision guards
cannot cross threads. The accuracy runner and CI execute these separately from
`--all-targets`, which does not select documentation tests.

## Selecting a suite

Run each correctness configuration once. An ignored test is not necessarily a
release requirement: endurance checks, performance smoke tests and legacy
scheduler experiments serve different purposes.

| Suite | Required coverage | Selection |
| --- | --- | --- |
| Daily | All ordinary unit and cross-module tests, including operator boundaries, gradients and short training | `cargo test --release --locked --all-targets --features dev-tools` |
| Release | Portable correctness; native correctness when supported; all nine independent references per configuration; public API state/training; Clippy, format, docs and source package | `check_release.ps1 -NativeCpu`, or `-Cuda` on a CUDA host |
| Endurance | 1000-step training, graph/buffer release, long KV routes and real-weight reference when assets are supplied | Add `-Endurance -ModelDirectory ...` to the release runner, once on the optimized build |
| Performance | Representative warmed forward/backward, four weight types, decoding, request scaling and tail shapes | `check_performance.py`; independent of the correctness runner |
| Investigation | Intermediate thread budgets, precision/request cross products, SIMD timing, old/new scheduler comparisons | Select a named ignored test or use `check_performance.py --extended` |

`-Cuda` includes the native CPU configuration. Without CUDA, `-NativeCpu` selects
native CPU separately from the portable fallback. Native features follow the
host architecture: compilation uses `target-cpu=native`, and x86-64 additionally
enables the optional x86 floating-point and int8 features. Unsupported
architectures are not covered by running more combinations on one x86 machine.

### Coverage retained

- Keep all ordinary correctness cases. Empty/tail/strided tensors, precision
  conversion, quantization scales, forward values, VJPs, frozen parameters,
  retained graphs, accumulation, optimizer state and checkpoints catch different
  failures. A training loss decrease alone cannot replace numerical references.
- Keep all nine Python references: operator FP64, dense cross entropy, autograd
  reuse, gradient flags, accumulation, generic NN, generic prefill and row matmul.
  The Qwen3 reference also checks expanded attention heads, Q/K normalization,
  tied embeddings, full logits and all parameter VJPs against independent FP64
  equations. Their targets differ; none is replaced by the real-model generation check.
- Keep the 1000-step endurance assertions unchanged. Short training is already
  included in ordinary/public API tests. Endurance requires finite values and
  overall loss descent, allowing intermediate rebounds. It also checks lifetime
  release and cache boundaries/reset independently of training.
- Keep real-weight free generation separate from forced-token timing. For a
  routine release with a small model, use simple English questions, such as
  asking for a capital city or a basic fact, across CPU/CUDA and
  F32/F16/BF16/I8 where supported. Small models have limited capabilities and
  may give poor answers, so the primary criteria are fluent sentences and no
  significant text corruption. Record relevance and abnormal repetition as
  additional observations; use PyTorch at matching floating precision to
  investigate inference differences. Model language quality is not a strict
  token-equality gate.

### Repetition removed from routine release work

- Do not rerun the full suite and nine references in both `dev-tools` and
  production-only builds. Check production compilation/package separately;
  rerun feature-gated behavior when that behavior changes. For `cpu-profiling`,
  select its two scope-restoration tests instead of rerunning all library tests.
- Do not repeat the entire long matrix in portable and native builds. Portable
  kernels retain their ordinary numerical references; run portable endurance
  explicitly when fallback execution or lifetime behavior changes.
- Do not run every `--ignored` test. In particular,
  `compare_attention_schedulers` and `compare_model_schedulers` retain legacy
  implementations for ablation and are investigation tools. SIMD and optimizer
  performance smoke tests are also selected by the component being changed.
- Do not cross every loader/cache/precision/CLI option with every device. Keep
  the complete small-fixture loader and CLI tests; use one real-weight example
  per distinct loader route (mmap, stream, serialized I8, parameter copies,
  dynamic I8), and test both devices where device behavior differs. Expand the
  matrix when those paths change or a mismatch appears.
- Do not independently requantize and scan a billion parameters for unrelated
  changes. Small-fixture scale/rounding tests remain mandatory; the full
  checkpoint audit is appropriate for quantizer/loader changes or a new format.
- Keep performance warmup and raw samples. The representative CPU matrix has
  21 processes for a worker count above one, versus 109 extended processes at
  eight workers. Intermediate task budgets, all projection/decode variants and
  precision/request combinations are available through `--extended`.

For whole-model performance, run the eight supported Lumen device/weight
configurations and six matching floating PyTorch configurations once each:
14 processes, with two warmups and five measured fixed-token runs per process.
Use the same prompt, replay tokens, thread settings and timing boundary; report
prefill and decode separately. No PyTorch I8 ratio is implied. This is an initial
comparison: repeat a noisy or material difference with counterbalanced ABBA
ordering before concluding that a change improved or regressed performance.
Eager-attention controls are diagnostic, selected when attention results differ.

The representative suite is not an exhaustive configuration claim. Record every
skip, unavailable backend, failed run and selected suite in the evidence; retain
earlier results rather than replacing them with better-looking samples.
