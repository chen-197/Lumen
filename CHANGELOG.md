# Changelog

## 0.1.7 — 2026-10-28

### Added

- Architecture-independent scaled dot-product attention for independently
  projected Q/K/V, MHA/GQA/MQA, cross-attention, broadcast boolean/additive masks,
  and explicit causal query offsets. Fully masked rows return zero context and
  score gradients. Existing self-attention training reuses the shared core.
- LayerNorm over arbitrary trailing dimensions, optional weight/bias, floating
  CPU/CUDA forward/backward and F32 outputs. Centered statistics use F64
  reductions. Independent PyTorch FP64 checks include strided inputs, empty
  batches, frozen gradients and training without positional encoding.

- Explicit long-test suite for 1,000-step SGD/momentum/Adam training, graph and
  CUDA buffer ownership release, 513-token full/chunked/tokenwise KV inference,
  and real TinyLlama weights with independent PyTorch references. Missing model
  assets are reported as skipped by the runner; explicitly selected model tests
  fail instead of silently succeeding. Run with `dev-tools/check_endurance.ps1` or
  opt into `dev-tools/check_release.ps1 -Endurance -ModelDirectory ...`.
- Scoped CPU kernel and inference contexts that reuse caller-owned Rayon pools,
  support per-kernel task budgets, and separate prefill/decode policies.
- Explicit F32 master weights and dynamic I8 master-weight calibration for SGD
  and Adam. Native CPU/CUDA F16 and BF16 optimizer-state updates use F32 arithmetic.
- Explicit opt-in dynamic I8 inference activation quantization; the default
  preserves the declared floating activation dtype.
- CPU projection, decode-working-set and scaling benchmarks, raw inference
  samples, task-budget sweeps and optional CPU profiling.
- Default numerical/gradient regressions, independent PyTorch FP64 references,
  and portable CPU accuracy CI, including dense-target cross entropy.
- Explicit `Tensor::backward_with_grad()` for shaped upstream gradients, including
  strided seeds and retained graphs, and `backward_with_cuda_grad()` for resident
  F32 CUDA seeds.

### Improved

- Split Tensor methods into storage, quantization, device, gradient, creation,
  view, checkpoint, and optimizer modules. Group matrix-product kernels and
  CUDA argument validation/FFI dispatch by operation family while preserving
  public Tensor/operator paths. Remove the unreachable duplicate CUDA-disabled
  implementation; portable builds continue to use the existing stub module.

- Separate runtime implementation (`src/`), unit/validation/profiling tests
  (`tests/`), and developer diagnostics/benchmarks/reference scripts
  (`dev-tools/`). Preserve public module paths and Cargo target names. Check
  runners discover build outputs through Cargo metadata, including configured
  target directories.

- Native/mixed lowp batch-matmul weight gradients update contiguous columns
  together and convert each lhs value once per reduction row. Each gradient
  keeps its original row accumulation and I8 scale multiplication order;
  caller-owned scheduling and task budgets remain in force.

- CPU matmul/batch-matmul backward observes caller-owned Rayon pools, task
  budgets and nested-kernel guards. Native lowp loops keep their reduction
  order; F32/mixed fallbacks reuse disjoint GEMM scheduling. Warmed public-API
  backward benchmarks and full mixed-dtype BMM VJPs have independent references.

- General and batched matrix products support zero reduction/output/leading
  axes across CPU/CUDA and all dtype pairs. Empty reductions produce shaped
  zeros; zero VJPs preserve frozen inputs and earlier gradient contributions.
  Empty CUDA tensors never request nonexistent operand buffers.

- CPU batch-matmul and mixed-precision gradient fallbacks reuse two bounded,
  caller-thread F32 conversion buffers. Native values are refreshed per call;
  oversized operands and deeper reentry use temporary allocations. General
  prefill benchmarks support case filters and opt-in conversion/GEMM profiling.
- General CPU F32 matrix products consume strided weights without a full F32
  panel copy and schedule disjoint head/row/column jobs within the caller's
  budget, with coarser GEMM grain to amortize packing and worker wakeup.
- F32 CPU scaled-dot-product inference views grouped KV heads, bounds retained
  score scratch to 256 KiB per worker, balances query tiles, and skips keys beyond
  the causal block. Lowp, CUDA and gradient routes keep their precision contracts.
- Warmed general-shape prefill benchmarks and independent FP64 forward/gradient
  references cover caller budgets, layouts, rectangular attention and masks.

- CPU KV head repetition accepts non-contiguous native storage in inference,
  preserving logical element order for floating and I8 tensors.
- Shared attention internals retain frozen/partially frozen I8 projection
  execution; the new public masked-attention API accepts floating Q/K/V.

- Work-aware scheduling for integrated CPU MatVec, fused projections, argmax,
  prefill, attention and low-precision optimizer kernels. Large jobs retain
  Rayon work stealing; small and nested numerical jobs avoid extra dispatch.
- Long-query attention partitions independent query/head outputs without
  changing causal-mask offsets or requiring shared output locks.
- CPU F16/I8 projection weight reuse and vocabulary argmax; CUDA tiled
  floating-activation/I8-weight GEMM keeps bounded temporary unpacking storage.

### Fixed

- Validate every parameter's gradient shape before SGD/Adam update weights,
  moments, master weights or step counters, including gradients edited through
  `grad_mut()`. Invalid gradients no longer silently truncate CPU updates or
  leave partial optimizer state. Taking raw data clears old host/CUDA gradients
  and leaves readable empty F32 storage.

- Isolate shared CUDA backward seeds and gradient accumulators so repeated
  backward calls add contributions linearly without modifying another tensor's
  gradient or the caller's seed. Uniquely owned accumulators retain in-place
  updates. `grad_ref()` now materializes resident gradients on first access,
  consistently with the other gradient getters.

- Respect `no_grad` for single-input concatenation and full-range slicing of
  tracked tensors, including intermediate nodes. Empty slices preserve source
  storage and I8 scales without reading all input values; CUDA zero gradients
  accumulate on the device without materializing earlier resident gradients.

- Concatenate empty and nonempty CUDA inputs without requesting empty device
  buffers or materializing inference outputs on the host. Preserve native
  inference dtypes, I8 scale promotion, shaped empty gradients and gradient
  accumulation when an input appears more than once.

- Quantize finite values correctly with tiny positive manual I8 scales whose
  reciprocals overflow. CPU tensor conversion, optimizer stores and the offline
  converter use a division fallback; ordinary scales retain their existing path.
- Match scalar half-away-from-zero I8 rounding in AVX2/AVX-512 arithmetic and
  optimizer kernels and CUDA storage conversion. Clamp SIMD values before
  integer conversion so large positive updates cannot flip to negative saturation.

- Keep public no-grad, strict-device, precision, runtime-dtype, parameter
  quantization, initialization and CUDA-selection guards on their creating
  thread. Their types no longer implement Send/Sync, preventing a drop on
  another thread from leaking or replacing thread-local execution settings.
- Flatten empty batches using explicit trailing dimensions instead of an
  ambiguous zero-size inferred reshape. Preserve dtype, device and shaped
  empty input gradients.
- Empty CUDA reshape/permute/concat gradients do not upload zero-length
  buffers. Empty training slices skip CUDA launches, avoid division by zero
  for zero-width inputs, and preserve existing input gradient contributions.

- Mixed F32×BF16 AVX-512 single/dual/triple dot kernels preserve F32 activation
  values instead of rounding vector blocks to BF16. Mixed F32/F16×BF16 prefill
  also avoids additional BF16 rounding. Native BF16×BF16 dot paths remain available.
- Eleven CPU low/mixed-precision matmul row fallbacks observe the caller's pool
  and task budget, avoiding unscoped Rayon row loops and nested MatVec dispatch.
  Independent FP64 checks cover all 16 dtype pairs, layouts and task budgets.
- F32 CUDA inference activations select their resident kernel before requesting
  a host compute view, eliminating discarded SiLU input downloads in prefill.
- CUDA binary dispatch rejects F32 inputs before probing low-precision host
  snapshots, avoiding discarded device downloads in F32 residual additions.
- CUDA matrix multiplication skips unsupported F32/mixed host-storage probes,
  retaining resident inference inputs while selecting its fallback kernel.
- GPU-resident tensors now export, copy, detach and move back to CPU using
  current storage. Native host views preserve dtype and fixed I8 scales;
  CUDA snapshots own distinct buffers and pending gradients survive CPU moves.
- Low-precision parameter replacement clears stale CUDA gradients. Checkpoint
  loading validates every payload before replacing parameters, rejects invalid
  I8 scales, and saving reports final buffered-write failures.
- Mixed host/resident CUDA gradient accumulation preserves every contribution
  and keeps cached totals consistent. Mutable host-gradient access invalidates
  the device mirror and CUDA consumers re-upload the edited value on demand.
- Frozen backward destinations can retain shared read views. CUDA gradient
  upload/accumulation failures now fail explicitly instead of dropping prior
  contributions or silently substituting the newest gradient.
- Operator backward accumulation now respects each input's `requires_grad` on
  CPU and CUDA. Frozen inputs retain no new gradients, including when a shared
  optimizer parameter list contains both trainable and frozen weights. Explicit
  manual gradient assignment retains its existing behavior.
- Fused QKV training sets gradient metadata separately for each projection;
  frozen projections do not build a backward graph.
- Mixed-storage/differing-scale concatenation of constants stays graph-free.
  Low-precision and mixed CUDA training concatenation now retains device-resident
  F32 gradients instead of going through a host-backed backward path.
- CPU convolution forward/backward preserve logical storage order. Backward
  reshapes standard-layout weights and upstream gradients, supports
  transposed/strided inputs, and uses compute views rather
  than reading potentially absent raw F32 storage. CPU convolution compute views
  preserve the declared input dtype when dtype copies are disabled.
- Retained graphs no longer propagate stale intermediate gradients on repeated
  backward calls or when separate losses share a subgraph. Leaf gradients still
  accumulate until cleared.
- CUDA backward closures use weak output references in shape, arithmetic,
  activation, normalization, embedding, RoPE, convolution, pooling and fused
  projection paths, allowing discarded training graphs to be released.
- Ready F32 read-only views permit nested reads of aliased operands without
  requesting a conflicting mutable borrow.
- Logical element ordering across arithmetic, matmul, activations, optimizers,
  reshape, RMSNorm and CUDA uploads of transposed/strided tensors.
- Reentrant F32 compute views of the same cached low-precision parameter no
  longer request a conflicting mutable borrow. Aliased loss inputs accumulate
  both gradient contributions.
- MSE backward reads native inputs through compute views, preserves their
  declared dtype, supports aliases, and only writes requested gradients.
- Stable logits-based CPU/CUDA cross entropy for very small target probabilities.
- Dense-target cross entropy now uses `sum(target) * softmax - target` for logits
  gradients, supports target gradients, includes signed weights in the forward
  expression, and only writes requested gradients. Targets are not normalized
  implicitly; gradients use the actual represented low-precision values.
- Adam bias correction counts effective updates separately for each parameter;
  missing gradients and present zero gradients keep their distinct semantics.
- CUDA activation dtype conversion and selected low-precision inference paths.
- UTF-8 byte-fallback output boundaries: raw token IDs/text remain available,
  while display output waits for a complete character prefix.

### Compatibility and limitations

- `backward()` starts each traversal with a fresh all-ones seed. Replace
  `output.add_grad(seed); output.backward()` with
  `output.backward_with_grad(seed)` for custom upstream gradients. Intermediate
  gradients describe the latest traversal; leaf gradients accumulate. Reusing
  a graph does not recompute forward values after parameter changes.
- Tensor/graph handles remain on their calling thread; the CPU context submits
  numerical slices to workers. It does not make Tensor `Send`/`Sync`.
- CPU scheduling defaults remain caller-controlled; no host-specific worker
  count, affinity policy, automatic calibration or request fairness is imposed.
- Direct greedy argmax can compare F32 accumulated scores, whereas materialized
  F16/BF16 logits introduce output rounding and may create ties. Exact generated
  token equality across these routes is not guaranteed.
- Fixed-scale I8 and low-precision states can lose small updates. F32 masters
  remain opt-in; CUDA allocation slots do not yet guarantee half-sized state memory.
- Model language ability is evaluated separately from implementation accuracy.
  Small-model repetition reproduced by the reference implementation is not, by
  itself, evidence of a Lumen arithmetic defect.

### Release checks

Run `dev-tools/check_release.ps1` for portable CPU validation and local package
verification. Add `-Cuda` on a machine with a working CUDA runtime for native
CPU/CUDA accuracy checks. Python must already have PyTorch installed.
This script does not publish, push, tag or upload anything.
