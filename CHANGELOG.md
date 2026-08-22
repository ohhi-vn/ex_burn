# Changelog

## [0.6.0]

### Fixed
- **Silent wrong results replaced with errors**: `Nx.Backend` callbacks that had no
  implementation (`fft`, `sort`, `argsort`, `triangular_solve`, all `window_*` ops,
  `indexed_add/put`, `put_slice`, `to_batched`) now raise a structured
  `ExBurn.Error` instead of silently returning the input tensor or garbage
- **`argmax`/`argmin`** now return *indices* (via exact Nx computation) instead of
  the max/min value
- **Reductions respect `:axes`**: partial `sum` / `reduce_max` / `reduce_min`
  reductions produce correctly shaped results; `all`/`any` compare against zero;
  `product` is computed exactly instead of via `exp(sum(log x))`
- **`dot`/`transpose`/`concatenate`** honor their axis/contract arguments; general
  contractions are computed exactly via Nx, common cases keep the NIF fast path
- **`remainder`/`quotient`** use correct truncating semantics; `reverse` and
  `slice` produce correct results for arbitrary axes
- **`select`** works for arbitrary predicates and shapes (was arithmetic encoding)
- **Dtype handling is honest**: dtype mapping is centralized in `ExBurn.Tensor`
  and faithful (no silent `f16→f32` bit reinterpretation); the backend converts
  non-f32 inputs to f32 *value-preserving* at the boundary; direct tensor APIs
  raise a structured error for dtypes the NIF cannot store yet
- **Defn compiler** raises `ExBurn.Error` on conversion failures instead of
  creating fake tensor references that crashed later in unrelated places
- **`Model.forward/2`** passes nested params (as Axon expects) — previously sent
  the flattened map and failed
- **`Model.predict/2`/`forward/2`** reuse a predict function built once at
  compile time (was re-tracing the Axon graph on every call) and silence Axon
  deprecation warnings by using the proper parameter structure
- **`Training.evaluate/3`** no longer divides by zero on empty datasets
- **`:clip_norm` / `:clip_value`** accept integers (previously silently ignored
  unless float)
- **NIF registration**: `debug_nif_count/0` is registered again, so loading
  `ExBurn.Nif` succeeds and `ExBurn.nif_function_count/0` reports real counts
- **Defn compiler could not evaluate any traced expression**: an overly broad
  fallback clause shadowed the generic expression evaluator, and op arguments
  were evaluated twice. The evaluator now handles nx 0.13's `:while`, `:cond`,
  `:fun` and `:hook` node shapes, resets its cache between loop iterations
  (previously loops never progressed), supports hooks resolved by name, and
  converts host-callback results back to plain tensors
- **`Nx.Serving` integration restored for nx >= 0.10**: the serving callback
  returns the new `{:execute, fun, state}` contract; the old shape made
  `Nx.Serving.run/2` raise
- **`Backend.inspect/2` crashed** whenever an ExBurn tensor was inspected
  (`%Inspect.Opts{}` is not a keyword list)
- **`constant/3` panicked on multi-element shapes**: the scalar value is now
  replicated to fill the declared shape (defn constants share the fix)
- **`conv2d_tensor/4` passed an invalid `:stride` option to `Nx.conv`**
  (the correct key is `:strides`) — convolution raised for every input
- **`stack/3` crashed in both call conventions**; it now accepts wrapped
  tensors and bare backend structs
- **`allocate_gpu/3` always failed**: ExCubecl validates byte counts against
  shape × dtype, so zero-filled storage is provided instead of an empty
  binary (also fixed for kernel auto-output buffers)
- **`CubeclBridge.host_to_device/2`** passes atom dtypes as ExCubecl expects
  (was sending Nx type tuples, which raised a function-clause error)
- **`BurnBridge.transpose/3`** builds a full axis permutation — transposing
  rank > 2 tensors no longer raises

### Changed
- **Rust NIF no longer syncs GPU→CPU after every operation** — result shapes are
  read from tensor metadata; data is only materialized by
  `tensor_to_binary`. This removes the biggest serialization bottleneck.
- **Broadcasting implemented properly** via Burn's `expand` (was an incorrect
  reshape); GPU probes for Metal/Vulkan are panic-guarded like CUDA
- Release profile builds with `opt-level = 3` + thin LTO (was `opt-level = 1`);
  unused Rust dependencies removed (`serde`, `serde_json`, `thiserror`,
  `log`, `env_logger`, `libc`)
- Rust license field corrected to Apache-2.0 (matches the LICENSE file)
- Training progress output uses `Logger` instead of `IO.puts`; callbacks no
  longer send undocumented messages or leak unlinked state
- Removed `HistoryCallback.get_history/0` stub (always returned `[]`);
  use `get_history(pid)` with the `:history_pid` from metrics
- `ExBurn.smoke_test/0` restores the caller's previous backend instead of
  hardcoding `Nx.BinaryBackend`
- `build.sh` is portable (no hardcoded paths) and validates its argument;
  removed personal scratch script `find_api.sh`

## [0.5.0]

### Added
- **ExCubecl v0.5.0 compatibility**: Updated ex_cubecl dependency to `>= 0.5.0` with new `jason` dependency for kernel params JSON encoding
- **Command struct support**: `ExCubecl.Command` typed struct for pipeline commands (from ex_cubecl v0.5.0)
- **`:u8` dtype support**: Picks up new `:u8` (8-bit unsigned integer) dtype from ex_cubecl v0.5.0

### Changed
- Updated ex_cubecl minimum version from `>= 0.4.0` to `>= 0.5.0`
- Added `jason ~> 1.4` dependency (required by ex_cubecl v0.5.0 for kernel parameter encoding)
- Fixed `async_submit/1` type spec to accept `String.t()` (matching ex_cubecl's `submit/1`) instead of `ExCubecl.Command.t()`
- Updated README and guides to reference ex_cubecl v0.5+

### Fixed
- Fixed `describe/3` compile error in `test/cuda_test.exs` (reverted to `describe/2` with `@tag` on individual tests)
- **Model compilation improvements**: GPU forward pass via `Nx.Defn.jit_apply` + `ExBurn.Defn.Compiler`
- **Glorot/Xavier initialization**: Proper weight initialization for all model parameters
- **Model summary**: Keras/PyTorch-style layer-by-layer summary with `ExBurn.Model.summary/1`
- **Layer freeze/unfreeze**: `freeze/2`, `unfreeze/2`, `frozen?/2` for fine-tuning workflows
- **Device management**: `to_device/2` for CPU ↔ GPU parameter transfer
- **Weight decay**: L2 regularization support in model compilation and training
- **Batch shuffling**: `:shuffle` option (default `true`) in training loop
- **Nesterov momentum**: `:nesterov` option for SGD optimizer
- **Gradient accumulation**: `:accumulate_gradients` option for effective larger batch sizes
- **Accuracy tracking**: `:accuracy` option computes classification accuracy during training
- **Improved progress reporting**: ETA, samples/sec, epoch time in training output
- **Custom training loops**: Public `train_step/3` and `compute_gradients/3` functions
- **Improved numerical gradients**: `:numerical_batch` method (~2x faster than `:numerical`)
- **Better evaluation**: `evaluate/2` with accuracy tracking and proper partial batch handling
- **Training optimization guide**: Comprehensive guide covering optimizers, LR schedules, gradient clipping, weight decay, batch size selection, memory optimization, and troubleshooting

### Changed
- Updated all guides with accurate, detailed documentation
- Updated README with current feature status and guide links
- Updated ROADMAP with completed improvements

## [0.1.0] — Initial Release

### Added
- Initial Nx.Backend behaviour implementation (basic ops, shape ops, reductions, linear algebra)
- Rust NIF bridge to Burn Autodiff<CubeCL> via rustler
- ExBurn.BurnBridge for direct Burn tensor operations
- ExBurn.CubeclBridge for GPU context management
- ExBurn.Model for model compilation and management
- ExBurn.Training with SGD, Adam, RMSprop optimizers, LR scheduling, gradient clipping, callbacks
- ExBurn.Error structured error type
- CI pipeline (GitHub Actions) with Elixir tests, Rust fmt/clippy
- Guides: Getting Started, Training, Mobile Deployment, Architecture

### Known Limitations
- Training uses numerical gradients (not yet connected to Burn's autodiff)
- No precompiled NIF binaries (requires Rust toolchain until v0.2.0)
