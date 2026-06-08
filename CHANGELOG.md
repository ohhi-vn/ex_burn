# Changelog

## [Unreleased]

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
