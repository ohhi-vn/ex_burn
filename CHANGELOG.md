# Changelog

## [Unreleased]

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
- No precompiled NIF binaries (requires Rust toolchain)
- CUDA backend not yet supported
- Training uses numerical gradients (not yet connected to Burn's autodiff)
