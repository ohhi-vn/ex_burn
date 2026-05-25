# ExBurn Roadmap

## v0.1.0 (Current — Early Alpha)
- Nx.Backend behaviour (basic ops, shape ops, reductions, linear algebra)
- Nx.Defn.Compiler — GPU-accelerated defn expression evaluation
- Rust NIF bridge to Burn/CubeCL
- Training loop with numerical gradients
- Nx.Serving integration

### Completed Improvements
- Axon model compilation with GPU forward pass via `Nx.Defn.jit_apply` + `ExBurn.Defn.Compiler`
- Glorot/Xavier parameter initialization
- Keras/PyTorch-style model summary with layer-by-layer inspection
- Layer freeze/unfreeze for fine-tuning
- Device management (`to_device/2`) for CPU ↔ GPU parameter transfer
- Training loop: batch shuffling, Nesterov momentum, weight decay, gradient accumulation
- Training loop: accuracy tracking, ETA/progress reporting, `train_step/3` for custom loops
- Improved numerical gradient (`:numerical_batch` method, 2x fewer forward passes)
- `evaluate/2` with accuracy tracking and proper partial batch handling

## v0.2.0 — Precompiled NIFs
- rustler_precompiled with GitHub Actions
- Cross-compiled binaries for aarch64-apple-ios, aarch64-linux-android
- Remove Rust toolchain requirement for end users

## v0.3.0 — Autodiff Integration
- Connect training to Burn's Autodiff backend (replace numerical gradients)
- Gradient checkpointing for memory-constrained devices
- Support for fine-tuning on mobile

## v0.4.0 — Nx.Serving
- Nx.Serving integration for concurrent batched inference
- Bumblebee-style pipeline compatibility

## v0.5.0 — CUDA Support
- CUDA backend via Burn/CubeCL
- NVIDIA GPU training and inference

## Future
- WebGPU backend
- ROCm (AMD GPU) support
- Model zoo with pre-trained mobile-optimized models
