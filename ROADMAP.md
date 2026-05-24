# ExBurn Roadmap

## v0.1.0 (Current — Early Alpha)
- Nx.Backend behaviour (basic ops, shape ops, reductions, linear algebra)
- Rust NIF bridge to Burn/CubeCL
- Training loop with numerical gradients
- Nx.Serving integration

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
