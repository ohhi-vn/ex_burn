# ExBurn

[![CI](https://github.com/ohhi-vn/ex_burn/actions/workflows/ci.yml/badge.svg)](https://github.com/ohhi-vn/ex_burn/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Hex.pm](https://img.shields.io/hexpm/v/ex_burn.svg)](https://hex.pm/packages/ex_burn)
[![Documentation](https://img.shields.io/badge/hexdocs-docs-purple.svg)](https://hexdocs.pm/ex_burn)

> **Status:** Early development. Not yet ready for production use.

**ExBurn** is a middle layer between [Nx](https://github.com/elixir-nx/nx) and [Burn](https://github.com/tracel-ai/burn) that enables GPU-accelerated ML/DL on mobile and desktop devices.

## Architecture

```
Axon model
   ↓
Nx.Defn graph
   ↓
ExBurn.Backend (Nx.Backend behaviour)
   ↓
ExBurn.Nif (Rustler NIF) ←→ ExCubecl (GPU buffers, kernels, pipelines)
   ↓
Burn Autodiff<CubeCL> (Rust)
   ↓
CubeCL kernels
   ↓
Metal (iOS) / Vulkan (Android) / CUDA → GPU
```

## Status

**Version 0.1.0 — Early Alpha**

| Feature | Status |
|---------|--------|
| Nx.Backend behaviour (basic ops) | ✅ Implemented |
| Nx.Backend behaviour (shape ops) | ✅ Implemented |
| Nx.Backend behaviour (reductions) | ✅ Implemented |
| Nx.Backend behaviour (linear algebra) | ✅ Implemented |
| Rust NIF bridge (Burn CubeCL) | ✅ Implemented |
| GPU acceleration (Metal/Vulkan) | ✅ Via Burn/CubeCL |
| Axon model compilation | 🔄 Basic support |
| Training loop (SGD/Adam/RMSprop) | 🔄 Basic support |
| Nx.Serving | ✅ Implemented |
| Nx.Defn.Compiler | 🚧 Planned |
| CUDA backend | 🚧 Planned |
| Precompiled NIF binaries | 🚧 Planned |

> ⚠️ **Note**: The Quick Start examples show the target API. Some features
> (training loop, mobile deployment) are partially implemented and may not
> work end-to-end yet. See the [guides](guides/) for what's currently working.

## Features

- **Nx Backend**: Full `Nx.Backend` behaviour implementation — drop-in replacement for `Nx.BinaryBackend`
- **GPU Acceleration**: Burn's CubeCL backend with Metal (Apple), Vulkan (Android), CUDA (NVIDIA)
- **ExCubecl Integration**: GPU buffer management, kernel execution, async commands, and pipeline orchestration via [ExCubecl](https://hex.pm/packages/ex_cubecl)
- **Autodiff**: Automatic differentiation via Burn's `Autodiff` backend decorator
- **Training Loop**: Complete training with Adam, SGD, RMSprop optimizers, LR scheduling, gradient clipping, callbacks
- **Model Management**: Save/load, serialize, quantize (f16), benchmark
- **Structured Errors**: `ExBurn.Error` exception type with operation context

## Quick Start

> **Note**: This section shows the target API. Some features may not work
> end-to-end yet — see the [Status](#status) section above.

```elixir
# Set ExBurn as the default Nx backend
Nx.default_backend(ExBurn.Backend)

# Create and manipulate tensors
t = Nx.tensor([1.0, 2.0, 3.0])
Nx.add(t, t) |> Nx.to_list()

# Define a model with Axon
model =
  Axon.input("input", shape: {nil, 784})
  |> Axon.dense(256, activation: :relu)
  |> Axon.dropout(rate: 0.2)
  |> Axon.dense(10)

# Compile for training
compiled = ExBurn.Model.compile(model,
  loss: :cross_entropy,
  optimizer: :adam,
  learning_rate: 0.001
)

# Train
ExBurn.Training.fit(compiled, {train_x, train_y},
  epochs: 10,
  batch_size: 32,
  validation_data: {val_x, val_y},
  callbacks: [&ExBurn.Training.LoggingCallback.log/1]
)
```

## Prerequisites

- **Elixir** ~> 1.18 and **OTP** 27+
- **Rust** stable toolchain (required for NIF compilation)
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  ```
- **For iOS development**: Xcode + `aarch64-apple-ios` target
  ```bash
  rustup target add aarch64-apple-ios
  ```
- **For Android development**: Android NDK + `aarch64-linux-android` target
  ```bash
  rustup target add aarch64-linux-android
  ```

> **Note**: Precompiled NIF binaries are planned for v0.2.0. Until then, a Rust
> toolchain is required to build the NIF from source.

## Installation

Add `ex_burn` to your `mix.exs`:

```elixir
def deps do
  [
    {:ex_burn, "~> 0.1"},
    {:nx, ">= 0.7.0"},
    {:axon, "~> 0.7"}
  ]
end
```

## Training on Mobile — Caveats

Burn's Autodiff backend is memory-intensive. On iOS/Android with limited RAM,
training even small models may cause out-of-memory errors. Realistic expectations:

- **Fine-tuning** small models (< 10M parameters) is feasible on modern devices
- **Full training** of large models is not recommended on mobile
- **Inference** is the primary use case for mobile deployment
- Minimum recommended: 4GB RAM, A12+ chip (iOS) / Snapdragon 700+ (Android)

The training loop in ExBurn currently uses numerical gradients. Burn's autodiff
integration is planned for v0.3.0.

## Examples

```bash
# Linear regression (simplest possible ML workflow)
mix run examples/linear_regression.exs

# MNIST-like classifier (full deep learning pipeline)
mix run examples/mnist_simple.exs

# GPU-accelerated inference with ExCubecl
mix run examples/gpu_inference.exs
```

## Project Structure

```
lib/ex_burn/
  ex_burn.ex          — Main API (version, configure!, default_device)
  backend.ex          — Nx.Backend implementation (delegates to Burn via NIF)
  nif.ex              — Rustler NIF stubs (40+ functions)
  tensor.ex           — Nx ↔ Burn tensor conversion utilities
  error.ex            — Structured error type (ExBurn.Error)
  burn_bridge.ex      — High-level Burn API (direct tensor ops)
  cubecl_bridge.ex    — GPU compute via ExCubecl (buffers, kernels, pipelines)
  model.ex            — Model definition, compilation, save/load
  training.ex         — Training loop (optimizers, LR schedules, callbacks)

native/ex_burn_nif/
  src/lib.rs          — Rust NIF with real Burn Autodiff<CubeCL> operations
  Cargo.toml          — Burn 0.21 + CubeCL + Autodiff dependencies

examples/
  linear_regression.exs  — Simplest ML workflow
  mnist_simple.exs        — Full deep learning pipeline

guides/
  01_getting_started.md   — Installation, basic ops, GPU check
  02_training.md          — Models, training, callbacks, save/load
  03_mobile_deployment.md — iOS/Android compilation, optimization
  04_architecture.md      — Deep-dive into the pipeline
```

## GPU Backends

| Platform | Backend | Status |
|----------|---------|--------|
| iOS      | Metal   | ✅     |
| Android  | Vulkan  | ✅     |
| macOS    | Metal   | ✅     |
| Linux    | Vulkan  | ✅     |
| NVIDIA   | CUDA    | 🔜     |

## Error Handling

All operations raise `ExBurn.Error` with structured context:

```elixir
raise ExBurn.Error,
  op: :matmul,
  reason: "shape mismatch",
  details: %{lhs: [3, 4], rhs: [5, 6]}
```

## Dependencies

- [Burn](https://github.com/tracel-ai/burn) — Deep learning framework (Rust)
- [Nx](https://github.com/elixir-nx/nx) — Numerical Elixir
- [Axon](https://github.com/elixir-nx/axon) — Neural network library
- [CubeCL](https://github.com/tracel-ai/cubecl) — GPU compute language
- [ExCubecl](https://hex.pm/packages/ex_cubecl) v0.4+ — GPU compute runtime for Elixir (buffers, kernels, pipelines, media)

---

**Topics**: `elixir` · `machine-learning` · `burn` · `ios` · `android` · `nx` · `rustler` · `gpu` · `deep-learning`

## License

Apache 2.0
