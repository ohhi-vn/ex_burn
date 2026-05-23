# ExBurn

**ExBurn** is a middle layer between [Nx](https://github.com/elixir-nx/nx) and [Burn](https://github.com/tracel-ai/burn) that enables training ML/DeepLearning models on mobile devices via the [Dala](https://github.com/ohhi-vn/dala) framework.

## Architecture

```
Axon model
   ↓
Nx.Defn graph
   ↓
ExBurn.Backend (Nx.Backend behaviour)
   ↓
ExBurn.Nif (Rustler NIF)
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
| Mobile deployment (Dala) | 🚧 Planned |
| Nx.Defn.Compiler | 🚧 Planned |
| CUDA backend | 🚧 Planned |
| Precompiled NIF binaries | 🚧 Planned |

> ⚠️ **Note**: The Quick Start examples show the target API. Some features
> (training loop, mobile deployment) are partially implemented and may not
> work end-to-end yet. See the [guides](guides/) for what's currently working.

## Features

- **Nx Backend**: Full `Nx.Backend` behaviour implementation — drop-in replacement for `Nx.BinaryBackend`
- **GPU Acceleration**: Burn's CubeCL backend with Metal (Apple), Vulkan (Android), CUDA (NVIDIA)
- **Autodiff**: Automatic differentiation via Burn's `Autodiff` backend decorator
- **Mobile Deployment**: Compile models for iOS/Android via `ExBurn.DalaML`
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

## Mobile Deployment (Dala)

> **Note**: `ExBurn.DalaML` is tightly coupled to the Dala framework and is
> currently aspirational. The core Nx backend works independently.

```elixir
# Compile for iOS (Metal GPU)
{:ok, model} = ExBurn.DalaML.compile(axon_model,
  input_shape: {1, 784},
  target: :ios,
  precision: :f16
)

# Run inference
{:ok, output} = ExBurn.DalaML.predict(model, input_tensor)

# Export for deployment
ExBurn.DalaML.export(model, "model_ios.bin")

# Benchmark
{:ok, stats} = ExBurn.DalaML.benchmark(model, iterations: 100)
IO.puts("Avg: #{stats.avg_milliseconds}ms, P95: #{stats.p95_microseconds}μs")
```

## Examples

```bash
# Linear regression (simplest possible ML workflow)
mix run examples/linear_regression.exs

# MNIST-like classifier (full deep learning pipeline)
mix run examples/mnist_simple.exs

# Mobile deployment (iOS + Android compilation and benchmarking)
mix run examples/mobile_inference.exs
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
  cubecl_bridge.ex    — GPU context management (Metal/Vulkan)
  dala_ml.ex          — Mobile deployment compiler (iOS/Android)
  model.ex            — Model definition, compilation, save/load
  training.ex         — Training loop (optimizers, LR schedules, callbacks)

native/ex_burn_nif/
  src/lib.rs          — Rust NIF with real Burn Autodiff<CubeCL> operations
  Cargo.toml          — Burn 0.21 + CubeCL + Autodiff dependencies

examples/
  linear_regression.exs  — Simplest ML workflow
  mnist_simple.exs        — Full deep learning pipeline
  mobile_inference.exs    — iOS/Android deployment

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
- [ExCubecl](https://github.com/ohhi-vn/ex_cubecl) — GPU runtime for Elixir

## License

Apache 2.0
