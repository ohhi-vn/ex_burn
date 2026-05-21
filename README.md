# ExBurn

**ExBurn** is a middle layer between [Nx](https://github.com/elixir-nx/nx) and [Burn](https://github.com/tracel-ai/burn) that enables training ML/DeepLearning models on mobile devices via the [Dala](https://github.com/ohhi-vn/dala) framework.

## Architecture

```
Axon model
   ↓
Nx.Defn graph
   ↓
Dala ML Compiler
   ↓
optimized Burn graph
   ↓
CubeCL kernels
   ↓
Metal/Vulkan GPU execution
```

## Overview

ExBurn provides:

- **`ExBurn.Backend`** — An Nx backend that delegates tensor operations to Burn via Rust NIFs
- **`ExBurn.Nif`** — Rustler NIF stubs for Burn interop
- **`ExBurn.BurnBridge`** — High-level bridge for direct Burn operations
- **`ExBurn.DalaML`** — Dala ML compiler integration for mobile deployment
- **`ExBurn.CubeclBridge`** — Bridge to ExCubecl for GPU execution
- **`ExBurn.Model`** — Model definition and training orchestration
- **`ExBurn.Training`** — Training loop with GPU-accelerated gradient computation

## Quick Start

```elixir
# Set ExBurn as the default Nx backend
Nx.default_backend(ExBurn.Backend)

# Create and manipulate tensors
t = Nx.tensor([1.0, 2.0, 3.0])
Nx.add(t, t) |> Nx.to_list()

# Define a model with Axon
model =
  Axon.input("input", shape: {nil, 784})
  |> Axon.dense(256)
  |> Axon.relu()
  |> Axon.dense(10)

# Compile for training
compiled = ExBurn.Model.compile(model, loss: :cross_entropy, optimizer: :adam)

# Train
ExBurn.Training.fit(compiled, {train_x, train_y}, epochs: 10, batch_size: 32)
```

## Mobile Deployment (Dala)

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
```

## GPU Backends

| Platform | Backend | Status |
|----------|---------|--------|
| iOS      | Metal   | ✅     |
| Android  | Vulkan  | ✅     |
| macOS    | Metal   | ✅     |
| Linux    | Vulkan  | ✅     |
| NVIDIA   | CUDA    | 🔜     |

## Dependencies

- [Burn](https://github.com/tracel-ai/burn) — Deep learning framework (Rust)
- [Nx](https://github.com/elixir-nx/nx) — Numerical Elixir
- [Axon](https://github.com/elixir-nx/axon) — Neural network library
- [CubeCL](https://github.com/tracel-ai/cubecl) — GPU compute language
- [ExCubecl](https://github.com/ohhi-vn/ex_cubecl) — GPU runtime for Elixir

## License

Apache 2.0
