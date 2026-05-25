# Getting Started with ExBurn

## What is ExBurn?

ExBurn is a middle layer between [Nx](https://github.com/elixir-nx/nx) (Numerical Elixir) and [Burn](https://github.com/tracel-ai/burn) (a Rust deep learning framework). It lets you write ML code in Elixir that runs on the GPU — on NVIDIA cards (CUDA), Apple Silicon (Metal), or Android (Vulkan).

```
Elixir code → Nx.Defn → ExBurn → Burn/CubeCL → GPU
```

## Installation

Add to `mix.exs`:

```elixir
def deps do
  [
    {:ex_burn, "~> 0.2"},
    {:nx, ">= 0.12.0"},
    {:axon, "~> 0.8"},
    {:ex_cubecl, ">= 0.4.0"}
  ]
end
```

```bash
mix deps.get
mix compile
```

### Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Elixir | ~> 1.18 | |
| OTP | 27+ | |
| Rust stable | any | Needed for NIF compilation (until v0.2.0 precompiled binaries) |
| GPU drivers | — | CUDA / Metal / Vulkan depending on platform |

For iOS: `rustup target add aarch64-apple-ios`
For Android: `rustup target add aarch64-linux-android`

## Basic Tensor Operations

```elixir
# Set ExBurn as the default backend — all Nx ops now go through Burn
Nx.default_backend(ExBurn.Backend)

# Or use the convenience function
ExBurn.configure!()

# Create tensors
a = Nx.tensor([1.0, 2.0, 3.0])
b = Nx.tensor([4.0, 5.0, 6.0])

# Element-wise operations
Nx.add(a, b)        # [5.0, 7.0, 9.0]
Nx.multiply(a, b)   # [4.0, 10.0, 18.0]

# Matrix operations
m = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
Nx.transpose(m)     # [[1.0, 3.0], [2.0, 4.0]]
```

## GPU-Accelerated Functions with `defn`

The `ExBurn.Defn.Compiler` implements the `Nx.Defn.Compiler` behaviour, letting you write GPU-accelerated numerical functions:

```elixir
# Set ExBurn as both backend and compiler
Nx.default_backend(ExBurn.Backend)
Nx.Defn.global_default_options(compiler: ExBurn.Defn.Compiler)

defmodule MyMath do
  import Nx.Defn

  defn add_and_scale(x, y, scale) do
    x |> Nx.add(y) |> Nx.multiply(scale)
  end

  defn dot_product(a, b) do
    a |> Nx.multiply(b) |> Nx.sum()
  end
end

# These execute on GPU via Burn
MyMath.add_and_scale(Nx.tensor([1.0, 2.0]), Nx.tensor([3.0, 4.0]), Nx.tensor(2.0))
#=> #Nx.Tensor<[8.0, 12.0]>
```

Per-function compiler override:

```elixir
defn my_fun(x, opts \\ []) do
  Nx.sin(x)
end
compiler: ExBurn.Defn.Compiler
```

## Checking GPU Availability

```elixir
# Quick check
ExBurn.default_device()    # :gpu or :cpu
ExBurn.device_name()       # e.g. "CUDA (NVIDIA RTX 4090)" or "Metal (Apple M4)"
ExBurn.device_info()       # full map with :device, :gpu_available, :backend, :available_backends
ExBurn.cuda_available?()   # true if NVIDIA GPU detected
```

## Using BurnBridge Directly

For performance-critical code, bypass the Nx layer and talk to Burn directly:

```elixir
# Create Burn tensors directly
t1 = ExBurn.BurnBridge.zeros([3, 3], :f32)
t2 = ExBurn.BurnBridge.ones([3, 3], :f32)

# Perform operations (single NIF call each)
t3 = ExBurn.BurnBridge.add(t1, t2)
t4 = ExBurn.BurnBridge.matmul(t1, t2)
t5 = ExBurn.BurnBridge.relu(t3)

# Convert back to Nx when needed
nx_tensor = ExBurn.BurnBridge.to_nx(t3)
```

## Using ExCubecl for GPU Buffers

ExCubecl provides low-level GPU buffer management:

```elixir
# Create GPU-resident buffers
{:ok, a} = ExCubecl.buffer([1.0, 2.0, 3.0], [3], :f32)
{:ok, b} = ExCubecl.buffer([4.0, 5.0, 6.0], [3], :f32)

# Inspect
{:ok, [3]} = ExCubecl.shape(a)
{:ok, 12} = ExCubecl.size(a)  # bytes

# Read data back
{:ok, data} = ExCubecl.read(a)

# Buffers are automatically freed when GC'd
```

## Project Structure

```
lib/ex_burn/
  ex_burn.ex            — Main API (version, configure!, device_info)
  defn_compiler.ex      — Nx.Defn.Compiler for GPU-accelerated defn
  backend.ex            — Nx.Backend implementation (delegates to Burn via NIF)
  nif.ex                — Rustler NIF stubs (40+ functions)
  nif_helper.ex         — Safe NIF wrappers ({:ok, result} tuples)
  tensor.ex             — Nx ↔ Burn tensor conversion utilities
  error.ex              — Structured error type (ExBurn.Error)
  burn_bridge.ex        — High-level Burn API (direct tensor ops)
  cubecl_bridge.ex      — GPU compute via ExCubecl (buffers, kernels, pipelines)
  model.ex              — Model definition, compilation, save/load
  training.ex           — Training loop (optimizers, LR schedules, callbacks)
  serving.ex            — Nx.Serving integration for batched inference
  serving/server.ex     — Serving server implementation

native/ex_burn_nif/
  src/lib.rs            — Rust NIF with real Burn Autodiff<CubeCL> operations
  Cargo.toml            — Burn + CubeCL + Autodiff dependencies
```

## Next Steps

- [Training Models](02_training.md) — Define, compile, and train neural networks
- [Mobile Deployment](03_mobile_deployment.md) — iOS/Android compilation and optimization
- [Architecture Deep-Dive](04_architecture.md) — How the pipeline works internally
- [Training Optimization Guide](05_training_optimization.md) — Best practices for fast, stable training
