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
ExBurn.Defn.Compiler (Nx.Defn.Compiler behaviour)
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

**Version 0.6.0 — Early Alpha**

> ⚠️ **Note**: This library is in early development. The API may change
> between minor versions. Not yet recommended for production use.

| Feature | Status |
|---------|--------|
| Nx.Backend behaviour (basic ops) | ✅ Implemented |
| Nx.Backend behaviour (shape ops) | ✅ Implemented |
| Nx.Backend behaviour (reductions) | ✅ Implemented |
| Nx.Backend behaviour (linear algebra) | ✅ Implemented |
| Nx.Defn.Compiler | ✅ Implemented |
| Rust NIF bridge (Burn CubeCL) | ✅ Implemented |
| GPU acceleration (Metal/Vulkan) | ✅ Via Burn/CubeCL |
| Axon model compilation | ✅ Implemented |
| Training loop (SGD/Adam/RMSprop) | ✅ Implemented |
| GPU forward pass (defn compiler) | ✅ Implemented |
| Glorot/Xavier initialization | ✅ Implemented |
| Layer freeze/unfreeze | ✅ Implemented |
| Gradient accumulation | ✅ Implemented |
| Nesterov momentum | ✅ Implemented |
| Weight decay (L2) | ✅ Implemented |
| Model summary | ✅ Implemented |
| Device management (CPU↔GPU) | ✅ Implemented |
| Nx.Serving | ✅ Implemented |
| CUDA backend | ✅ Implemented |
| Precompiled NIF binaries | 🚧 Planned |
| Autodiff gradients | 🚧 Planned |

### Known limitations

- **Dtype support**: The NIF stores tensors as `f32` only. Other dtypes are
  value-converted to `f32` at the backend boundary (the direct tensor API
  raises instead). Full dtype preservation is planned for a future release.
- **Gradient computation**: The default gradient method is `:numerical`
  (finite differences). Autodiff gradients are planned but not yet
  implemented.
- **Global backend mutation**: By default, `Training.fit/3` no longer
  changes the global Nx backend. Pass `set_default_backend?: true` to
  restore the previous behavior.
- **Evaluation loss**: Loss is now correctly weighted by sample count
  when batches have different sizes.

#### Direct NIF operation constraints

When calling `ExBurn.BurnBridge` / `ExBurn.Nif` **directly** (bypassing the
Nx backend, which normalizes these cases), the Rust layer currently imposes:

| Operation | Constraint |
|---|---|
| `matmul_tensor/2` | Both operands rank ≥ 2 |
| `pow_tensor/2` | Exponent must be an f32 scalar, not a tensor |
| `transpose_tensor/1` | 2-D tensors only (use the Nx backend for other ranks) |
| `broadcast_tensor/2` | Expands existing dimensions only (e.g. `[1,2] → [4,2]`) |
| `iota_tensor/3` | Produces a 1-D iota of length `shape[axis]` |
| `reshape_tensor/2` | Element count of the new shape must match |

The `Nx.Backend` layer (`ExBurn.Backend`) lifts several of these by computing
general cases exactly via Nx — prefer going through Nx unless you need the
raw single-NIF-call path.

## Features

- **Nx Backend**: `Nx.Backend` behaviour implementation — partial drop-in
  replacement for `Nx.BinaryBackend` with some dtype limitations
- **Nx Defn Compiler**: Custom `Nx.Defn.Compiler` that executes defn expressions on the Burn GPU backend
- **GPU Acceleration**: Burn's CubeCL backend with CUDA (NVIDIA), Metal (Apple), Vulkan (Android)
- **ExCubecl Integration**: GPU buffer management, kernel execution, async commands, and pipeline orchestration via [ExCubecl](https://hex.pm/packages/ex_cubecl)
- **Autodiff**: Planned automatic differentiation via Burn's `Autodiff`
  backend decorator; currently uses numerical gradients
- **Training Loop**: Complete training with Adam, SGD, RMSprop optimizers, LR scheduling, gradient clipping, callbacks
- **Model Management**: Save/load, serialize, quantize (f16), benchmark
- **Structured Errors**: `ExBurn.Error` exception type with operation context

## Quick Start

### 1. Install

Add `ex_burn` to your `mix.exs`:

```elixir
def deps do
  [
    {:ex_burn, "~> 0.5"},
    {:nx, ">= 0.12.0 and < 2.0.0"},
    {:axon, "~> 0.8"}
  ]
end
```

```bash
mix deps.get
mix compile   # first build compiles the Rust NIF — expect a few minutes
```

> The NIF builds CPU-only by default. For GPU acceleration see
> [GPU Backends](#gpu-backends) below.

### 2. Verify the installation

```elixir
# In iex -S mix:
ExBurn.smoke_test()
#=> :ok                       ← Nx → Backend → NIF → Burn pipeline works

ExBurn.summary()
#=>
# ExBurn v0.5.0
# ──────────────────────────────
# Device: Metal (Apple M2 Pro)  ← or "NdArray (CPU)" without GPU features
# GPU: available
# Backends: metal
```

### 3. Tensor operations through Burn

```elixir
# Set ExBurn as the default Nx backend
Nx.default_backend(ExBurn.Backend)
# …or equivalently: ExBurn.configure!()

t = Nx.tensor([1.0, 2.0, 3.0])
Nx.add(t, t) |> Nx.to_list()
#=> [2.0, 4.0, 6.0]
```

### 4. GPU-accelerated functions with `defn`

```elixir
# Set ExBurn as both backend and compiler
Nx.default_backend(ExBurn.Backend)
Nx.Defn.global_default_options(compiler: ExBurn.Defn.Compiler)

defmodule MyMath do
  import Nx.Defn

  defn add_and_scale(x, y, scale) do
    x
    |> Nx.add(y)
    |> Nx.multiply(scale)
  end
end

result = MyMath.add_and_scale(Nx.tensor([1.0, 2.0]), Nx.tensor([3.0, 4.0]), Nx.tensor(2.0))
Nx.to_list(result)
#=> [8.0, 12.0]
```

To use the compiler for a single call only (no global setting):

```elixir
Nx.Defn.jit_apply(&MyMath.add_and_scale/3, [x, y, scale],
  compiler: ExBurn.Defn.Compiler
)
```

### 5. Train a model with Axon

```elixir
model =
  Axon.input("input", shape: {nil, 784})
  |> Axon.dense(256, activation: :relu)
  |> Axon.dropout(rate: 0.2)
  |> Axon.dense(10)

compiled =
  ExBurn.Model.compile(model,
    loss: :cross_entropy,
    optimizer: :adam,
    learning_rate: 0.001
  )

trained =
  ExBurn.Training.fit(compiled, {train_x, train_y},
    epochs: 10,
    batch_size: 32,
    validation_data: {val_x, val_y},
    callbacks: [&ExBurn.Training.LoggingCallback.log/1]
  )

{:ok, output} = ExBurn.Model.forward(trained, input_tensor)
```

### 6. Batched inference with Nx.Serving

```elixir
serving =
  ExBurn.Serving.new(trained, batch_size: 8, batch_timeout: 10)

%Nx.Serving{} = result = Nx.Serving.run(serving, Nx.Batch.stack([input1, input2]))
output = result.output
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

> **Note**: Precompiled NIF binaries are planned. Until then, a Rust
> toolchain is required to build the NIF from source.

## Training on Mobile — Caveats

Burn's Autodiff backend is memory-intensive. On iOS/Android with limited RAM,
training even small models may cause out-of-memory errors. Realistic expectations:

- **Fine-tuning** small models (< 10M parameters) is feasible on modern devices
- **Full training** of large models is not recommended on mobile
- **Inference** is the primary use case for mobile deployment
- Minimum recommended: 4GB RAM, A12+ chip (iOS) / Snapdragon 700+ (Android)

The training loop in ExBurn currently uses numerical gradients (finite differences).
Two methods are available: `:numerical` (central differences, more accurate) and
`:numerical_batch` (one-sided, ~2x faster). Burn's autodiff integration is
planned for v0.3.0 and will replace numerical gradients entirely.

## Examples

```bash
# Linear regression (simplest possible ML workflow)
mix run examples/linear_regression.exs

# MNIST-like classifier (full deep learning pipeline)
mix run examples/mnist_simple.exs

# XOR classifier (non-linear problem, early stopping, model summary)
mix run examples/xor_classifier.exs

# Dataset utilities (split, normalize, one-hot, data loaders)
mix run examples/dataset_utils.exs

# BurnBridge direct tensor operations (arithmetic, math, linear algebra)
mix run examples/burn_bridge_ops.exs

# Model management (save/load, quantize, freeze, benchmark, export)
mix run examples/model_management.exs

# Training callbacks (logging, early stopping, checkpoint, LR scheduling)
mix run examples/training_callbacks.exs
```

## Benchmarks

Benchmark scripts in `bench/` compare ExBurn (Burn GPU backend) against plain Nx (BinaryBackend) across tensor sizes from 10×10 to 2000×2000.

```bash
# Tensor creation (zeros, ones, rand)
mix run bench/tensor_creation_bench.exs

# Element-wise arithmetic (add, mul, exp)
mix run bench/arithmetic_bench.exs

# Linear algebra (matmul, transpose)
mix run bench/linear_algebra_bench.exs

# Nx <-> Burn tensor conversion overhead
mix run bench/conversion_bench.exs

# End-to-end training (small/medium MLPs, optimizer comparison)
mix run bench/training_bench.exs

# Inference latency and throughput (single + batched + Nx.Serving)
mix run bench/serving_bench.exs
```

Each script prints a formatted table with timing results. All benchmarks include warmup runs and report averaged measurements.

## Testing

```bash
mix test                          # full suite (~600 tests, <30s)
mix test test/backend_test.exs    # single file
mix test --only nif               # only NIF-backed tests
mix test --cover                  # suite + coverage report (threshold enforced)
```

Test tags (`:nif`, `:cuda`, `:metal`, `:vulkan`) are excluded automatically
when the NIF or matching GPU hardware isn't available — see
`test/test_helper.exs`.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `:erlang.nif_error(:nif_not_loaded)` | Native library wasn't compiled/linked | `mix clean && mix compile`; check Rust is installed |
| `dtype :f64 is not supported by the NIF` | Non-f32 dtype crossed the raw NIF boundary | Convert via `Nx.as_type(t, {:f, 32})` or go through `Nx` with `ExBurn.Backend` |
| `{:error, "Erlang error: :nif_panicked"}` | Rust-side assertion (e.g. rank/dtype mismatch on a direct NIF call) | Re-run with `RUST_BACKTRACE=1`; check the [NIF constraints](#known-limitations) table |
| `gpu_available: false` | NIF built CPU-only, or drivers missing | Rebuild with `./build.sh metal` / `cuda` / `vulkan` |
| First compile takes minutes | Debug-mode Rust build of Burn/CubeCL | Expected; release mode is used for production builds |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full development workflow:
environment setup, running tests and coverage, linting, and a step-by-step
walkthrough for adding new operations.

## Guides

- [Getting Started](guides/01_getting_started.md) — Installation, basic ops, GPU check
- [Training Models](guides/02_training.md) — Models, training, callbacks, save/load
- [Mobile Deployment](guides/03_mobile_deployment.md) — iOS/Android compilation, optimization
- [Architecture](guides/04_architecture.md) — Deep-dive into the pipeline
- [Training Optimization](guides/05_training_optimization.md) — Best practices for fast, stable training
- [Deep Learning Guide](guides/06_deep_learning_guide.md) — Step-by-step lessons for learning DL with ExBurn
- [Benchmarks](guides/07_benchmarks.md) — Performance benchmarks and how to run them
- [Contributing](CONTRIBUTING.md) — Development workflow, testing, adding operations

## Project Structure

```
lib/ex_burn/
  ex_burn.ex          — Main API (version, configure!, default_device)
  defn_compiler.ex    — Nx.Defn.Compiler for GPU-accelerated defn
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
  xor_classifier.exs      — Non-linear classification with early stopping
  dataset_utils.exs       — Data preprocessing utilities
  burn_bridge_ops.exs     — Direct Burn tensor operations
  model_management.exs    — Save/load, quantize, freeze, benchmark
  training_callbacks.exs  — All callback types + custom callbacks

bench/
  tensor_creation_bench.exs   — zeros/ones/rand: Nx vs Burn
  arithmetic_bench.exs       — add/mul/exp: Nx vs Burn
  linear_algebra_bench.exs   — matmul/transpose: Nx vs Burn
  conversion_bench.exs       — Nx<->Burn conversion overhead
  training_bench.exs         — End-to-end training performance
  serving_bench.exs          — Inference latency & throughput

guides/
  01_getting_started.md   — Installation, basic ops, GPU check
  02_training.md          — Models, training, callbacks, save/load
  03_mobile_deployment.md — iOS/Android compilation, optimization
  04_architecture.md      — Deep-dive into the pipeline
```

## GPU Backends

| Platform | Backend | Status |
|----------|---------|--------|
| NVIDIA   | CUDA    | ✅     |
| iOS      | Metal   | ✅     |
| Android  | Vulkan  | ✅     |
| macOS    | Metal   | ✅     |
| Linux    | Vulkan  | ✅     |

### CUDA Support

The NIF compiles **without a GPU backend by default** (CPU-only NdArray).
To build with GPU acceleration, use the bundled helper script, which
auto-detects the best backend for your platform:

```bash
./build.sh              # auto-detect: cuda / metal / vulkan / cpu
./build.sh metal        # force a specific backend
```

Or set the Cargo feature manually before compiling:

```bash
RUSTLER_NIF_CARGO_FEATURES=cuda mix compile
```

On systems without the requested GPU hardware, the NIF automatically falls
back to the NdArray (CPU) backend at runtime.

Check CUDA availability from Elixir:

```elixir
ExBurn.cuda_available?()   # true if NVIDIA GPU detected
ExBurn.device_name()       # "CUDA (NVIDIA GPU)" or "NdArray (CPU)"
ExBurn.device_info()       # full device info map
```

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
- [ExCubecl](https://hex.pm/packages/ex_cubecl) v0.5+ — GPU compute runtime for Elixir (buffers, kernels, pipelines, media)

---

**Topics**: `elixir` · `machine-learning` · `burn` · `ios` · `android` · `nx` · `rustler` · `gpu` · `deep-learning`

## License

Apache 2.0
