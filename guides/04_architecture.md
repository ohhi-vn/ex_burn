# Architecture Deep-Dive

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Elixir / BEAM VM                             │
│                                                                     │
│  Axon model ──→ Nx.Defn graph ──→ ExBurn.Defn.Compiler              │
│                                           │                         │
│                                           ↓                         │
│                                ExBurn.Backend                       │
│                                           │                         │
│                                           ↓                         │
│                                ExBurn.Nif (Rustler)                 │
│                                           │                         │
│                                           ↕                         │
│                                ExCubecl (GPU runtime)               │
│                                - Buffer management                  │
│                                - Kernel execution                   │
│                                - Pipeline orchestration             │
│                                - Async commands                     │
└─────────────────────────────┬───────────────────────────────────────┘
                              │ NIF calls
┌─────────────────────────────↓───────────────────────────────────────┐
│                        Rust NIF Layer                               │
│                                                                     │
│  BurnTensor enum ──→ Burn operations ──→ CubeCL runtime             │
│                                                                     │
│  Backend: Autodiff<CubeCL>                                          │
│    - Autodiff: gradient tracking                                    │
│    - CubeCL: GPU compute abstraction                                │
└─────────────────────────────┬───────────────────────────────────────┘
                              │ kernel dispatch
┌─────────────────────────────↓───────────────────────────────────────┐
│                        GPU Hardware                                 │
│                                                                     │
│  Metal (iOS/macOS)  │  Vulkan (Android/Linux)  │  CUDA (NVIDIA)   │
└─────────────────────────────────────────────────────────────────────┘
```

## Layer-by-Layer Breakdown

### 1. Axon Model Definition

Axon provides a functional API for defining neural network architectures. Models are built as a pipeline of layers:

```elixir
model =
  Axon.input("input", shape: {nil, 784})
  |> Axon.dense(256, activation: :relu)
  |> Axon.dense(10)
```

This creates an `Axon.ModelState` struct containing the layer graph. No computation happens at this stage — it's a description of the model.

### 2. Nx.Defn Graph

When you call a `defn` function, `Nx.Defn` traces the function body into an expression tree of `Nx.Defn.Expr` nodes. Each node represents an operation (add, multiply, dot, etc.) with its arguments.

```
Nx.Defn.Expr
  op: :dot
  args: [
    Nx.Defn.Expr{op: :parameter, args: [0]},     # input
    Nx.Defn.Expr{op: :tensor, args: [weight]}     # weight matrix
  ]
```

### 3. ExBurn.Defn.Compiler

`ExBurn.Defn.Compiler` implements the `Nx.Defn.Compiler` behaviour. It receives the expression tree and evaluates each node:

1. **Parameters** are looked up from the params list and converted to Burn tensors
2. **Tensor constants** are converted to Burn tensors
3. **Operations** are dispatched to `ExBurn.Backend`, which calls the NIF
4. **Results** are cached by expression ID to avoid recomputation
5. **Control flow** (`:cond`, `:while`) is handled recursively

```elixir
# Global default
Nx.Defn.global_default_options(compiler: ExBurn.Defn.Compiler)

# Per-function
defn my_fun(x) do
  Nx.sin(x)
end
compiler: ExBurn.Defn.Compiler
```

### 4. ExBurn.Backend

`ExBurn.Backend` implements the `Nx.Backend` behaviour. Every Nx operation is translated to a NIF call:

```elixir
# Elixir side
Nx.add(a, b)
  ↓
ExBurn.Backend.add(%BurnTensor{ref: ref_a}, %BurnTensor{ref: ref_b})
  ↓
ExBurn.Nif.add_tensor(ref_a, ref_b)  # NIF call to Rust
  ↓
{:ok, ref_c}  # New tensor reference
```

The backend implements 100+ operation callbacks including:
- **Arithmetic**: add, subtract, multiply, divide, negate, abs, exp, log, sqrt, pow
- **Trig**: sin, cos, tan, asin, acos, atan, sinh, cosh, tanh
- **Reductions**: sum, product, reduce_max, reduce_min, argmax, argmin, all, any
- **Linear algebra**: dot, transpose, conv
- **Shape ops**: reshape, squeeze, broadcast, pad, slice, concatenate, stack, reverse, gather
- **Random**: random_uniform, random_normal
- **Creation**: eye, iota, from_binary
- **Comparison**: equal, not_equal, greater, less, greater_equal, less_equal
- **Logical**: logical_and, logical_or, logical_xor, bitwise_and, bitwise_or, bitwise_xor

### 5. ExBurn.Nif (Rustler NIF)

The NIF layer provides 40+ Rust functions that call into Burn. These are defined in `native/ex_burn_nif/src/lib.rs` using the `rustler` crate.

Key functions:
- `new_tensor/3` — create a tensor from binary data
- `add_tensor/2`, `sub_tensor/2`, `mul_tensor/2`, `div_tensor/2` — arithmetic
- `matmul_tensor/2` — matrix multiplication
- `sum_tensor/1`, `mean_tensor/1` — reductions
- `softmax_tensor/2`, `layer_norm_tensor/1` — neural network ops
- `gpu_available/0`, `device_name/0` — device queries
- `to_gpu/1`, `to_cpu/1` — device transfer
- `free_tensor/1` — explicit deallocation

### 6. ExCubecl Integration

ExBurn uses [ExCubecl](https://hex.pm/packages/ex_cubecl) v0.4+ as its GPU compute runtime:

- **GPU Buffers**: `ExCubecl.buffer/3` creates GPU-resident buffers with automatic GC
- **Kernel Execution**: `ExCubecl.run_kernel/4` dispatches CubeCL kernels
- **Pipelines**: Chain multiple GPU kernels without CPU round-trips
- **Async Commands**: Non-blocking GPU execution with `submit/poll/wait`

`ExBurn.CubeclBridge` wraps ExCubecl with a higher-level API.

## Tensor Representation

### Elixir Side

```elixir
%ExBurn.Tensor{
  ref: #Reference<...>,    # Opaque NIF reference to Rust tensor
  shape: [3, 256],         # Shape tracked on Elixir side (no NIF call needed)
  type: :f32               # Element type tag (:f32, :f16, :bf16, :f64, :i32, :i64, :i16, :i8, :u8)
}
```

### Rust Side

```rust
enum BurnTensor {
    F32x1(Tensor<B, 1>),   # 1D f32 tensor
    F32x2(Tensor<B, 2>),   # 2D f32 tensor
    F32x3(Tensor<B, 3>),   # 3D f32 tensor
    F32x4(Tensor<B, 4>),   # 4D f32 tensor (images: batch, channels, height, width)
}
```

## Memory Management

- Tensors are owned by `ResourceArc<TensorResource>` on the Rust side
- Erlang GC triggers NIF resource destructor → Burn tensor freed automatically
- Explicit `ExBurn.Tensor.free/1` for eager deallocation when needed
- GPU buffers via ExCubecl are automatically freed when GC'd

## Gradient Computation

### Current: Numerical Gradients

The training loop uses **finite differences** to approximate gradients:

```
∂L/∂w ≈ (L(w + ε) - L(w - ε)) / 2ε
```

This requires **2 forward passes per parameter**, making it slow for large models. Two methods are available:

| Method | Forward Passes | Accuracy | Speed |
|---|---|---|---|
| `:numerical` | 2N (central differences) | Higher (O(ε²)) | Slower |
| `:numerical_batch` | N+1 (one-sided) | Good (O(ε)) | ~2x faster |

Where N = number of scalar parameters.

### Planned: Burn Autodiff

```
Forward pass                    Backward pass
─────────────                   ─────────────
input → Linear → ReLU → output
              ↓
         loss = cross_entropy(output, target)
              ↓
         backward(loss)  ← Autodiff<CubeCL> computes ∂L/∂W
              ↓
         optimizer.step()  ← Adam/SGD updates W -= lr * ∂L/∂W
```

Burn's Autodiff backend will compute exact gradients in a single backward pass, replacing numerical differentiation entirely.

## Training Loop Architecture

```
fit(model, data, opts)
  │
  ├─ For each epoch:
  │    ├─ Apply LR schedule
  │    ├─ Shuffle data (if :shuffle)
  │    ├─ For each mini-batch:
  │    │    ├─ Forward pass → compute loss
  │    │    ├─ Backward pass → compute gradients
  │    │    ├─ Clip gradients (by norm / by value)
  │    │    ├─ Add weight decay to gradients
  │    │    └─ Optimizer step → update params
  │    ├─ Evaluate on validation data
  │    ├─ Print progress (loss, accuracy, ETA)
  │    └─ Run callbacks
  │
  └─ Return trained model
```

## Error Handling

All NIF functions return `{:ok, result}` or `{:error, reason}`. The Elixir layer wraps these in `ExBurn.Error` exceptions:

```elixir
raise ExBurn.Error,
  op: :matmul,
  reason: "shape mismatch",
  details: %{lhs: [3, 4], rhs: [5, 6]}
```

## Thread Safety

- NIF calls are scheduled on dirty CPU schedulers for long-running operations
- Burn's CubeCL runtime handles GPU command queue synchronization
- `ExBurn.Nif.gpu_available/0` is safe to call from any process
- The training loop is single-process; use `Nx.Serving` for concurrent inference

## Performance Considerations

1. **Minimize NIF round-trips**: Each NIF call has overhead. Use `BurnBridge` for multi-op sequences instead of individual Nx calls.
2. **Batch conversions**: Convert multiple tensors at once when possible.
3. **Shape caching**: Shapes are tracked on the Elixir side — no NIF call needed to check shape.
4. **f16 on mobile**: Use `ExBurn.Model.quantize(model, :f16)` for 2x memory reduction on mobile GPUs.
5. **Use ExCubecl pipelines**: Chain multiple GPU kernels without CPU round-trips.
6. **Gradient accumulation**: Use `:accumulate_gradients` to simulate larger batch sizes without increasing memory usage.
