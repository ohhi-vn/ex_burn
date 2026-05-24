# Architecture Deep-Dive

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Elixir / BEAM VM                        │
│                                                             │
│  Axon model ──→ Nx.Defn graph ──→ ExBurn.Backend            │
│                                         │                   │
│                                         ↓                   │
│                              ExBurn.Nif (Rustler)           │
│                                         │                   │
│                                         ↕                   │
│                              ExCubecl (GPU runtime)         │
│                              - Buffer management            │
│                              - Kernel execution             │
│                              - Pipeline orchestration       │
│                              - Async commands               │
│                              - Media I/O                    │
└─────────────────────────────┬───────────────────────────────┘
                              │ NIF calls
┌─────────────────────────────↓───────────────────────────────┐
│                     Rust NIF Layer                          │
│                                                             │
│  BurnTensor enum ──→ Burn operations ──→ CubeCL runtime     │
│                                                             │
│  Backend: Autodiff<CubeCL>                                  │
│    - Autodiff: gradient tracking                            │
│    - CubeCL: GPU compute abstraction                        │
└─────────────────────────────┬───────────────────────────────┘
                              │ kernel dispatch
┌─────────────────────────────↓───────────────────────────────┐
│                     GPU Hardware                            │
│                                                             │
│  Metal (iOS/macOS)  │  Vulkan (Android/Linux)  │  CUDA     │
└─────────────────────────────────────────────────────────────┘
```

## Nx Backend Protocol

`ExBurn.Backend` implements the `Nx.Backend` behaviour. Every Nx operation
is translated to a NIF call:

```elixir
# Elixir side
Nx.add(a, b)
  ↓
ExBurn.Backend.add(%BurnTensor{ref: ref_a}, %BurnTensor{ref: ref_b})
  ↓
ExBurn.Nif.add_tensor(ref_a, ref_b)  # NIF call
  ↓
{:ok, ref_c}  # New tensor reference
```

## Tensor Representation

### Elixir Side

```elixir
%ExBurn.Tensor{
  ref: #Reference<...>,    # Opaque NIF reference
  shape: [3, 256],         # Shape tracked on Elixir side
  type: :f32               # Element type tag
}
```

### Rust Side

```rust
enum BurnTensor {
    F32x1(Tensor<B, 1>),   # 1D f32 tensor
    F32x2(Tensor<B, 2>),   # 2D f32 tensor
    F32x3(Tensor<B, 3>),   # 3D f32 tensor
    F32x4(Tensor<B, 4>),   # 4D f32 tensor (images)
    I32x1(Tensor<B, 1, Int>),
    I64x1(Tensor<B, 1, Int>),
}
```

## Memory Management

- Tensors are owned by `ResourceArc<TensorResource>` on the Rust side
- Erlang GC triggers NIF resource destructor → Burn tensor freed
- Explicit `ExBurn.Tensor.free/1` for eager deallocation

## Gradient Computation

```
Forward pass                Backward pass
─────────────               ─────────────
input → Linear → ReLU → output
              ↓
         loss = cross_entropy(output, target)
              ↓
         backward(loss)  ← Autodiff<CubeCL> computes ∂L/∂W
              ↓
         optimizer.step()  ← Adam/SGD updates W -= lr * ∂L/∂W
```

## ExCubecl Integration

ExBurn uses [ExCubecl](https://hex.pm/packages/ex_cubecl) v0.4+ as its GPU compute runtime. ExCubecl provides:

- **GPU Buffers**: `ExCubecl.buffer/3` creates GPU-resident buffers with automatic GC
- **Kernel Execution**: `ExCubecl.run_kernel/4` dispatches CubeCL kernels
- **Pipelines**: `ExCubecl.pipeline/0` + `pipeline_add/5` + `pipeline_run/1` for multi-kernel orchestration
- **Async Commands**: `ExCubecl.submit/1` + `poll/1` + `wait/1` for non-blocking execution
- **Media I/O**: `ExCubecl.Media`, `ExCubecl.Video`, `ExCubecl.Audio`, `ExCubecl.Filter`, `ExCubecl.Transcode`

`ExBurn.CubeclBridge` wraps ExCubecl with a higher-level API, and `ExBurn.BurnBridge` provides ExCubecl buffer helpers.

## Performance Considerations

1. **Minimize NIF round-trips**: Use `BurnBridge` for multi-op sequences
2. **Batch conversions**: `ExBurn.Tensor.from_nx_batch/1` for multiple tensors
3. **Shape caching**: Shapes tracked on Elixir side, no NIF call needed
4. **f16 on mobile**: Use `precision: :f16` for 2x memory reduction
5. **Use ExCubecl pipelines**: Chain multiple GPU kernels without CPU round-trips

## Error Handling

All NIF functions return `{:ok, result}` or `{:error, reason}`.
The Elixir layer wraps these in `ExBurn.Error` exceptions:

```elixir
raise ExBurn.Error,
  op: :matmul,
  reason: "shape mismatch",
  details: %{lhs: [3, 4], rhs: [5, 6]}
```

## Thread Safety

- NIF calls are scheduled on dirty CPU schedulers for long operations
- Burn's CubeCL runtime handles GPU command queue synchronization
- `ExBurn.Nif.gpu_available/0` is safe to call from any process
