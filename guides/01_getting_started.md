# Getting Started with ExBurn

## Installation

Add `ex_burn` to your `mix.exs`:

```elixir
def deps do
  [
    {:ex_burn, "~> 0.1"},
    {:nx, ">= 0.7.0"},
    {:axon, "~> 0.7"},
    {:ex_cubecl, ">= 0.4.0"}
  ]
end
```

Then run:

```bash
mix deps.get
mix compile
```

> **Note**: ExBurn is not yet on Hex.pm. Install from GitHub until the first stable release.

## Basic Tensor Operations

```elixir
# Set ExBurn as the default Nx backend
Nx.default_backend(ExBurn.Backend)

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

## Using the BurnBridge Directly

For performance-critical code, bypass the Nx layer:

```elixir
# Create Burn tensors directly
t1 = ExBurn.BurnBridge.zeros([3, 3], :f32)
t2 = ExBurn.BurnBridge.ones([3, 3], :f32)

# Perform operations
t3 = ExBurn.BurnBridge.add(t1, t2)

# Convert to Nx when needed
nx_tensor = ExBurn.BurnBridge.to_nx(t3)
```

## Checking GPU Availability

ExBurn uses [ExCubecl](https://hex.pm/packages/ex_cubecl) for GPU buffer management and kernel execution:

```elixir
# Check if ExCubecl (GPU runtime) is available
if ExCubecl.available?() do
  {:ok, info} = ExCubecl.device_info()
  IO.puts("GPU: #{info.device_name}")
else
  IO.puts("Running on CPU (ExCubecl not available)")
end
```

You can also use the ExBurn helper:

```elixir
if ExBurn.NifHelper.gpu_available() do
  IO.puts("GPU: #{ExBurn.NifHelper.device_name()}")
else
  IO.puts("Running on CPU")
end
```

## Using ExCubecl Buffers Directly

For GPU-native workflows, create buffers directly via ExCubecl:

```elixir
# Create GPU buffers
{:ok, a} = ExCubecl.buffer([1.0, 2.0, 3.0], [3], :f32)
{:ok, b} = ExCubecl.buffer([4.0, 5.0, 6.0], [3], :f32)

# Inspect
{:ok, [3]} = ExCubecl.shape(a)
{:ok, "f32"} = ExCubecl.dtype(a)
{:ok, 12} = ExCubecl.size(a)  # bytes

# Read data back
{:ok, data} = ExCubecl.read(a)

# Run a kernel
output = ExCubecl.buffer!([0.0, 0.0, 0.0], [3], :f32)
ExCubecl.run_kernel("elementwise_add", [a, b], output)

# Buffers are automatically freed when GC'd — no manual free needed
```

## Using the CubeclBridge

`ExBurn.CubeclBridge` provides a higher-level wrapper around ExCubecl with pipeline support:

```elixir
# Check available backends
ExBurn.CubeclBridge.available_backends()

# Create a pipeline
{:ok, pipeline} = ExBurn.CubeclBridge.pipeline()
ExBurn.CubeclBridge.pipeline_add(pipeline, "elementwise_add", [a, b], output)
ExBurn.CubeclBridge.pipeline_add(pipeline, "relu", [output], output)
{:ok, _cmd_ids} = ExBurn.CubeclBridge.pipeline_run(pipeline)
:ok = ExBurn.CubeclBridge.pipeline_free(pipeline)
```

## Next Steps

- [Training Models](02_training.md)
- [Mobile Deployment](03_mobile_deployment.md)
- [Architecture Deep-Dive](04_architecture.md)
