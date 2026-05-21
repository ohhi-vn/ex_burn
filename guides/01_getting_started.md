# Getting Started with ExBurn

## Installation

Add `ex_burn` to your `mix.exs`:

```elixir
def deps do
  [
    {:ex_burn, "~> 0.1.0"},
    {:nx, "~> 0.7"},
    {:axon, "~> 0.7"}
  ]
end
```

Then run:

```bash
mix deps.get
mix compile
```

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

```elixir
if ExBurn.Nif.gpu_available() do
  IO.puts("GPU: #{ExBurn.Nif.device_name()}")
else
  IO.puts("Running on CPU")
end
```

## Next Steps

- [Training Models](02_training.md)
- [Mobile Deployment](03_mobile_deployment.md)
- [Architecture Deep-Dive](04_architecture.md)
