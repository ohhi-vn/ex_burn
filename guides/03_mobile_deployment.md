# Mobile Deployment with ExBurn

## Overview

ExBurn compiles trained models for mobile deployment via Burn's CubeCL backend.
The pipeline optimizes models for the target GPU backend:

- **iOS**: Metal via CubeCL
- **Android**: Vulkan via CubeCL

ExBurn is designed as a library — it provides the Nx backend and GPU
acceleration layer that other frameworks can build on top of.

## Compiling a Model

```elixir
# Define a model with Axon
model =
  Axon.input("input", shape: {nil, 784})
  |> Axon.dense(128, activation: :relu)
  |> Axon.dropout(rate: 0.2)
  |> Axon.dense(10)

# Compile for training/inference
compiled = ExBurn.Model.compile(model,
  loss: :cross_entropy,
  optimizer: :adam,
  learning_rate: 0.001
)

# Run inference
{:ok, output} = ExBurn.Model.predict(compiled, input_tensor)

# Save for deployment
ExBurn.Model.save(compiled, "model.bin")

# Load
{:ok, loaded} = ExBurn.Model.load(compiled, "model.bin")
```

## Using ExCubecl for GPU Inference

ExBurn integrates with ExCubecl for GPU buffer management and kernel execution:

```elixir
# Create GPU buffers via ExCubecl
{:ok, input_buf} = ExCubecl.buffer([1.0, 2.0, 3.0], [3], :f32)
{:ok, output_buf} = ExCubecl.buffer([0.0, 0.0, 0.0], [3], :f32)

# Run a kernel
ExCubecl.run_kernel("elementwise_add", [input_buf, input_buf], output_buf)

# Read results back
{:ok, data} = ExCubecl.read(output_buf)
```

## Using ExBurn.Serving for Batched Inference

For production inference with concurrent batching:

```elixir
# Build a serving from a compiled model
serving = ExBurn.Serving.build(compiled,
  batch_size: 32,
  batch_timeout: 50
)

# Run batched inference
output = Nx.Serving.run(serving, input_tensor)
```

## Model Optimization Tips

1. **Use f16 quantization**: Halves memory usage with minimal accuracy loss
2. **Reduce model size**: Target < 10MB for mobile apps
3. **Batch inference**: Process multiple inputs together for better throughput
4. **Use ExCubecl pipelines**: Chain multiple GPU kernels without CPU round-trips
5. **Profile on device**: Benchmark on the target hardware before deploying

## Supported Operations

| Operation | iOS (Metal) | Android (Vulkan) |
|-----------|-------------|------------------|
| Dense     | ✅          | ✅               |
| Conv2D    | ✅          | ✅               |
| ReLU      | ✅          | ✅               |
| Sigmoid   | ✅          | ✅               |
| Softmax   | ✅          | ✅               |
| Dropout   | ✅          | ✅               |
| LayerNorm | ✅          | ✅               |
