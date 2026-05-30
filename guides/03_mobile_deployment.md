# Mobile Deployment with ExBurn

## Overview

ExBurn compiles models for mobile deployment via Burn's CubeCL backend:

- **iOS**: Metal via CubeCL
- **Android**: Vulkan via CubeCL

The typical workflow is: train on a desktop GPU → save the model → load and run inference on mobile.

## Training and Saving on Desktop

```elixir
# Train on desktop (CUDA or Metal)
model =
  Axon.input("input", shape: {nil, 784})
  |> Axon.dense(128, activation: :relu)
  |> Axon.dropout(rate: 0.2)
  |> Axon.dense(10)

compiled = ExBurn.Model.compile(model,
  loss: :cross_entropy,
  optimizer: :adam,
  learning_rate: 0.001
)

trained = ExBurn.Training.fit(compiled, {train_x, train_y},
  epochs: 20,
  batch_size: 64
)

# Save for deployment
ExBurn.Model.save(trained, "model.bin")
```

## Loading and Inference on Mobile

```elixir
# Load the model on the mobile device
{:ok, model} = ExBurn.Model.load(compiled, "model.bin")

# Run inference
{:ok, output} = ExBurn.Model.predict(model, input_tensor)
```

## Using ExBurn.Serving for Batched Inference

For production inference with concurrent batching:

```elixir
serving = ExBurn.Serving.build(model,
  batch_size: 32,
  batch_timeout: 50,
  partitions: System.schedulers_online()
)

output = Nx.Serving.run(serving, input_tensor)
```

## Cross-Compilation

### iOS (Metal)

```bash
# Add the iOS target
rustup target add aarch64-apple-ios

# Build the NIF for iOS
cd native/ex_burn_nif
cargo build --target aarch64-apple-ios --features metal --no-default-features --release
```

### Android (Vulkan)

```bash
# Add the Android target
rustup target add aarch64-linux-android

# Build the NIF for Android
cd native/ex_burn_nif
cargo build --target aarch64-linux-android --features vulkan --no-default-features --release
```

### CPU-only Fallback

```bash
cd native/ex_burn_nif
cargo build --no-default-features --release
```

## Model Optimization for Mobile

### 1. Use f16 Precision

Halves memory usage with minimal accuracy loss on inference:

```elixir
# Quantize model parameters to f16
quantized_model = ExBurn.Model.quantize(model, :f16)
```

### 2. Reduce Model Size

| Model Size | Feasibility on Mobile |
|---|---|
| < 1M params | ✅ Comfortable on all modern devices |
| 1M – 10M params | ✅ Fine for inference, training may OOM |
| 10M – 50M params | ⚠️ Inference only, may need quantization |
| > 50M params | ❌ Not recommended for mobile |

### 3. Use ExCubecl Pipelines

Chain multiple GPU kernels without CPU round-trips:

```elixir
{:ok, pipeline} = ExBurn.CubeclBridge.pipeline()
ExBurn.CubeclBridge.pipeline_add(pipeline, :dense, [input_buf, weight_buf, bias_buf], output_buf)
ExBurn.CubeclBridge.pipeline_add(pipeline, :relu, [output_buf], output_buf)
{:ok, _} = ExBurn.CubeclBridge.pipeline_run(pipeline)
```

### 4. Batch Inference

Process multiple inputs together for better GPU utilization:

```elixir
serving = ExBurn.Serving.build(model, batch_size: 16, batch_timeout: 100)
```

## Supported Operations

| Operation | iOS (Metal) | Android (Vulkan) | Notes |
|---|---|---|---|
| Dense / Linear | ✅ | ✅ | |
| Conv2D | ✅ | ✅ | |
| ReLU | ✅ | ✅ | |
| Sigmoid | ✅ | ✅ | |
| Softmax | ✅ | ✅ | |
| Dropout | ✅ | ✅ | No-op during inference |
| LayerNorm | ✅ | ✅ | |
| MatMul | ✅ | ✅ | |
| Transpose | ✅ | ✅ | |
| Reshape | ✅ | ✅ | |
| Concatenate | ✅ | ✅ | |
| Slice | ✅ | ✅ | |

## Memory Considerations

- Burn's Autodiff backend is memory-intensive. **Training on mobile is only feasible for small models** (< 10M parameters).
- **Inference is the primary use case** for mobile deployment.
- Minimum recommended: 4GB RAM, A12+ chip (iOS) / Snapdragon 700+ (Android).
- Use gradient accumulation (`:accumulate_gradients`) to reduce per-batch memory usage.

## Precompiled NIFs

Precompiled NIF binaries are planned for future release via `rustler_precompiled`, which would eliminate the Rust toolchain requirement for end users. The NIF would automatically download the correct binary for the target platform.
