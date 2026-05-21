# Mobile Deployment with ExBurn + Dala

## Overview

ExBurn compiles trained models for mobile deployment via the Dala framework.
The pipeline optimizes models for the target GPU backend:

- **iOS**: Metal via CubeCL
- **Android**: Vulkan via CubeCL

## Compiling for Mobile

```elixir
# Compile for iOS
{:ok, ios_model} = ExBurn.DalaML.compile(model,
  input_shape: {1, 784},
  target: :ios,
  precision: :f16    # f16 quantization for mobile efficiency
)

# Compile for Android
{:ok, android_model} = ExBurn.DalaML.compile(model,
  input_shape: {1, 784},
  target: :android,
  precision: :f16
)
```

## Running Inference

```elixir
{:ok, output} = ExBurn.DalaML.predict(compiled_model, input_tensor)
```

## Exporting

```elixir
# Export for iOS deployment
{:ok, path} = ExBurn.DalaML.export(ios_model, "model_ios.bin")

# Export for Android deployment
{:ok, path} = ExBurn.DalaML.export(android_model, "model_android.bin")
```

## Benchmarking

```elixir
{:ok, stats} = ExBurn.DalaML.benchmark(compiled_model,
  iterations: 100,
  warmup: 10
)

IO.puts("Avg: #{stats.avg_milliseconds}ms")
IO.puts("Min: #{stats.min_microseconds}μs")
IO.puts("Max: #{stats.max_microseconds}μs")
```

## Model Optimization Tips

1. **Use f16 quantization**: Halves memory usage with minimal accuracy loss
2. **Reduce model size**: Target < 10MB for mobile apps
3. **Batch inference**: Process multiple inputs together for better throughput
4. **Profile on device**: Use `ExBurn.DalaML.benchmark/2` on the target device

## Integration with Dala

In your Dala app:

```elixir
defmodule MyApp.ML do
  use Dala.Plugin

  component "image_classifier" do
    prop "model_path", :string
    prop "input_size", :integer
    prop "num_classes", :integer

    event "prediction"

    native "ios", "ExBurnClassifierView"
    native "android", "com.exburn.ClassifierView"
  end
end
```

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
