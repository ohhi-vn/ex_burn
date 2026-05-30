# Training Optimization Guide

## Table of Contents

1. [Understanding Gradient Computation](#understanding-gradient-computation)
2. [Choosing an Optimizer](#choosing-an-optimizer)
3. [Learning Rate Strategies](#learning-rate-strategies)
4. [Gradient Clipping](#gradient-clipping)
5. [Weight Decay](#weight-decay)
6. [Gradient Accumulation](#gradient-accumulation)
7. [Batch Size Selection](#batch-size-selection)
8. [Memory Optimization](#memory-optimization)
9. [Common Problems and Solutions](#common-problems-and-solutions)
10. [Performance Benchmarks](#performance-benchmarks)

## Understanding Gradient Computation

### Current Limitation: Numerical Gradients

ExBurn uses **numerical differentiation** (finite differences) to compute gradients. This is the main performance bottleneck.

```
Central differences:  ∂L/∂w ≈ (L(w + ε) - L(w - ε)) / 2ε
One-sided:            ∂L/∂w ≈ (L(w + ε) - L(w)) / ε
```

**Impact**: For a model with N scalar parameters, central differences requires 2N forward passes per mini-batch. A 100K-parameter model needs 200K forward passes per batch.

### Choosing a Gradient Method

```elixir
# Default: central differences (more accurate, slower)
grads = ExBurn.Training.compute_gradients(model, {x, y}, grad_method: :numerical)

# Faster: one-sided differences (less accurate, ~2x faster)
grads = ExBurn.Training.compute_gradients(model, {x, y}, grad_method: :numerical_batch)
```

| Method | Forward Passes | Error Order | When to Use |
|---|---|---|---|
| `:numerical` | 2N | O(ε²) | Small models, high accuracy needed |
| `:numerical_batch` | N+1 | O(ε) | Larger models, speed matters more |

### When Autodiff Arrives

Burn's Autodiff backend will compute exact gradients in a **single backward pass**, regardless of parameter count. This is a game-changer:

```
Numerical (current):  200K forward passes for 100K params
Autodiff (planned):   1 backward pass for any model size
```

**Recommendation**: For now, keep models small (< 50K params) for training. Use larger models only for inference.

## Choosing an Optimizer

### Adam (Default)

Best general-purpose optimizer. Adapts learning rates per-parameter.

```elixir
ExBurn.Model.compile(model, optimizer: :adam, learning_rate: 0.001)
# beta1=0.9, beta2=0.999, epsilon=1e-8
```

**When to use**: Default choice for most tasks. Works well with default hyperparameters.

**Tips**:
- `learning_rate: 0.001` is a good starting point
- Reduce to `0.0001` if training is unstable
- Increase to `0.01` if convergence is very slow

### SGD with Momentum

Can achieve better generalization than Adam with proper tuning.

```elixir
ExBurn.Model.compile(model, optimizer: :sgd, learning_rate: 0.01)
# momentum=0.9
```

**When to use**: When you need maximum generalization and have time to tune.

**Tips**:
- Requires higher learning rate than Adam (typically 0.01–0.1)
- Use Nesterov momentum for faster convergence:
  ```elixir
  ExBurn.Training.fit(model, data, nesterov: true)
  ```
- Combine with cosine annealing LR schedule for best results

### RMSprop

Good for recurrent networks and non-stationary objectives.

```elixir
ExBurn.Model.compile(model, optimizer: :rmsprop, learning_rate: 0.001)
# decay=0.9, epsilon=1e-8
```

**When to use**: RNNs, LSTMs, or when Adam diverges.

### Optimizer Comparison

| Optimizer | Convergence Speed | Generalization | Tuning Effort | Memory |
|---|---|---|---|---|
| Adam | Fast | Good | Low | 2x params (m + v) |
| SGD + Momentum | Medium | Best | High | 1x params (velocity) |
| RMSprop | Medium | Good | Medium | 1x params (cache) |

## Learning Rate Strategies

### Fixed Learning Rate

```elixir
# No schedule — use constant learning rate
ExBurn.Model.compile(model, learning_rate: 0.001)
```

### Step Decay

Reduce LR by a factor every N epochs. Good for long training runs.

```elixir
# Halve the learning rate every 10 epochs
lr_schedule: {:step, 0.001, 10, 0.5}
```

### Exponential Decay

Smooth decay. Good for medium-length training.

```elixir
# Multiply LR by 0.95 each epoch
lr_schedule: {:exponential, 0.001, 0.95}
```

### Cosine Annealing

Smoothly decay from base_lr to min_lr following a cosine curve. Often gives the best results.

```elixir
# Decay from 0.001 to 0.00001 over the training run
lr_schedule: {:cosine, 0.001, 1.0e-5}
```

### Learning Rate Schedule Comparison

```
LR
│
0.001 ─┤ ████
│ ████  ╲         Step (sudden drops)
│  ████  ╲  ╲
│   ████  ╲  ╲
│    ████   ╲   ╲
0.0001 ┤     ╲    ╲
│      ╲     ╲    ╲
│       ╲      ╲    ╲
│        ╲      ╲     ╲
0.00001 ┤──────────────╲──── Cosine (smooth)
└──────────────────────── Epochs
```

### Tips

- Start with Adam + cosine annealing for best results
- If loss oscillates, reduce the base learning rate
- If convergence is too slow, increase the base learning rate
- Use warmup for large batch sizes: `callbacks: [ExBurn.Training.WarmupCallback.linear(5, 1.0e-5, 0.001)]`

## Gradient Clipping

Prevents exploding gradients, which cause NaN loss.

### Clip by Norm

Scales all gradients so their total norm doesn't exceed a threshold:

```elixir
# If ||gradients||_2 > 1.0, scale them down
clip_norm: 1.0
```

**When to use**: Always enable for recurrent networks. Recommended for deep networks.

### Clip by Value

Clips each gradient element to a range:

```elixir
# Clip each gradient to [-5.0, 5.0]
clip_value: 5.0
```

**When to use**: As a safety net alongside norm clipping.

### Tips

- `clip_norm: 1.0` is a good default
- If you see NaN loss, enable clipping immediately
- Clipping doesn't prevent vanishing gradients — use residual connections for that

## Weight Decay

L2 regularization that penalizes large weights, improving generalization:

```elixir
ExBurn.Model.compile(model, weight_decay: 1.0e-4)
```

This adds `weight_decay * param` to each gradient before the optimizer step.

### Tips

- `1.0e-4` is a good default for most tasks
- `1.0e-5` for small datasets (less regularization)
- `1.0e-3` for large models that overfit
- Don't use with AdamW (not yet implemented) — with standard Adam, weight decay interacts with the adaptive learning rate

## Gradient Accumulation

Simulates a larger batch size by accumulating gradients across multiple mini-batches:

```elixir
# Effective batch size = 32 * 4 = 128
ExBurn.Training.fit(model, data,
  batch_size: 32,
  accumulate_gradients: 4
)
```

### When to Use

- GPU memory limits your batch size
- You want the stability of large batches but can't fit them in memory
- Training on mobile devices with limited RAM

### Tips

- Increase learning rate proportionally to the accumulation factor (e.g., 4x accumulation → 2x LR)
- Batch normalization (when available) will still see the small mini-batch statistics

## Batch Size Selection

| Batch Size | Pros | Cons |
|---|---|---|
| 8–16 | Better generalization, less memory | Noisy gradients, slower training |
| 32–64 | Good default | Balanced |
| 128–256 | Faster training, stable gradients | May generalize worse, more memory |
| 512+ | Very stable gradients | Often worse generalization, high memory |

### Tips

- Start with 32 and increase if you have memory headroom
- If you increase batch size, increase learning rate proportionally
- Use gradient accumulation to simulate large batches on memory-constrained devices

## Memory Optimization

### On Desktop (CUDA/Metal)

```elixir
# Use f16 for 2x memory reduction
# (convert parameters to f16 before training)

# Use gradient accumulation to reduce per-batch memory
accumulate_gradients: 4
```

### On Mobile (iOS/Android)

```elixir
# Keep models small (< 10M params)
# Use CPU for training (GPU autodiff is memory-intensive)
ExBurn.Model.compile(model, device: :cpu)

# Free intermediate tensors explicitly
ExBurn.Tensor.free(intermediate_tensor)
```

### Memory-Saving Tips

1. **Reduce batch size** — the single biggest lever
2. **Use gradient accumulation** — same effective batch, less memory
3. **Free tensors explicitly** — don't wait for GC
4. **Use f16 precision** — halves memory for tensors
5. **Avoid storing all intermediate activations** — use `:accumulate_gradients` to reduce peak memory

## Common Problems and Solutions

### Loss is NaN

**Causes**: Exploding gradients, too high learning rate, numerical instability

**Solutions**:
```elixir
# 1. Enable gradient clipping
clip_norm: 1.0

# 2. Reduce learning rate
learning_rate: 0.0001

# 3. Use :numerical_batch gradient method (more stable)
grad_method: :numerical_batch
```

### Loss Doesn't Decrease

**Causes**: Too low learning rate, bad initialization, wrong loss function

**Solutions**:
```elixir
# 1. Increase learning rate
learning_rate: 0.01

# 2. Check loss function matches task
#    Classification → :cross_entropy
#    Regression → :mse
#    Binary → :binary_cross_entropy

# 3. Verify data preprocessing (normalization, etc.)
```

### Loss Oscillates

**Causes**: Learning rate too high, batch size too small

**Solutions**:
```elixir
# 1. Reduce learning rate
learning_rate: 0.0005

# 2. Increase batch size or use gradient accumulation
accumulate_gradients: 4

# 3. Use learning rate schedule
lr_schedule: {:cosine, 0.001, 1.0e-6}
```

### Overfitting

**Causes**: Model too complex, not enough data, no regularization

**Solutions**:
```elixir
# 1. Add weight decay
weight_decay: 1.0e-3

# 2. Add dropout in the Axon model
|> Axon.dropout(rate: 0.5)

# 3. Freeze early layers
model = ExBurn.Model.freeze(model, ["hidden1"])

# 4. Use early stopping
callbacks: [ExBurn.Training.EarlyStoppingCallback.wait(5)]
```

### Training is Very Slow

**Causes**: Numerical gradients on large model, too many epochs

**Solutions**:
```elixir
# 1. Use faster gradient method
grad_method: :numerical_batch

# 2. Reduce model size
# 3. Use fewer epochs with early stopping
callbacks: [ExBurn.Training.EarlyStoppingCallback.wait(3)]

# 4. Increase batch size (fewer optimizer steps)
batch_size: 128
```

## Performance Benchmarks

Approximate training times per epoch on synthetic data (will vary by hardware):

| Model Size | Params | Batch | Method | Time/Epoch |
|---|---|---|---|---|
| Tiny MLP | 1K | 32 | :numerical | ~2s |
| Small MLP | 10K | 32 | :numerical | ~15s |
| Small MLP | 10K | 32 | :numerical_batch | ~8s |
| Medium MLP | 100K | 32 | :numerical | ~3min |
| Medium MLP | 100K | 32 | :numerical_batch | ~1.5min |

**Key takeaway**: With numerical gradients, training time scales linearly with parameter count. Keep models under 50K parameters for interactive training, or switch to inference-only for larger models until autodiff arrives in v0.3.0.

## Quick Reference: Recommended Settings

### For Quick Experiments

```elixir
compiled = ExBurn.Model.compile(model,
  loss: :cross_entropy,
  optimizer: :adam,
  learning_rate: 0.001
)

ExBurn.Training.fit(compiled, data,
  epochs: 10,
  batch_size: 32,
  verbose: true
)
```

### For Best Results

```elixir
compiled = ExBurn.Model.compile(model,
  loss: :cross_entropy,
  optimizer: :adam,
  learning_rate: 0.001,
  weight_decay: 1.0e-4
)

ExBurn.Training.fit(compiled, data,
  epochs: 50,
  batch_size: 64,
  shuffle: true,
  validation_data: val_data,
  lr_schedule: {:cosine, 0.001, 1.0e-6},
  clip_norm: 1.0,
  accuracy: true,
  callbacks: [
    &ExBurn.Training.LoggingCallback.log/1,
    ExBurn.Training.EarlyStoppingCallback.wait(10, 1.0e-5),
    ExBurn.Training.CheckpointCallback.every(10, "/checkpoints")
  ]
)
```

### For Memory-Constrained Devices

```elixir
compiled = ExBurn.Model.compile(model,
  loss: :cross_entropy,
  optimizer: :adam,
  learning_rate: 0.0005,
  device: :cpu
)

ExBurn.Training.fit(compiled, data,
  epochs: 20,
  batch_size: 16,
  accumulate_gradients: 4,
  clip_norm: 1.0
)
```
