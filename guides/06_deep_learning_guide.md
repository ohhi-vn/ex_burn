# Deep Learning with ExBurn: A Step-by-Step Guide

## Table of Contents

1. [What You'll Learn](#what-youll-learn)
2. [Prerequisites](#prerequisites)
3. [Lesson 1: Tensors — The Building Blocks](#lesson-1-tensors--the-building-blocks)
4. [Lesson 2: Your First Neural Network](#lesson-2-your-first-neural-network)
5. [Lesson 3: Training a Classifier](#lesson-3-training-a-classifier)
6. [Lesson 4: Understanding Loss Functions](#lesson-4-understanding-loss-functions)
7. [Lesson 5: Optimizers and Learning Rates](#lesson-5-optimizers-and-learning-rates)
8. [Lesson 6: Overfitting and Regularization](#lesson-6-overfitting-and-regularization)
9. [Lesson 7: Working with Real Data](#lesson-7-working-with-real-data)
10. [Lesson 8: Inference and Deployment](#lesson-8-inference-and-deployment)
11. [Lesson 9: GPU-Accelerated Numerical Functions](#lesson-9-gpu-accelerated-numerical-functions)
12. [Lesson 10: Putting It All Together](#lesson-10-putting-it-all-together)

---

## What You'll Learn

This guide teaches deep learning fundamentals through hands-on ExBurn examples. By the end, you'll be able to:

- Create and manipulate tensors (the core data structure of deep learning)
- Build neural network architectures using Axon
- Train models with different optimizers and learning rate strategies
- Prevent overfitting with regularization techniques
- Preprocess real-world data
- Run inference and deploy models
- Write GPU-accelerated numerical functions with `defn`

Each lesson builds on the previous one. Code examples are complete and runnable.

---

## Prerequisites

- Elixir ~> 1.18 and OTP 27+
- Rust stable (for NIF compilation)
- Basic Elixir knowledge (modules, functions, pipes)
- No prior deep learning experience required

Add to your `mix.exs`:

```elixir
def deps do
  [
    {:ex_burn, "~> 0.3"},
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

Check that your GPU is available:

```elixir
ExBurn.default_device()   # :gpu or :cpu
ExBurn.device_name()      # e.g. "CUDA (NVIDIA GPU)" or "Metal (Apple GPU)"
ExBurn.summary()          # full environment summary
```

---

## Lesson 1: Tensors — The Building Blocks

### What is a Tensor?

A tensor is a multi-dimensional array of numbers. Deep learning is essentially tensor math:

| Tensor rank | Example | Shape |
|---|---|---|
| 0 (scalar) | `5.0` | `{}` |
| 1 (vector) | `[1.0, 2.0, 3.0]` | `{3}` |
| 2 (matrix) | `[[1, 2], [3, 4]]` | `{2, 2}` |
| 3 (image) | batch of 8 RGB 32x32 images | `{8, 3, 32, 32}` |

### Creating Tensors

```elixir
import Nx

# From a list
t = Nx.tensor([1.0, 2.0, 3.0])

# 2D tensor (matrix)
m = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])

# With explicit type
t_f64 = Nx.tensor([1.0, 2.0], type: {:f, 64})
t_i32 = Nx.tensor([1, 2, 3], type: {:s, 32})

# Useful constructors
zeros = Nx.broadcast(0.0, {3, 4})     # 3x4 matrix of zeros
ones  = Nx.broadcast(1.0, {3, 4})     # 3x4 matrix of ones
iota  = Nx.iota({5})                   # [0, 1, 2, 3, 4]
eye   = Nx.eye(3)                      # 3x3 identity matrix
```

### Inspecting Tensors

```elixir
Nx.shape(t)     # {3} — the shape
Nx.type(t)      # {:f, 32} — the element type
Nx.rank(t)      # 1 — number of dimensions
Nx.size(t)      # 3 — total number of elements
Nx.to_list(t)   # [1.0, 2.0, 3.0] — convert to Elixir list
```

### Element-wise Operations

```elixir
a = Nx.tensor([1.0, 2.0, 3.0])
b = Nx.tensor([4.0, 5.0, 6.0])

Nx.add(a, b)        # [5.0, 7.0, 9.0]
Nx.subtract(a, b)   # [-3.0, -3.0, -3.0]
Nx.multiply(a, b)   # [4.0, 10.0, 18.0]
Nx.divide(a, b)     # [0.25, 0.4, 0.5]
Nx.negate(a)        # [-1.0, -2.0, -3.0]
Nx.abs(a)           # [1.0, 2.0, 3.0]
Nx.exp(a)           # [2.718, 7.389, 20.085]
Nx.log(a)           # [0.0, 0.693, 1.099]
Nx.sqrt(a)          # [1.0, 1.414, 1.732]
```

### Broadcasting

When shapes don't match, Nx automatically broadcasts the smaller tensor:

```elixir
a = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])  # shape {2, 2}
b = Nx.tensor([10.0, 20.0])               # shape {2}

Nx.add(a, b)
# [[11.0, 22.0],
#  [13.0, 24.0]]
# b is broadcast across rows
```

### Reductions

Collapse dimensions to produce summaries:

```elixir
m = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])

Nx.sum(m)           # 10.0 — sum all elements
Nx.mean(m)          # 2.5 — mean of all elements
Nx.reduce_max(m)    # 4.0 — maximum value
Nx.reduce_min(m)    # 1.0 — minimum value

# Reduce along a specific axis
Nx.sum(m, axes: [0])  # [4.0, 6.0] — sum along rows (column sums)
Nx.sum(m, axes: [1])  # [3.0, 7.0] — sum along columns (row sums)
```

### Shape Manipulation

```elixir
t = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

Nx.reshape(t, {2, 3})
# [[1.0, 2.0, 3.0],
#  [4.0, 5.0, 6.0]]

m = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
Nx.transpose(m)
# [[1.0, 4.0],
#  [2.0, 5.0],
#  [3.0, 6.0]]

# Concatenation
a = Nx.tensor([1.0, 2.0])
b = Nx.tensor([3.0, 4.0])
Nx.concatenate([a, b])  # [1.0, 2.0, 3.0, 4.0]
```

### Linear Algebra

```elixir
a = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
b = Nx.tensor([[5.0, 6.0], [7.0, 8.0]])

Nx.dot(a, b)
# Matrix multiplication:
# [[1*5+2*7, 1*6+2*8],
#  [3*5+4*7, 3*6+4*8]]
# = [[19.0, 22.0], [43.0, 50.0]]

# Dot product of vectors
x = Nx.tensor([1.0, 2.0, 3.0])
y = Nx.tensor([4.0, 5.0, 6.0])
Nx.dot(x, y)  # 1*4 + 2*5 + 3*6 = 32.0
```

### Try It Yourself

```elixir
# Create a 3x3 matrix, transpose it, then multiply by the original
m = Nx.iota({3, 3}) |> Nx.as_type(:f32)
mt = Nx.transpose(m)
result = Nx.dot(m, mt)
Nx.to_list(result)
```

---

## Lesson 2: Your First Neural Network

### What is a Neural Network?

A neural network is a function that transforms input data into predictions through a series of learned transformations:

```
input → [Linear → Activation] × N → output
```

Each **Linear** layer computes `output = input × weights + bias`. The **Activation** function introduces non-linearity, enabling the network to learn complex patterns.

### Defining a Model with Axon

Axon provides a functional, Keras-like API for building models:

```elixir
model =
  Axon.input("input", shape: {nil, 4})
  |> Axon.dense(8, activation: :relu)
  |> Axon.dense(3, activation: :softmax)
```

Breaking this down:

- `Axon.input("input", shape: {nil, 4})` — defines the input. `nil` means "any batch size", `4` means 4 features per sample.
- `Axon.dense(8, activation: :relu)` — a fully-connected layer with 8 neurons and ReLU activation.
- `Axon.dense(3, activation: :softmax)` — output layer with 3 neurons (one per class) and softmax activation.

### Understanding Layer Shapes

```elixir
# Input: {batch_size, 4}
#   ↓ Dense(8)  — learns a {4, 8} weight matrix + {8} bias
# Hidden: {batch_size, 8}
#   ↓ Dense(3)  — learns a {8, 3} weight matrix + {3} bias
# Output: {batch_size, 3}
```

The `nil` in the input shape is the batch dimension — it can be any size.

### Compiling the Model

Before training, we need to compile the model. This initializes parameters and sets up the optimizer:

```elixir
compiled = ExBurn.Model.compile(model,
  loss: :cross_entropy,
  optimizer: :adam,
  learning_rate: 0.01
)
```

### Inspecting the Model

```elixir
# Keras/PyTorch-style summary
IO.puts(ExBurn.Model.summary(compiled))

# Get model info
info = ExBurn.Model.info(compiled)
IO.puts("Total parameters: #{info.total_params}")
IO.puts("Layers: #{info.layer_count}")
IO.puts("Memory: #{info.estimated_memory_mb} MB")

# Access individual components
ExBurn.Model.parameters(compiled)     # parameter map
ExBurn.Model.loss_function(compiled)  # :cross_entropy
ExBurn.Model.optimizer(compiled)      # :adam
```

### Forward Pass (Inference)

```elixir
# Create some dummy input
input = Nx.tensor([[1.0, 2.0, 3.0, 4.0]])

# Run inference
{:ok, output} = ExBurn.Model.predict(compiled, input)
Nx.to_list(output)
# e.g. [[0.2, 0.5, 0.3]] — class probabilities from softmax
```

### Activation Functions

Activation functions introduce non-linearity. Without them, stacking linear layers would be equivalent to a single linear layer:

```elixir
# Common activations in Axon:
Axon.dense(64, activation: :relu)         # ReLU: max(0, x) — most common
Axon.dense(64, activation: :sigmoid)      # Sigmoid: 1/(1+e^-x) — outputs in [0,1]
Axon.dense(64, activation: :tanh)         # Tanh: outputs in [-1, 1]
Axon.dense(64, activation: :softmax)      # Softmax: normalizes to probabilities
```

**ReLU** (Rectified Linear Unit) is the default choice for hidden layers. It's simple, fast, and avoids the vanishing gradient problem.

### Try It Yourself

```elixir
# Build a model with 2 hidden layers
model =
  Axon.input("x", shape: {nil, 10})
  |> Axon.dense(32, activation: :relu, name: "hidden1")
  |> Axon.dense(16, activation: :relu, name: "hidden2")
  |> Axon.dense(5, name: "output")

compiled = ExBurn.Model.compile(model)
IO.puts(ExBurn.Model.summary(compiled))
```

---

## Lesson 3: Training a Classifier

### The Training Loop

Training is the process of adjusting the model's parameters to minimize the loss function. Each iteration:

1. **Forward pass**: Compute predictions from input data
2. **Loss computation**: Measure how wrong the predictions are
3. **Backward pass**: Compute gradients (how to adjust each parameter)
4. **Optimizer step**: Update parameters to reduce loss

### Complete Training Example

```elixir
import Nx

# ── Step 1: Create synthetic data ──────────────────────────
# 100 samples, 4 features, 3 classes
num_samples = 100
num_features = 4
num_classes = 3

# Random features
x = Nx.random_uniform({num_samples, num_features})

# Random integer labels (0, 1, or 2)
y = Nx.random_uniform({num_samples}, type: {:u, 8})
y = Nx.remainder(y, num_classes) |> Nx.as_type({:s, 64})

# ── Step 2: Split into train/validation ────────────────────
{train, val} = ExBurn.Dataset.split({x, y}, val_split: 0.2, shuffle: false)
{train_x, train_y} = train
{val_x, val_y} = val

# ── Step 3: Define the model ──────────────────────────────
model =
  Axon.input("input", shape: {nil, num_features})
  |> Axon.dense(8, activation: :relu)
  |> Axon.dense(num_classes)

# ── Step 4: Compile ───────────────────────────────────────
compiled = ExBurn.Model.compile(model,
  loss: :cross_entropy,
  optimizer: :adam,
  learning_rate: 0.01
)

# ── Step 5: Train ─────────────────────────────────────────
trained = ExBurn.Training.fit(compiled, {train_x, train_y},
  epochs: 20,
  batch_size: 16,
  validation_data: {val_x, val_y},
  verbose: true
)

# ── Step 6: Evaluate ──────────────────────────────────────
{loss, accuracy} = ExBurn.Training.evaluate(trained, {val_x, val_y}, true)
IO.puts("Validation loss: #{loss}, accuracy: #{accuracy}")
```

### Understanding the Output

When `verbose: true`, you'll see output like:

```
Training: 80 samples, 5 batches/epoch, 20 epochs
  batch_size=16, effective_batch_size=16, optimizer=adam
Epoch 1: loss=1.0986 (1250 samples/s, 64ms) ETA=1s
Epoch 2: loss=1.0852 (1300 samples/s, 61ms) ETA=1s
...
Epoch 20: loss=0.5234 (1350 samples/s, 59ms)
```

Key metrics:
- **loss**: The average loss per batch (lower is better)
- **samples/s**: Training throughput
- **ETA**: Estimated time remaining

### Batch Size

The `batch_size` controls how many samples are processed before updating parameters:

```elixir
# Small batch: noisier gradients, slower training, less memory
batch_size: 8

# Large batch: smoother gradients, faster training, more memory
batch_size: 64
```

### Epochs

One epoch = one full pass through the training data. More epochs = more training, but too many can cause overfitting.

### Try It Yourself

```elixir
# Experiment: try different batch sizes and learning rates
# Which combination converges fastest?
# Which gives the best final accuracy?
```

---

## Lesson 4: Understanding Loss Functions

### What is a Loss Function?

A loss function measures how far the model's predictions are from the true values. Training aims to minimize this value.

### Cross-Entropy Loss (Classification)

Used for multi-class classification. Measures the difference between predicted class probabilities and true labels:

```elixir
# Target as integer class indices
pred = Nx.tensor([[2.0, 1.0, 0.1]])   # model logits for 3 classes
target = Nx.tensor([0])                  # true class is 0

# Or target as one-hot encoded
target_onehot = Nx.tensor([[1.0, 0.0, 0.0]])
```

The loss is lower when the model assigns high probability to the correct class:

```elixir
model = Axon.input("x", shape: {nil, 3}) |> Axon.dense(3)
compiled = ExBurn.Model.compile(model, loss: :cross_entropy)

# Good prediction → low loss
good_pred = Nx.tensor([[10.0, 0.1, 0.1]])   # confident and correct
{:ok, loss} = ExBurn.Model.compute_loss(compiled, good_pred, Nx.tensor([0]))
# loss ≈ 0.0001

# Bad prediction → high loss
bad_pred = Nx.tensor([[0.1, 0.1, 10.0]])    # confident but wrong
{:ok, loss} = ExBurn.Model.compute_loss(compiled, bad_pred, Nx.tensor([0]))
# loss ≈ 10.0
```

### Mean Squared Error (Regression)

Used for regression tasks where the target is a continuous value:

```elixir
model = Axon.input("x", shape: {nil, 5}) |> Axon.dense(1)
compiled = ExBurn.Model.compile(model, loss: :mse)

pred = Nx.tensor([[3.0]])
target = Nx.tensor([[5.0]])

{:ok, loss} = ExBurn.Model.compute_loss(compiled, pred, target)
# MSE = (3-5)² = 4.0
```

### Binary Cross-Entropy (Binary Classification)

Used when there are exactly two classes:

```elixir
model = Axon.input("x", shape: {nil, 10}) |> Axon.dense(1)
compiled = ExBurn.Model.compile(model, loss: :binary_cross_entropy)

# Targets are 0.0 or 1.0
pred = Nx.tensor([[0.9]])     # model predicts class 1 with 90% confidence
target = Nx.tensor([[1.0]])   # true class is 1

{:ok, loss} = ExBurn.Model.compute_loss(compiled, pred, target)
# loss ≈ 0.105 (low, because prediction matches target)
```

### Choosing the Right Loss

| Task | Loss Function | Target Format |
|---|---|---|
| Multi-class classification | `:cross_entropy` | Integer indices or one-hot |
| Binary classification | `:binary_cross_entropy` | 0.0 or 1.0 |
| Regression | `:mse` | Continuous values |

---

## Lesson 5: Optimizers and Learning Rates

### What is an Optimizer?

An optimizer determines how to update the model's parameters based on the computed gradients. Different optimizers have different strategies.

### Adam (Default)

Adam adapts the learning rate for each parameter individually. It's a good default for most tasks:

```elixir
ExBurn.Model.compile(model,
  optimizer: :adam,
  learning_rate: 0.001    # good starting point
)
```

**When to use**: Default choice. Works well with minimal tuning.

**Tips**:
- If loss oscillates → reduce learning rate (try `0.0001`)
- If convergence is very slow → increase learning rate (try `0.01`)

### SGD with Momentum

SGD with momentum accumulates a velocity vector in directions of consistent gradient:

```elixir
ExBurn.Model.compile(model,
  optimizer: :sgd,
  learning_rate: 0.01       # needs higher LR than Adam
)

# With Nesterov momentum (often converges faster):
ExBurn.Training.fit(model, data, nesterov: true)
```

**When to use**: When you need maximum generalization and have time to tune.

### RMSprop

RMSprop adapts learning rates based on the magnitude of recent gradients:

```elixir
ExBurn.Model.compile(model,
  optimizer: :rmsprop,
  learning_rate: 0.001
)
```

**When to use**: RNNs, LSTMs, or when Adam diverges.

### Learning Rate Schedules

Instead of a fixed learning rate, you can vary it during training:

```elixir
# Step decay: halve LR every 10 epochs
ExBurn.Training.fit(model, data,
  lr_schedule: {:step, 0.001, 10, 0.5}
)

# Exponential decay: multiply LR by 0.95 each epoch
ExBurn.Training.fit(model, data,
  lr_schedule: {:exponential, 0.001, 0.95}
)

# Cosine annealing: smooth decay (often best results)
ExBurn.Training.fit(model, data,
  lr_schedule: {:cosine, 0.001, 1.0e-5}
)
```

Visual comparison:

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

### Warmup

Gradually increase the learning rate at the start of training for stability:

```elixir
ExBurn.Training.fit(model, data,
  callbacks: [
    ExBurn.Training.WarmupCallback.linear(5, 1.0e-5, 0.001)
  ]
)
```

This ramps the LR from `1.0e-5` to `0.001` over the first 5 epochs.

### Reduce on Plateau

Automatically reduce the learning rate when validation loss stops improving:

```elixir
ExBurn.Training.fit(model, data,
  callbacks: [
    ExBurn.Training.ReduceLROnPlateauCallback.new(
      patience: 5,
      factor: 0.5,
      min_lr: 1.0e-6
    )
  ]
)
```

### Try It Yourself

```elixir
# Compare optimizers on the same data:
# 1. Adam with lr=0.001
# 2. SGD with lr=0.01 and nesterov=true
# 3. Adam with cosine annealing
# Which converges fastest? Which gives the best final loss?
```

---

## Lesson 6: Overfitting and Regularization

### What is Overfitting?

Overfitting happens when the model memorizes the training data instead of learning general patterns. Signs:

- Training loss keeps decreasing, but validation loss starts increasing
- Large gap between training and validation accuracy

```
Loss
│
│  ╲        ╱ ── training loss (keeps decreasing)
│   ╲      ╱
│    ╲    ╱
│     ╲  ╱  ╱── validation loss (starts increasing = overfitting!)
│      ╲╱  ╱
│          ╱
└──────────────── Epochs
```

### Technique 1: Dropout

Randomly "drops" (sets to zero) a fraction of neurons during training. Forces the network to not rely on any single neuron:

```elixir
model =
  Axon.input("x", shape: {nil, 10})
  |> Axon.dense(64, activation: :relu)
  |> Axon.dropout(rate: 0.5)          # drop 50% of neurons
  |> Axon.dense(64, activation: :relu)
  |> Axon.dropout(rate: 0.3)          # drop 30% of neurons
  |> Axon.dense(3)
```

**Rule of thumb**: Use `rate: 0.2-0.5` for hidden layers. Don't use dropout on the output layer.

### Technique 2: Weight Decay (L2 Regularization)

Penalizes large weights, encouraging the model to learn simpler patterns:

```elixir
ExBurn.Model.compile(model,
  weight_decay: 1.0e-4    # L2 regularization coefficient
)
```

**Rule of thumb**:
- `1.0e-4` — good default
- `1.0e-5` — small datasets (less regularization needed)
- `1.0e-3` — large models that overfit

### Technique 3: Early Stopping

Stop training when validation loss stops improving:

```elixir
ExBurn.Training.fit(model, data,
  validation_data: val_data,
  callbacks: [
    ExBurn.Training.EarlyStoppingCallback.wait(5, 1.0e-4)
  ]
)
```

This stops training after 5 epochs without at least `1.0e-4` improvement in validation loss.

### Technique 4: Gradient Clipping

Prevents exploding gradients (which cause NaN loss):

```elixir
ExBurn.Training.fit(model, data,
  clip_norm: 1.0,     # clip gradient norm to 1.0
  clip_value: 5.0     # also clip individual gradient values to [-5, 5]
)
```

### Technique 5: Freezing Layers

When fine-tuning a pre-trained model, freeze early layers to preserve learned features:

```elixir
# Freeze the first layer
frozen_model = ExBurn.Model.freeze(model, ["hidden1"])

# Check which layers are frozen
ExBurn.Model.frozen_layers(frozen_model)  # #MapSet<["hidden1"]>

# Unfreeze later
unfrozen_model = ExBurn.Model.unfreeze(frozen_model, ["hidden1"])
```

### Try It Yourself

```elixir
# Train a model WITHOUT regularization → observe overfitting
# Then add dropout + weight decay + early stopping → compare
```

---

## Lesson 7: Working with Real Data

### Data Splitting

Always split your data into training, validation, and test sets:

```elixir
# Split into 80% train, 20% validation
{train, val} = ExBurn.Dataset.split({x, y}, val_split: 0.2, shuffle: true, seed: 42)

# For a three-way split:
{train, temp} = ExBurn.Dataset.split({x, y}, val_split: 0.3, seed: 42)
{val, test} = ExBurn.Dataset.split(temp, val_split: 0.5, seed: 42)
# Result: 70% train, 15% val, 15% test
```

Use `seed` for reproducible splits.

### Data Loading

Create a batched data loader for efficient training:

```elixir
loader = ExBurn.Dataset.loader({x, y},
  batch_size: 32,
  shuffle: true,
  drop_last: false    # keep partial last batch
)

# Iterate through batches
Enum.each(loader, fn {batch_x, batch_y} ->
  # process batch
end)
```

### Normalization

Neural networks train better when input features are on a similar scale:

```elixir
# Standard normalization: zero mean, unit variance
{train_norm, stats} = ExBurn.Dataset.normalize(train_x, method: :standard)

# Apply the same transformation to validation/test data
val_norm = ExBurn.Dataset.normalize_with_stats(val_x, stats)
```

Three normalization methods:

| Method | What it does | When to use |
|---|---|---|
| `:standard` | `(x - mean) / std` | Default for most features |
| `:minmax` | `(x - min) / (max - min)` | When you need values in [0, 1] |
| `:l2` | `x / ||x||_2` | When direction matters more than magnitude |

**Important**: Always compute statistics on training data only, then apply them to validation/test data.

### One-Hot Encoding

Convert integer class labels to one-hot vectors:

```elixir
labels = Nx.tensor([0, 2, 1, 3])
one_hot = ExBurn.Dataset.one_hot(labels, num_classes: 4)
# [[1, 0, 0, 0],
#  [0, 0, 1, 0],
#  [0, 1, 0, 0],
#  [0, 0, 0, 1]]
```

### Dataset Statistics

```elixir
stats = ExBurn.Dataset.stats({x, y})
# %{num_samples: 100, input_shape: {100, 4}, target_shape: {100},
#   input_type: {:f, 32}, target_type: {:s, 64}}
```

### Complete Data Pipeline Example

```elixir
# 1. Load your data (however you get it)
# x = ...  # your features
# y = ...  # your labels

# 2. Split
{train, val} = ExBurn.Dataset.split({x, y}, val_split: 0.2, seed: 42)

# 3. Normalize
{train_x_norm, norm_stats} = ExBurn.Dataset.normalize(elem(train, 0), method: :standard)
val_x_norm = ExBurn.Dataset.normalize_with_stats(elem(val, 0), norm_stats)

# 4. Train
{ExBurn.Model.compile(model), {train_x_norm, elem(train, 1)}}
|> then(fn {compiled, train_data} ->
  ExBurn.Training.fit(compiled, train_data,
    validation_data: {val_x_norm, elem(val, 1)},
    epochs: 50
  )
end)
```

---

## Lesson 8: Inference and Deployment

### Running Inference

After training, use the model to make predictions:

```elixir
# Single prediction
input = Nx.tensor([[1.0, 2.0, 3.0, 4.0]])
{:ok, output} = ExBurn.Model.predict(trained_model, input)
Nx.argmax(output)  # predicted class

# Batch prediction
batch = Nx.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
{:ok, outputs} = ExBurn.Model.predict(trained_model, batch)
```

### GPU vs CPU Inference

```elixir
# GPU inference (via defn compiler)
{:ok, output} = ExBurn.Model.forward(trained_model, input)

# CPU inference (via Axon predict)
{:ok, output} = ExBurn.Model.predict(trained_model, input)
```

### Batched Concurrent Inference with Serving

For production use, `Nx.Serving` handles concurrent batching:

```elixir
serving = ExBurn.Serving.build(trained_model,
  batch_size: 32,
  batch_timeout: 50,
  partitions: System.schedulers_online()
)

# Run inference
output = Nx.Serving.run(serving, input)
```

### Saving and Loading Models

```elixir
# Save to file
ExBurn.Model.save(trained_model, "my_model.bin")

# Load from file
{:ok, loaded_model} = ExBurn.Model.load(compiled_model, "my_model.bin")

# Serialize to binary (for network transfer)
binary = ExBurn.Model.serialize_params(trained_model)
{:ok, params} = ExBurn.Model.deserialize_params(binary)
```

### Export Formats

```elixir
# Compressed Erlang terms (default, portable)
ExBurn.Model.export(model, "model.etf", format: :elixir_terms)

# JSON (human-readable, larger)
ExBurn.Model.export(model, "model.json", format: :json)

# Import
{:ok, model} = ExBurn.Model.import_params(model, "model.etf")
{:ok, model} = ExBurn.Model.import_params(model, "model.json", format: :json)
```

### Model Quantization

Reduce model size for deployment:

```elixir
# Convert to half precision (f16) — 2x smaller
quantized = ExBurn.Model.quantize(trained_model, :f16)

# Or brain float 16 (bf16) — better range than f16
quantized = ExBurn.Model.quantize(trained_model, :bf16)
```

### Benchmarking

Measure inference speed:

```elixir
results = ExBurn.Model.benchmark(trained_model, input, warmup: 3, runs: 10)
# %{avg_ms: 1.234, min_ms: 1.100, max_ms: 1.500,
#   median_ms: 1.200, std_ms: 0.120, runs: 10, warmup: 3}
```

---

## Lesson 9: GPU-Accelerated Numerical Functions

### What is `defn`?

`defn` lets you write numerical functions that run on the GPU. The `ExBurn.Defn.Compiler` traces your function and compiles it to Burn GPU kernels.

### Setup

```elixir
Nx.default_backend(ExBurn.Backend)
Nx.Defn.global_default_options(compiler: ExBurn.Defn.Compiler)
```

### Writing `defn` Functions

```elixir
defmodule MyMath do
  import Nx.Defn

  # Element-wise sigmoid: 1 / (1 + e^(-x))
  defn sigmoid(x) do
    Nx.divide(1.0, Nx.add(1.0, Nx.exp(Nx.negate(x))))
  end

  # Linear regression prediction: X @ w + b
  defn predict(X, w, b) do
    Nx.add(Nx.dot(X, w), b)
  end

  # Mean squared error
  defn mse_loss(y_true, y_pred) do
    diff = Nx.subtract(y_true, y_pred)
    Nx.mean(Nx.multiply(diff, diff))
  end

  # ReLU activation
  defn relu(x) do
    Nx.max(x, 0.0)
  end

  # L2 normalization
  defn l2_normalize(x) do
    norm = Nx.sqrt(Nx.sum(Nx.multiply(x, x), axes: [-1], keep_axes: true))
    Nx.divide(x, norm)
  end
end

# These all run on the GPU!
MyMath.sigmoid(Nx.tensor([1.0, 2.0, 3.0]))
MyMath.relu(Nx.tensor([-1.0, 0.0, 1.0]))
```

### Per-Function Compiler Override

```elixir
defmodule MyModule do
  import Nx.Defn

  # This function uses ExBurn's GPU compiler
  defn gpu_function(x) do
    Nx.sin(x) |> Nx.exp()
  end
  compiler: ExBurn.Defn.Compiler

  # This function uses the default (CPU) compiler
  defn cpu_function(x) do
    Nx.cos(x)
  end
end
```

### Control Flow in `defn`

```elixir
defmodule ControlFlow do
  import Nx.Defn

  defn clip_and_scale(x, min_val, max_val, scale) do
    x
    |> Nx.clip(min_val, max_val)
    |> Nx.multiply(scale)
  end

  defn conditional_compute(x, threshold) do
    # Use Nx.select for conditional operations
    Nx.select(
      Nx.greater(x, threshold),  # condition
      Nx.multiply(x, 2.0),        # value when true
      Nx.divide(x, 2.0)           # value when false
    )
  end
end
```

### Using BurnBridge Directly

For maximum performance, bypass Nx and talk to Burn directly:

```elixir
# Create tensors directly on the GPU
t1 = ExBurn.BurnBridge.zeros([100, 100], :f32)
t2 = ExBurn.BurnBridge.ones([100, 100], :f32)

# Each operation is a single NIF call
t3 = ExBurn.BurnBridge.add(t1, t2)
t4 = ExBurn.BurnBridge.matmul(t1, t2)
t5 = ExBurn.BurnBridge.relu(t3)

# Convert back to Nx when needed
nx_tensor = ExBurn.BurnBridge.to_nx(t3)
```

### Try It Yourself

```elixir
# Implement a GPU-accelerated softmax function using defn
defmodule SoftmaxGPU do
  import Nx.Defn

  defn softmax(x) do
    # Numerically stable softmax
    shifted = x - Nx.reduce_max(x, axes: [-1], keep_axes: true)
    exp_shifted = Nx.exp(shifted)
    exp_shifted / Nx.sum(exp_shifted, axes: [-1], keep_axes: true)
  end
end

# Test it
input = Nx.tensor([[1.0, 2.0, 3.0]])
SoftmaxGPU.softmax(input)
# Should sum to 1.0 across the last dimension
```

---

## Lesson 10: Putting It All Together

### Complete Example: Iris-like Classification

This example combines everything from the previous lessons:

```elixir
import Nx

# ── 1. Prepare Data ────────────────────────────────────────
num_samples = 150
num_features = 4
num_classes = 3

# Synthetic data (replace with real data in practice)
x = Nx.random_uniform({num_samples, num_features})
y = Nx.remainder(Nx.iota({num_samples}), num_classes) |> Nx.as_type({:s, 64})

# Split
{train, val} = ExBurn.Dataset.split({x, y}, val_split: 0.2, seed: 42)
{train_x, train_y} = train
{val_x, val_y} = val

# Normalize
{train_x_norm, stats} = ExBurn.Dataset.normalize(train_x, method: :standard)
val_x_norm = ExBurn.Dataset.normalize_with_stats(val_x, stats)

# ── 2. Define Model ────────────────────────────────────────
model =
  Axon.input("features", shape: {nil, num_features})
  |> Axon.dense(32, activation: :relu, name: "hidden1")
  |> Axon.dropout(rate: 0.2)
  |> Axon.dense(16, activation: :relu, name: "hidden2")
  |> Axon.dropout(rate: 0.2)
  |> Axon.dense(num_classes, name: "output")

# ── 3. Compile ─────────────────────────────────────────────
compiled = ExBurn.Model.compile(model,
  loss: :cross_entropy,
  optimizer: :adam,
  learning_rate: 0.001,
  weight_decay: 1.0e-4
)

IO.puts(ExBurn.Model.summary(compiled))

# ── 4. Train ───────────────────────────────────────────────
trained = ExBurn.Training.fit(compiled,
  {train_x_norm, train_y},
  epochs: 50,
  batch_size: 16,
  shuffle: true,
  validation_data: {val_x_norm, val_y},
  lr_schedule: {:cosine, 0.001, 1.0e-5},
  clip_norm: 1.0,
  accuracy: true,
  callbacks: [
    &ExBurn.Training.LoggingCallback.log/1,
    ExBurn.Training.EarlyStoppingCallback.wait(10, 1.0e-5),
    ExBurn.Training.HistoryCallback.new()
  ],
  verbose: true
)

# ── 5. Evaluate ────────────────────────────────────────────
{loss, accuracy} = ExBurn.Training.evaluate(trained, {val_x_norm, val_y}, true)
IO.puts("Final — loss: #{Float.round(loss, 4)}, accuracy: #{Float.round(accuracy * 100, 1)}%")

# ── 6. Inference ──────────────────────────────────────────
new_sample = Nx.tensor([[5.1, 3.5, 1.4, 0.2]])
new_sample_norm = ExBurn.Dataset.normalize_with_stats(new_sample, stats)
{:ok, prediction} = ExBurn.Model.predict(trained, new_sample_norm)
predicted_class = Nx.argmax(prediction) |> Nx.to_number()
IO.puts("Predicted class: #{predicted_class}")

# ── 7. Save ───────────────────────────────────────────────
ExBurn.Model.save(trained, "iris_model.bin")
IO.puts("Model saved!")
```

### Training Checklist

Use this checklist for every training run:

- [ ] **Data split**: Train/val/test split with a fixed seed
- [ ] **Normalization**: Fit on training data, transform all splits
- [ ] **Model architecture**: Appropriate depth/width for the problem
- [ ] **Loss function**: Matches the task (classification vs regression)
- [ ] ] **Optimizer**: Start with Adam, lr=0.001
- [ ] **Regularization**: Dropout + weight decay to prevent overfitting
- [ ] **Early stopping**: Stop when validation loss plateaus
- [ ] **Gradient clipping**: Enable if you see NaN loss
- [ ] **Learning rate schedule**: Cosine annealing for best results
- [ ] **Evaluation**: Check both loss and accuracy on validation set

### Common Problems and Solutions

| Problem | Likely Cause | Solution |
|---|---|---|
| Loss is NaN | Exploding gradients | Enable `clip_norm: 1.0`, reduce learning rate |
| Loss doesn't decrease | LR too low, wrong loss | Increase LR, check loss function |
| Loss oscillates | LR too high, batch too small | Reduce LR, increase batch size or use `accumulate_gradients` |
| Overfitting | Model too complex | Add dropout, weight decay, early stopping |
| Training very slow | Large model with numerical gradients | Use `grad_method: :numerical_batch`, reduce model size |

### Next Steps

- [Training Models](02_training.md) — Full API reference for training
- [Training Optimization Guide](05_training_optimization.md) — Advanced tuning techniques
- [Mobile Deployment](03_mobile_deployment.md) — Deploy to iOS/Android
- [Architecture Deep-Dive](04_architecture.md) — How ExBurn works internally
