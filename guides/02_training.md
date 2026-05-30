# Training Models with ExBurn

## Overview

ExBurn provides a complete training pipeline: define a model with Axon, compile it with `ExBurn.Model.compile/2`, and train it with `ExBurn.Training.fit/3`. The training loop supports multiple optimizers, learning rate schedules, gradient clipping, weight decay, and callbacks.

## Defining a Model with Axon

```elixir
model =
  Axon.input("input", shape: {nil, 784})
  |> Axon.dense(256, activation: :relu, name: "hidden1")
  |> Axon.dropout(rate: 0.2)
  |> Axon.dense(128, activation: :relu, name: "hidden2")
  |> Axon.dropout(rate: 0.2)
  |> Axon.dense(10, name: "output")
```

The `nil` in the shape represents the batch dimension (variable size).

## Compiling a Model

```elixir
compiled = ExBurn.Model.compile(model,
  loss: :cross_entropy,       # :cross_entropy | :mse | :binary_cross_entropy
  optimizer: :adam,           # :adam | :sgd | :rmsprop
  learning_rate: 0.001,
  device: :gpu,               # :gpu | :cpu
  weight_decay: 1.0e-4        # L2 regularization (default: 0.0)
)
```

### What `compile/2` Does

1. Builds the Axon expression graph via `Axon.build/2`
2. Initializes parameters with **Glorot/Xavier uniform initialization** for weights, zeros for biases
3. Optionally moves parameters to GPU via `BurnBridge.to_gpu/1`
4. Initializes optimizer state (momentum buffers for Adam, velocity for SGD, etc.)
5. Returns an `ExBurn.Model` struct ready for training

### Inspecting a Model

```elixir
# Keras/PyTorch-style summary
IO.puts(ExBurn.Model.summary(compiled))
# ╔══════════════════════════════════════════════════════════╗
# ║                   ExBurn Model Summary                  ║
# ╠══════════════════════════════════════════════════════════╣
# ║  Layer                 Type         Output Shape        ║
# ║  hidden1               Dense        [nil, 256]          ║
# ║  dropout_1             Dropout      [nil, 256]          ║
# ║  hidden2               Dense        [nil, 128]          ║
# ║  output                Dense        [nil, 10]           ║
# ╠══════════════════════════════════════════════════════════╣
# ║  Total params:     235,146                               ║
# ║  Trainable params: 235,146                               ║
# ╚══════════════════════════════════════════════════════════╝

# Access components
ExBurn.Model.parameters(compiled)     # parameter map
ExBurn.Model.loss_function(compiled)  # :cross_entropy
ExBurn.Model.optimizer(compiled)      # :adam
ExBurn.Model.weight_decay(compiled)   # 0.0001
```

## Training

```elixir
trained = ExBurn.Training.fit(compiled, {train_x, train_y},
  epochs: 10,
  batch_size: 32,
  shuffle: true,
  validation_data: {val_x, val_y},
  verbose: true
)
```

### Training Options

| Option | Type | Default | Description |
|---|---|---|---|
| `:epochs` | `pos_integer()` | `10` | Number of training epochs |
| `:batch_size` | `pos_integer()` | `32` | Mini-batch size |
| `:shuffle` | `boolean()` | `true` | Shuffle training data each epoch |
| `:validation_data` | `{tensor, tensor}` | `nil` | Validation dataset |
| `:callbacks` | `[function()]` | `[]` | Callback functions called after each epoch |
| `:verbose` | `boolean()` | `true` | Print training progress |
| `:lr_schedule` | see below | `nil` | Learning rate schedule |
| `:clip_norm` | `float()` | `nil` | Max gradient norm for clipping |
| `:clip_value` | `float()` | `nil` | Max absolute gradient value |
| `:weight_decay` | `float()` | `nil` | L2 regularization coefficient |
| `:accumulate_gradients` | `pos_integer()` | `1` | Accumulate N batches before optimizer step |
| `:accuracy` | `boolean()` | `false` | Compute classification accuracy |
| `:nesterov` | `boolean()` | `false` | Nesterov momentum (SGD only) |

### Learning Rate Schedules

```elixir
# Step decay: multiply LR by gamma every step_size epochs
lr_schedule: {:step, 0.001, 10, 0.5}

# Exponential decay: LR = base_lr * gamma^epoch
lr_schedule: {:exponential, 0.001, 0.95}

# Cosine annealing: smoothly decay from base_lr to min_lr
lr_schedule: {:cosine, 0.001, 1.0e-5}
```

### Gradient Clipping

```elixir
# Clip by global norm (prevents exploding gradients)
clip_norm: 1.0

# Clip by absolute value
clip_value: 5.0

# Both can be used together
```

### Gradient Accumulation

Effective when GPU memory limits batch size. Accumulates gradients across N mini-batches before performing one optimizer step:

```elixir
# Effective batch_size = 32 * 4 = 128
ExBurn.Training.fit(model, data,
  batch_size: 32,
  accumulate_gradients: 4
)
```

## Optimizers

### Adam (default)

Adaptive learning rate with momentum. Good default for most tasks.

```elixir
ExBurn.Model.compile(model, optimizer: :adam, learning_rate: 0.001)
# Internal state: m (1st moment), v (2nd moment), t (timestep)
# beta1=0.9, beta2=0.999, epsilon=1e-8
```

### SGD with Momentum

```elixir
ExBurn.Model.compile(model, optimizer: :sgd, learning_rate: 0.01)
# momentum=0.9
```

With Nesterov momentum (often converges faster):

```elixir
ExBurn.Training.fit(model, data, nesterov: true)
```

### RMSprop

Good for recurrent networks and non-stationary objectives:

```elixir
ExBurn.Model.compile(model, optimizer: :rmsprop, learning_rate: 0.001)
# decay=0.9, epsilon=1e-8
```

## Callbacks

Callbacks are functions that receive a metrics map after each epoch and return it (possibly modified).

### Built-in Callbacks

```elixir
# Logging
callbacks: [&ExBurn.Training.LoggingCallback.log/1]

# Early stopping (patience=5 epochs, min_delta=1e-4)
callbacks: [ExBurn.Training.EarlyStoppingCallback.wait(5, 1.0e-4)]

# Checkpoint every 5 epochs
callbacks: [ExBurn.Training.CheckpointCallback.every(5, "/checkpoints")]
```

### Custom Callbacks

The metrics map has this structure:

```elixir
%{
  epoch: 5,
  loss: 0.0234,
  val_loss: 0.0312,       # if validation_data provided
  accuracy: 0.98,          # if accuracy: true
  val_accuracy: 0.95,      # if validation_data + accuracy
  model: %ExBurn.Model{}   # current model state
}
```

Return `Map.put(metrics, :stop_training, true)` to halt training early:

```elixir
custom_callback = fn
  %{loss: loss} when loss < 0.01 ->
    IO.puts("Converged!")
    %{epoch: epoch, loss: loss, stop_training: true}

  metrics ->
    metrics
end
```

## Evaluating a Model

```elixir
# Returns average loss
loss = ExBurn.Training.evaluate(model, {test_x, test_y})

# Returns {loss, accuracy} tuple
{loss, accuracy} = ExBurn.Training.evaluate(model, {test_x, test_y}, true)
```

## Inference

```elixir
# Using the model's forward pass (GPU via defn compiler)
{:ok, output} = ExBurn.Model.forward(compiled, input_tensor)

# Using Axon predict (CPU via BinaryBackend)
{:ok, output} = ExBurn.Model.predict(compiled, input_tensor)
```

## Saving and Loading

```elixir
# Save (compressed Erlang term format)
ExBurn.Model.save(trained, "model.bin")

# Load
{:ok, model} = ExBurn.Model.load(trained, "model.bin")

# Serialize to binary (for network transfer)
binary = ExBurn.Model.serialize_params(trained)
{:ok, params} = ExBurn.Model.deserialize_params(binary)
```

## Freezing Layers (Fine-tuning)

Freeze layers to prevent them from updating during training:

```elixir
# Freeze specific layers
model = ExBurn.Model.freeze(compiled, ["hidden1"])

# Check if a layer is frozen
ExBurn.Model.frozen?(model, "hidden1")  # true

# Unfreeze
model = ExBurn.Model.unfreeze(model, ["hidden1"])

# Get all frozen layer names
ExBurn.Model.frozen_layers(model)  # #MapSet<["hidden1"]>
```

## Device Management

```elixir
# Move model to GPU
gpu_model = ExBurn.Model.to_device(compiled, :gpu)

# Move model to CPU
cpu_model = ExBurn.Model.to_device(compiled, :cpu)

# No-op if already on target device
same_model = ExBurn.Model.to_device(cpu_model, :cpu)
```

## Custom Training Loops

For full control, use `train_step/3` directly:

```elixir
{loss, updated_model} = ExBurn.Training.train_step(model, {batch_x, batch_y},
  clip_norm: 1.0
)
```

Compute gradients separately:

```elixir
grads = ExBurn.Training.compute_gradients(model, {batch_x, batch_y},
  grad_method: :numerical  # or :numerical_batch
)
```

## Loss Functions

| Loss | Use Case | Target Format |
|---|---|---|
| `:cross_entropy` | Multi-class classification | One-hot or integer class indices |
| `:mse` | Regression | Continuous values |
| `:binary_cross_entropy` | Binary classification | 0.0 or 1.0 |
