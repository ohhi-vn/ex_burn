# Training Models with ExBurn

## Defining a Model with Axon

```elixir
model =
  Axon.input("input", shape: {nil, 784})
  |> Axon.dense(128, activation: :relu, name: "hidden1")
  |> Axon.dropout(rate: 0.2)
  |> Axon.dense(64, activation: :relu, name: "hidden2")
  |> Axon.dense(10, name: "output")
```

## Compiling

```elixir
compiled = ExBurn.Model.compile(model,
  loss: :cross_entropy,     # or :mse, :binary_cross_entropy
  optimizer: :adam,         # or :sgd, :rmsprop
  learning_rate: 0.001
)
```

## Training

```elixir
trained = ExBurn.Training.fit(compiled, {train_x, train_y},
  epochs: 10,
  batch_size: 32,
  validation_data: {val_x, val_y},
  verbose: true
)
```

## Callbacks

### Logging

```elixir
ExBurn.Training.fit(model, data,
  callbacks: [&ExBurn.Training.LoggingCallback.log/1]
)
```

### Early Stopping

```elixir
ExBurn.Training.fit(model, data,
  callbacks: [ExBurn.Training.EarlyStoppingCallback.wait(5, 1.0e-4)]
)
```

### Checkpointing

```elixir
ExBurn.Training.fit(model, data,
  callbacks: [ExBurn.Training.CheckpointCallback.every(5, "/checkpoints")]
)
```

## Custom Callbacks

```elixir
custom_callback = fn
  %{epoch: epoch, loss: loss} when loss < 0.01 ->
    IO.puts("Converged at epoch #{epoch}!")
    %{epoch: epoch, loss: loss, stop_training: true}

  metrics ->
    metrics
end
```

## Saving and Loading

```elixir
# Save
ExBurn.Model.save(trained, "model.bin")

# Load
{:ok, model} = ExBurn.Model.load(trained, "model.bin")
```

## Inference

```elixir
# Single prediction
output = Axon.predict(model, params, input)

# Batch prediction
outputs = Axon.predict(model, params, batch_input)
```
