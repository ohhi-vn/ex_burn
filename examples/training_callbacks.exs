# Training Callbacks with ExBurn
# Run with: mix run examples/training_callbacks.exs
#
# Demonstrates the ExBurn.Training callback system:
#   1. LoggingCallback — log metrics after each epoch
#   2. EarlyStoppingCallback — stop when validation loss plateaus
#   3. CheckpointCallback — save model checkpoints at intervals
#   4. WarmupCallback — gradually increase learning rate
#   5. ReduceLROnPlateauCallback — reduce LR when loss stops improving
#   6. HistoryCallback — record all metrics for later analysis
#   7. Custom callbacks — writing your own

Mix.install([
  {:nx, "~> 0.7"},
  {:axon, "~> 0.7"},
  {:ex_burn, path: Path.expand("..", __DIR__)}
])

defmodule TrainingCallbacks do
  @moduledoc """
  Showcases ExBurn.Training callbacks for customizing training.
  """

  def run do
    IO.puts("=== Training Callbacks with ExBurn ===\n")

    # ── 1. Create synthetic classification data ────────────────
    {train_x, train_y, val_x, val_y} = generate_data(120, 30)

    IO.puts("Dataset:")

    IO.puts(
      "  Train: #{Nx.shape(train_x) |> elem(0)} samples, #{Nx.shape(train_x) |> elem(1)} features"
    )

    IO.puts("  Val:   #{Nx.shape(val_x) |> elem(0)} samples")
    IO.puts("  Classes: 3\n")

    # ── 2. Define model ────────────────────────────────────────
    # Keep the network small: gradients are computed numerically
    # (2 forward passes per parameter per batch), so large models are slow.
    model =
      Axon.input("input", shape: {nil, 10})
      |> Axon.dense(8, activation: :relu, name: "hidden1")
      |> Axon.dense(3, name: "output")

    compiled =
      ExBurn.Model.compile(model,
        loss: :cross_entropy,
        optimizer: :adam,
        learning_rate: 0.001
      )

    # ── 3. Train with all callbacks ────────────────────────────
    IO.puts("Training with callbacks...\n")

    # Simple in-example recorder so we can print the history afterwards
    {:ok, history_agent} = Agent.start_link(fn -> [] end)

    trained =
      ExBurn.Training.fit(compiled, {train_x, train_y},
        epochs: 15,
        batch_size: 16,
        validation_data: {val_x, val_y},
        verbose: false,
        lr_schedule: {:cosine, 0.001, 1.0e-5},
        clip_norm: 1.0,
        weight_decay: 1.0e-4,
        accuracy: true,
        callbacks: [
          # Log every epoch
          &ExBurn.Training.LoggingCallback.log/1,

          # Stop early if no improvement for 10 epochs
          ExBurn.Training.EarlyStoppingCallback.wait(10, 1.0e-4),

          # Save checkpoint every 10 epochs
          ExBurn.Training.CheckpointCallback.every(10, "/tmp/ex_burn_checkpoints"),

          # Warmup: lr from 1e-5 to 0.001 over 5 epochs
          ExBurn.Training.WarmupCallback.linear(5, 1.0e-5, 0.001),

          # Reduce LR on plateau
          ExBurn.Training.ReduceLROnPlateauCallback.new(patience: 5, factor: 0.5, min_lr: 1.0e-6),

          # Record history
          fn metrics ->
            Agent.update(history_agent, &[metrics | &1])
            metrics
          end,

          # Custom callback: print a message at epoch 25
          fn
            %{epoch: 25} = metrics ->
              IO.puts("  [Custom] Reached epoch 25! loss=#{Float.round(metrics.loss, 4)}")
              metrics

            metrics ->
              metrics
          end
        ]
      )

    # ── 4. Review history ──────────────────────────────────────
    IO.puts("\nTraining history (last 5 epochs):")

    history = Agent.get(history_agent, &Enum.reverse/1)

    history
    |> Enum.take(5)
    |> Enum.each(fn m ->
      epoch = m.epoch
      loss = Float.round(m.loss, 4)
      val_loss = if m.val_loss, do: Float.round(m.val_loss, 4), else: "N/A"
      acc = if m[:accuracy], do: "#{Float.round(m.accuracy * 100, 1)}%", else: "N/A"

      IO.puts(
        "  Epoch #{String.pad_leading("#{epoch}", 2)}: loss=#{loss} val_loss=#{val_loss} acc=#{acc}"
      )
    end)

    # ── 5. Final evaluation ────────────────────────────────────
    IO.puts("\nFinal evaluation:")
    {train_loss, train_acc} = ExBurn.Training.evaluate(trained, {train_x, train_y}, true)
    {val_loss, val_acc} = ExBurn.Training.evaluate(trained, {val_x, val_y}, true)
    IO.puts("  Train: loss=#{Float.round(train_loss, 4)} acc=#{Float.round(train_acc * 100, 1)}%")
    IO.puts("  Val:   loss=#{Float.round(val_loss, 4)} acc=#{Float.round(val_acc * 100, 1)}%")

    # ── 6. Profile a training step ─────────────────────────────
    IO.puts("\nProfile single training step:")
    batch_x = Nx.slice(train_x, [0, 0], [32, 10])
    batch_y = Nx.slice(train_y, [0, 0], [32, 3])
    profile = ExBurn.Training.profile_step(trained, {batch_x, batch_y})
    IO.puts("  Forward:   #{profile.forward_ms} ms")
    IO.puts("  Backward:  #{profile.backward_ms} ms")
    IO.puts("  Optimizer: #{profile.optimizer_ms} ms")
    IO.puts("  Total:     #{profile.total_ms} ms")

    IO.puts("\n=== Done ===")
  end

  # ── Data Generation ──────────────────────────────────────────

  defp generate_data(n_train, n_val) do
    key = Nx.Random.key(42)
    input_dim = 10
    num_classes = 3

    # Training data
    {train_x, key} = Nx.Random.normal(key, 0.0, 1.0, shape: {n_train, input_dim})
    {train_labels, key} = Nx.Random.uniform(key, 0, num_classes - 0.001, shape: {n_train})
    train_labels = Nx.as_type(train_labels, {:s, 64})
    train_y = ExBurn.Dataset.one_hot(train_labels, num_classes: num_classes)

    # Add class-specific signal: each class shifts inputs by its own vector,
    # so the classes are linearly separable.
    {class_vectors, key} = Nx.Random.normal(key, 0.0, 0.5, shape: {num_classes, input_dim})

    train_x = Nx.add(train_x, Nx.take(class_vectors, train_labels))

    # Validation data
    {val_x, _key} = Nx.Random.normal(key, 0.0, 1.0, shape: {n_val, input_dim})
    {val_labels, _key} = Nx.Random.uniform(key, 0, num_classes - 0.001, shape: {n_val})
    val_labels = Nx.as_type(val_labels, {:s, 64})
    val_y = ExBurn.Dataset.one_hot(val_labels, num_classes: num_classes)
    val_x = Nx.add(val_x, Nx.take(class_vectors, val_labels))

    {train_x, train_y, val_x, val_y}
  end

end

TrainingCallbacks.run()
