# XOR Classifier with ExBurn
# Run with: mix run examples/xor_classifier.exs
#
# Demonstrates training a neural network on the classic XOR problem.
# XOR is not linearly separable, so a hidden layer is required.
# This example shows:
#   1. A minimal non-linear classification problem
#   2. Using Model.summary() to inspect architecture
#   3. Training with validation data and early stopping
#   4. Using the Dataset module for splitting

Mix.install([
  {:nx, "~> 0.7"},
  {:axon, "~> 0.7"},
  {:ex_burn, path: Path.expand("..", __DIR__)}
])

defmodule XORClassifier do
  @moduledoc """
  XOR classification: learns the non-linear XOR function.
  Input: 2D binary vectors -> Output: XOR of the two bits.
  """

  def run do
    IO.puts("=== XOR Classifier with ExBurn ===\n")

    # 1. Create XOR dataset
    x = Nx.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], type: {:f, 32})
    y = Nx.tensor([[0], [1], [1], [0]], type: {:f, 32})

    # Augment with noisy copies so the model has enough data to learn
    {x_aug, y_aug} = augment_data(x, y, 50)

    IO.puts("Dataset: #{Nx.shape(x_aug) |> elem(0)} samples (4 unique + augmented)")
    IO.puts("  Input shape:  #{inspect(Nx.shape(x_aug))}")
    IO.puts("  Target shape: #{inspect(Nx.shape(y_aug))}")

    # 2. Split into train / validation
    {train, val} = ExBurn.Dataset.split({x_aug, y_aug}, val_split: 0.2, seed: 42)
    {train_x, train_y} = train
    {val_x, val_y} = val

    IO.puts("  Train: #{Nx.shape(train_x) |> elem(0)} samples")
    IO.puts("  Val:   #{Nx.shape(val_x) |> elem(0)} samples\n")

    # 3. Define model: 2 -> 8 (relu) -> 1 (sigmoid)
    model =
      Axon.input("input", shape: {nil, 2})
      |> Axon.dense(8, activation: :relu, name: "hidden")
      |> Axon.dense(1, activation: :sigmoid, name: "output")

    IO.puts("Model architecture: 2 -> 8 (relu) -> 1 (sigmoid)")

    # 4. Compile
    compiled =
      ExBurn.Model.compile(model,
        loss: :binary_cross_entropy,
        optimizer: :adam,
        learning_rate: 0.1
      )

    IO.puts(ExBurn.Model.summary(compiled))

    # 5. Train with early stopping
    IO.puts("Training...\n")

    trained =
      ExBurn.Training.fit(compiled, {train_x, train_y},
        epochs: 200,
        batch_size: 32,
        validation_data: {val_x, val_y},
        verbose: false,
        callbacks: [
          ExBurn.Training.EarlyStoppingCallback.wait(20, 1.0e-5),
          fn
            %{epoch: epoch, loss: loss, val_loss: val_loss} when rem(epoch, 50) == 0 ->
              IO.puts(
                "  Epoch #{String.pad_leading("#{epoch}", 3)}: loss=#{Float.round(loss, 5)} val_loss=#{Float.round(val_loss, 5)}"
              )

              %{epoch: epoch, loss: loss, val_loss: val_loss}

            metrics ->
              metrics
          end
        ]
      )

    # 6. Evaluate on all 4 XOR inputs
    IO.puts("\nPredictions on XOR truth table:")

    IO.puts(
      "  (0,0) -> #{predict(trained, Nx.tensor([[0, 0]], type: {:f, 32})) |> Nx.squeeze() |> Nx.to_number() |> round_pred()}"
    )

    IO.puts(
      "  (0,1) -> #{predict(trained, Nx.tensor([[0, 1]], type: {:f, 32})) |> Nx.squeeze() |> Nx.to_number() |> round_pred()}"
    )

    IO.puts(
      "  (1,0) -> #{predict(trained, Nx.tensor([[1, 0]], type: {:f, 32})) |> Nx.squeeze() |> Nx.to_number() |> round_pred()}"
    )

    IO.puts(
      "  (1,1) -> #{predict(trained, Nx.tensor([[1, 1]], type: {:f, 32})) |> Nx.squeeze() |> Nx.to_number() |> round_pred()}"
    )

    # Compute accuracy on validation set
    val_preds = predict_batch(trained, val_x)
    val_acc = compute_accuracy(val_preds, val_y)
    IO.puts("\nValidation accuracy: #{Float.round(val_acc * 100, 1)}%")

    IO.puts("\n=== Done ===")
  end

  defp predict(model, input) do
    {:ok, output} = ExBurn.Model.predict(model, input)
    output
  end

  defp predict_batch(model, input) do
    {:ok, output} = ExBurn.Model.predict(model, input)
    output
  end

  defp round_pred(val) when val >= 0.5, do: 1
  defp round_pred(_val), do: 0

  defp compute_accuracy(preds, targets) do
    pred_classes = Nx.greater_equal(preds, 0.5) |> Nx.as_type(:f32)
    correct = Nx.equal(pred_classes, targets) |> Nx.as_type(:f32)
    Nx.mean(correct) |> Nx.to_number()
  end

  defp augment_data(x, y, copies) do
    key = Nx.Random.key(123)

    num_rows = Nx.shape(x) |> elem(0)

    {noise, _key} =
      Nx.Random.normal(key, 0.0, 0.05, shape: {num_rows * copies, 2})

    x_aug = Nx.tile(x, [copies, 1]) |> Nx.add(noise)
    x_aug = Nx.clip(x_aug, 0.0, 1.0)
    y_aug = Nx.tile(y, [copies, 1])
    {x_aug, y_aug}
  end
end

XORClassifier.run()
