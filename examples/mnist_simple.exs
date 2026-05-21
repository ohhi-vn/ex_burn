# Simple MNIST-like Classifier with ExBurn
# Run with: mix run examples/mnist_simple.exs
#
# Demonstrates a complete deep learning workflow:
#   1. Generate synthetic data (simulating 28x28 images → 10 classes)
#   2. Define an MLP: 784 → 128 → 64 → 10
#   3. Train with Adam optimizer
#   4. Evaluate accuracy
#   5. Save and reload the model

Mix.install([
  {:nx, "~> 0.7"},
  {:axon, "~> 0.7"},
  {:ex_burn, path: Path.expand("..", __DIR__)}
])

defmodule MNISTSimple do
  @moduledoc """
  MNIST-like classifier using synthetic data.
  Architecture: 784 → 128 (relu) → 64 (relu) → 10 (softmax)
  """

  @input_dim 784
  @num_classes 10
  @num_train 500
  @num_test 100

  def run do
    IO.puts("=== MNIST-like Classifier with ExBurn ===\n")

    # ── 1. Generate synthetic data ──────────────────────────────
    IO.puts("Generating synthetic data...")

    {train_x, train_y} = generate_data(@num_train)
    {test_x, test_y} = generate_data(@num_test)

    IO.puts("  Train: #{@num_train} samples")
    IO.puts("  Test:  #{@num_test} samples")
    IO.puts("  Input: #{@input_dim} features")
    IO.puts("  Classes: #{@num_classes}\n")

    # ── 2. Define model ─────────────────────────────────────────
    model =
      Axon.input("input", shape: {nil, @input_dim})
      |> Axon.dense(128, activation: :relu, name: "hidden1")
      |> Axon.dropout(rate: 0.2)
      |> Axon.dense(64, activation: :relu, name: "hidden2")
      |> Axon.dropout(rate: 0.2)
      |> Axon.dense(@num_classes, name: "output")

    IO.puts("Model architecture:")
    IO.puts(Axon.Display.display(model, []))

    # ── 3. Compile ──────────────────────────────────────────────
    compiled = ExBurn.Model.compile(model,
      loss: :cross_entropy,
      optimizer: :adam,
      learning_rate: 0.001
    )

    IO.puts("Parameters: #{format_param_count(compiled.params)}\n")

    # ── 4. Train ────────────────────────────────────────────────
    IO.puts("Training...")

    trained =
      ExBurn.Training.fit(compiled, {train_x, train_y},
        epochs: 20,
        batch_size: 32,
        verbose: false,
        callbacks: [
          fn
            %{epoch: epoch, loss: loss} when rem(epoch, 5) == 0 ->
              acc = compute_accuracy(trained, test_x, test_y)
              IO.puts("  Epoch #{String.pad_leading("#{epoch}", 2)}: loss=#{Float.round(loss, 4)} test_acc=#{Float.round(acc * 100, 1)}%")
              %{epoch: epoch, loss: loss}

            %{epoch: epoch, loss: loss} ->
              IO.puts("  Epoch #{String.pad_leading("#{epoch}", 2)}: loss=#{Float.round(loss, 4)}")
              %{epoch: epoch, loss: loss}
          end
        ]
      )

    # ── 5. Evaluate ─────────────────────────────────────────────
    IO.puts("\nFinal Evaluation:")
    train_acc = compute_accuracy(trained, train_x, train_y)
    test_acc = compute_accuracy(trained, test_x, test_y)
    IO.puts("  Train accuracy: #{Float.round(train_acc * 100, 1)}%")
    IO.puts("  Test accuracy:  #{Float.round(test_acc * 100, 1)}%")

    # ── 6. Save and reload ──────────────────────────────────────
    model_path = "/tmp/mnist_simple.model"
    IO.puts("\nSaving model to #{model_path}...")
    ExBurn.Model.save(trained, model_path)
    {:ok, reloaded} = ExBurn.Model.load(trained, model_path)

    reload_acc = compute_accuracy(reloaded, test_x, test_y)
    IO.puts("  Reloaded model accuracy: #{Float.round(reload_acc * 100, 1)}%")

    # ── 7. Single prediction ────────────────────────────────────
    IO.puts("\nSample prediction:")
    sample = Nx.slice(test_x, [0, 0], [1, @input_dim])
    pred = forward_pass(reloaded, sample)
    pred_class = Nx.argmax(pred) |> Nx.to_number()
    actual_class = Nx.argmax(Nx.slice(test_y, [0, 0], [1, @num_classes])) |> Nx.to_number()
    probs = Nx.softmax(pred) |> Nx.to_flat_list() |> Enum.map(&Float.round(&1, 3))
    IO.puts("  Predicted: #{pred_class} | Actual: #{actual_class}")
    IO.puts("  Probabilities: #{inspect(probs)}")

    IO.puts("\n=== Done ===")
  end

  # ── Data Generation ────────────────────────────────────────────

  defp generate_data(n) do
    key = Nx.Random.key(:erlang.phash2(n))

    # Generate random "images" with class-dependent patterns
    {x, key} = Nx.Random.normal(key, 0.0, 1.0, shape: {n, @input_dim})

    # Create one-hot labels
    {labels, _key} = Nx.Random.uniform(key, 0, @num_classes - 0.001, shape: {n})
    labels = Nx.as_type(labels, {:s, 64})
    y = Nx.equal(Nx.iota({n, @num_classes}, axis: 1), Nx.new_axis(labels, -1))
    y = Nx.as_type(y, :f32)

    # Add class-specific signal to features
    signal = Nx.multiply(y, 0.5)
    signal = Nx.slice(signal, [0, 0], [n, @input_dim])
    x = Nx.add(x, signal)

    {x, y}
  end

  # ── Forward Pass ───────────────────────────────────────────────

  defp forward_pass(%ExBurn.Model{params: params}, input) do
    # Layer 1: hidden1
    h = relu(Nx.add(Nx.dot(input, params["hidden1"]["weight"]), params["hidden1"]["bias"]))

    # Layer 2: hidden2
    h = relu(Nx.add(Nx.dot(h, params["hidden2"]["weight"]), params["hidden2"]["bias"]))

    # Output layer
    Nx.add(Nx.dot(h, params["output"]["weight"]), params["output"]["bias"])
  end

  defp relu(tensor), do: Nx.max(tensor, 0.0)

  # ── Accuracy ───────────────────────────────────────────────────

  defp compute_accuracy(model, x, y) do
    preds = forward_pass(model, x)
    pred_classes = Nx.argmax(preds, axis: 1)
    true_classes = Nx.argmax(y, axis: 1)
    correct = Nx.equal(pred_classes, true_classes) |> Nx.as_type(:f32)
    Nx.mean(correct) |> Nx.to_number()
  end

  # ── Helpers ────────────────────────────────────────────────────

  defp format_param_count(params) do
    count =
      Enum.reduce(params, 0, fn {_name, layer_params}, acc ->
        Enum.reduce(layer_params, acc, fn {_param_name, tensor}, sum ->
          sum + Nx.size(tensor)
        end)
      end)

    if count > 1_000_000 do
      "#{Float.round(count / 1_000_000, 1)}M"
    else
      "#{div(count, 1_000)}K"
    end
  end
end

MNISTSimple.run()
