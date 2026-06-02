# Benchmark: Neural Network Training
# Run with: mix run bench/training_bench.exs
#
# Benchmarks end-to-end training performance for different model
# architectures and batch sizes. Compares different optimizers
# and tracks loss convergence.

Mix.install([
  {:nx, "~> 0.7"},
  {:axon, "~> 0.7"},
  {:ex_burn, path: Path.expand("..", __DIR__)}
])

defmodule TrainingBench do
  def run do
    IO.puts("=== Neural Network Training Benchmark ===\n")

    # 1. Small MLP
    IO.puts("--- Small MLP (10->32->16->3) ---")

    benchmark_training(
      build_mlp(10, [32, 16], 3),
      generate_classification_data(500, 10, 3, 42),
      epochs: 20,
      batch_size: 32
    )

    # 2. Medium MLP
    IO.puts("\n--- Medium MLP (50->128->64->10) ---")

    benchmark_training(
      build_mlp(50, [128, 64], 10),
      generate_classification_data(1000, 50, 10, 42),
      epochs: 15,
      batch_size: 64
    )

    # 3. Optimizer comparison
    IO.puts("\n--- Optimizer Comparison (10->64->32->5) ---")
    {train_x, train_y, val_x, val_y} = generate_classification_data(500, 10, 5, 42)
    model = build_mlp(10, [64, 32], 5)

    for opt <- [:adam, :sgd, :rmsprop] do
      compiled =
        ExBurn.Model.compile(model, loss: :cross_entropy, optimizer: opt, learning_rate: 0.001)

      {time_us, trained} =
        :timer.tc(fn ->
          ExBurn.Training.fit(compiled, {train_x, train_y},
            epochs: 15,
            batch_size: 32,
            validation_data: {val_x, val_y},
            verbose: false
          )
        end)

      {final_loss, _} = ExBurn.Training.evaluate(trained, {val_x, val_y}, false)

      IO.puts(
        "  #{pad("#{opt}", 8)} time=#{Float.round(time_us / 1000, 0)}ms  final_val_loss=#{Float.round(final_loss, 4)}"
      )
    end

    # 4. Batch size comparison
    IO.puts("\n--- Batch Size Comparison (10->64->32->5) ---")

    compiled =
      ExBurn.Model.compile(model, loss: :cross_entropy, optimizer: :adam, learning_rate: 0.001)

    for bs <- [16, 32, 64, 128] do
      {time_us, _trained} =
        :timer.tc(fn ->
          ExBurn.Training.fit(compiled, {train_x, train_y},
            epochs: 10,
            batch_size: bs,
            verbose: false
          )
        end)

      IO.puts("  batch_size=#{pad("#{bs}", 4)} time=#{Float.round(time_us / 1000, 0)}ms")
    end

    IO.puts("\n=== Done ===")
  end

  defp benchmark_training(model, {train_x, train_y, val_x, val_y}, opts) do
    epochs = Keyword.get(opts, :epochs, 10)
    batch_size = Keyword.get(opts, :batch_size, 32)

    compiled =
      ExBurn.Model.compile(model,
        loss: :cross_entropy,
        optimizer: :adam,
        learning_rate: 0.001
      )

    info = ExBurn.Model.info(compiled)
    IO.puts("  Params: #{info.total_params}")

    bench =
      ExBurn.Model.benchmark(compiled, Nx.slice(train_x, [0, 0], [1, elem(Nx.shape(train_x), 1)]),
        warmup: 3,
        runs: 20
      )

    IO.puts("  Forward pass: avg=#{bench.avg_ms}ms min=#{bench.min_ms}ms max=#{bench.max_ms}ms")

    {time_us, trained} =
      :timer.tc(fn ->
        ExBurn.Training.fit(compiled, {train_x, train_y},
          epochs: epochs,
          batch_size: batch_size,
          validation_data: {val_x, val_y},
          verbose: false
        )
      end)

    {val_loss, val_acc} = ExBurn.Training.evaluate(trained, {val_x, val_y}, true)

    IO.puts(
      "  Training: #{epochs} epochs in #{Float.round(time_us / 1000, 0)}ms (#{Float.round(time_us / epochs / 1000, 0)}ms/epoch)"
    )

    IO.puts(
      "  Final: val_loss=#{Float.round(val_loss, 4)} val_acc=#{Float.round(val_acc * 100, 1)}%"
    )
  end

  defp build_mlp(input_dim, hidden_layers, output_dim) do
    model = Axon.input("input", shape: {nil, input_dim})

    model =
      Enum.with_index(hidden_layers, fn units, idx ->
        Axon.dense(units, activation: :relu, name: "hidden_#{idx}")
      end)
      |> Enum.reduce(model, fn dense_fn, acc -> dense_fn.(acc) end)

    Axon.dense(model, output_dim, name: "output")
  end

  defp generate_classification_data(n, input_dim, num_classes, seed) do
    key = Nx.Random.key(seed)

    {train_x, key} = Nx.Random.normal(key, 0.0, 1.0, shape: {n, input_dim})
    {train_labels, key} = Nx.Random.uniform(key, 0, num_classes - 0.001, shape: {n})
    train_labels = Nx.as_type(train_labels, {:s, 64})
    train_y = ExBurn.Dataset.one_hot(train_labels, num_classes: num_classes)

    signal = Nx.multiply(train_y, 0.5) |> Nx.slice([0, 0], [n, input_dim])
    train_x = Nx.add(train_x, signal)

    {val_x, _key} = Nx.Random.normal(key, 0.0, 1.0, shape: {div(n, 5), input_dim})
    {val_labels, _key} = Nx.Random.uniform(key, 0, num_classes - 0.001, shape: {div(n, 5)})
    val_labels = Nx.as_type(val_labels, {:s, 64})
    val_y = ExBurn.Dataset.one_hot(val_labels, num_classes: num_classes)
    val_signal = Nx.multiply(val_y, 0.5) |> Nx.slice([0, 0], [div(n, 5), input_dim])
    val_x = Nx.add(val_x, val_signal)

    {train_x, train_y, val_x, val_y}
  end

  defp pad(str, len) do
    if String.length(str) >= len,
      do: str,
      else: str <> String.duplicate(" ", len - String.length(str))
  end
end

TrainingBench.run()
