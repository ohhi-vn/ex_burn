# Mobile Inference with ExBurn + Dala
# Run with: mix run examples/mobile_inference.exs
#
# Demonstrates the mobile deployment pipeline:
#   1. Define a small model suitable for mobile
#   2. Train on synthetic data
#   3. Compile for iOS (Metal) and Android (Vulkan)
#   4. Run inference
#   5. Export model for deployment
#   6. Benchmark inference speed

Mix.install([
  {:nx, "~> 0.7"},
  {:axon, "~> 0.7"},
  {:ex_burn, path: Path.expand("..", __DIR__)}
])

defmodule MobileInference do
  @moduledoc """
  Mobile-optimized image classifier.
  Architecture: 64 → 32 → 16 → 4 (small enough for mobile)
  """

  @input_dim 64
  @num_classes 4
  @model_name "mobile_classifier"

  def run do
    IO.puts("=== Mobile Inference with ExBurn + Dala ===\n")

    # ── 1. Define a mobile-friendly model ────────────────────────
    model =
      Axon.input("input", shape: {nil, @input_dim})
      |> Axon.dense(32, activation: :relu, name: "fc1")
      |> Axon.dense(16, activation: :relu, name: "fc2")
      |> Axon.dense(@num_classes, name: "output")

    IO.puts("Model: #{@input_dim} → 32 → 16 → #{@num_classes}")
    IO.puts(Axon.Display.display(model, []))

    # ── 2. Train ─────────────────────────────────────────────────
    IO.puts("Training on synthetic data...")

    {train_x, train_y} = generate_data(300)
    {test_x, test_y} = generate_data(50)

    compiled = ExBurn.Model.compile(model,
      loss: :cross_entropy,
      optimizer: :adam,
      learning_rate: 0.005
    )

    trained =
      ExBurn.Training.fit(compiled, {train_x, train_y},
        epochs: 15,
        batch_size: 16,
        verbose: false,
        callbacks: [
          fn
            %{epoch: epoch, loss: loss} when rem(epoch, 5) == 0 ->
              IO.puts("  Epoch #{String.pad_leading("#{epoch}", 2)}: loss=#{Float.round(loss, 4)}")
              %{epoch: epoch, loss: loss}

            metrics ->
              metrics
          end
        ]
      )

    acc = compute_accuracy(trained, test_x, test_y)
    IO.puts("  Test accuracy: #{Float.round(acc * 100, 1)}%\n")

    # ── 3. Compile for iOS ───────────────────────────────────────
    IO.puts("── iOS (Metal) ──")

    {:ok, ios_model} = ExBurn.DalaML.compile(model,
      input_shape: {1, @input_dim},
      target: :ios,
      precision: :f16
    )

    IO.puts("  Compiled for iOS with Metal backend")
    IO.puts("  Compute config: #{inspect(ExBurn.DalaML.compute_config(:ios))}")

    # Run inference on iOS model
    sample = Nx.slice(test_x, [0, 0], [1, @input_dim])
    {:ok, ios_output} = ExBurn.DalaML.predict(ios_model, sample)
    IO.puts("  Inference output shape: #{inspect(Nx.shape(ios_output))}")

    # Benchmark
    {:ok, bench} = ExBurn.DalaML.benchmark(ios_model, iterations: 50, warmup: 5)
    IO.puts("  Benchmark: #{Float.round(bench.avg_milliseconds, 2)}ms avg (#{bench.iterations} iterations)")

    # Export
    ios_path = "/tmp/#{@model_name}_ios.bin"
    {:ok, _} = ExBurn.DalaML.export(ios_model, ios_path)
    IO.puts("  Exported to: #{ios_path}")
    IO.puts("  File size: #{File.stat!(ios_path).size} bytes\n")

    # ── 4. Compile for Android ────────────────────────────────────
    IO.puts("── Android (Vulkan) ──")

    {:ok, android_model} = ExBurn.DalaML.compile(model,
      input_shape: {1, @input_dim},
      target: :android,
      precision: :f16
    )

    IO.puts("  Compiled for Android with Vulkan backend")
    IO.puts("  Compute config: #{inspect(ExBurn.DalaML.compute_config(:android))}")

    {:ok, android_output} = ExBurn.DalaML.predict(android_model, sample)
    IO.puts("  Inference output shape: #{inspect(Nx.shape(android_output))}")

    {:ok, bench_android} = ExBurn.DalaML.benchmark(android_model, iterations: 50, warmup: 5)
    IO.puts("  Benchmark: #{Float.round(bench_android.avg_milliseconds, 2)}ms avg")

    android_path = "/tmp/#{@model_name}_android.bin"
    {:ok, _} = ExBurn.DalaML.export(android_model, android_path)
    IO.puts("  Exported to: #{android_path}")
    IO.puts("  File size: #{File.stat!(android_path).size} bytes\n")

    # ── 5. GPU Info ───────────────────────────────────────────────
    IO.puts("── GPU Information ──")
    IO.puts("  GPU available: #{ExBurn.Nif.gpu_available()}")
    IO.puts("  Device name: #{ExBurn.Nif.device_name()}")
    IO.puts("  Available backends: #{inspect(ExBurn.CubeclBridge.available_backends())}")

    caps = ExBurn.CubeclBridge.device_capabilities(nil)
    IO.puts("  Device capabilities:")
    IO.puts("    Max workgroup size: #{caps.max_workgroup_size}")
    IO.puts("    Max shared memory: #{caps.max_shared_memory} bytes")
    IO.puts("    Supports f16: #{caps.supports_f16}")
    IO.puts("    Supports f32: #{caps.supports_f32}")

    IO.puts("\n=== Done ===")
  end

  # ── Data Generation ────────────────────────────────────────────

  defp generate_data(n) do
    key = Nx.Random.key(:erlang.phash2(n))
    {x, key} = Nx.Random.normal(key, 0.0, 1.0, shape: {n, @input_dim})
    {labels, _key} = Nx.Random.uniform(key, 0, @num_classes - 0.001, shape: {n})
    labels = Nx.as_type(labels, {:s, 64})
    y = Nx.equal(Nx.iota({n, @num_classes}, axis: 1), Nx.new_axis(labels, -1))
    y = Nx.as_type(y, :f32)
    {x, y}
  end

  defp compute_accuracy(model, x, y) do
    preds = forward_pass(model, x)
    pred_classes = Nx.argmax(preds, axis: 1)
    true_classes = Nx.argmax(y, axis: 1)
    correct = Nx.equal(pred_classes, true_classes) |> Nx.as_type(:f32)
    Nx.mean(correct) |> Nx.to_number()
  end

  defp forward_pass(%ExBurn.Model{params: params}, input) do
    h = Nx.max(Nx.add(Nx.dot(input, params["fc1"]["weight"]), params["fc1"]["bias"]), 0.0)
    h = Nx.max(Nx.add(Nx.dot(h, params["fc2"]["weight"]), params["fc2"]["bias"]), 0.0)
    Nx.add(Nx.dot(h, params["output"]["weight"]), params["output"]["bias"])
  end
end

MobileInference.run()
