# Benchmark: Nx.Serving Inference
# Run with: mix run bench/serving_bench.exs
#
# Benchmarks batched concurrent inference using ExBurn.Serving.
# Compares different batch sizes and measures throughput.

Mix.install([
  {:nx, "~> 0.7"},
  {:axon, "~> 0.7"},
  {:ex_burn, path: Path.expand("..", __DIR__)}
])

defmodule ServingBench do
  @batch_sizes [1, 4, 16, 32, 64]
  @input_dim 20
  @num_classes 5

  def run do
    IO.puts("=== Nx.Serving Inference Benchmark ===\n")

    model =
      Axon.input("input", shape: {nil, @input_dim})
      |> Axon.dense(64, activation: :relu, name: "hidden1")
      |> Axon.dense(32, activation: :relu, name: "hidden2")
      |> Axon.dense(@num_classes, name: "output")

    compiled =
      ExBurn.Model.compile(model,
        loss: :cross_entropy,
        optimizer: :adam,
        learning_rate: 0.001
      )

    info = ExBurn.Model.info(compiled)
    IO.puts("Model: #{@input_dim}->64->32->#{@num_classes} (#{info.total_params} params)\n")

    # Single inference latency
    IO.puts("--- Single Inference Latency ---")
    input = Nx.Random.uniform(Nx.Random.key(1), -1.0, 1.0, shape: {1, @input_dim})

    bench = ExBurn.Model.benchmark(compiled, input, warmup: 5, runs: 100)

    IO.puts(
      "  Avg: #{bench.avg_ms}ms  Min: #{bench.min_ms}ms  Max: #{bench.max_ms}ms  Median: #{bench.median_ms}ms\n"
    )

    # Batched inference throughput
    IO.puts("--- Batched Inference Throughput ---")
    IO.puts("Batch Size  Total (ms)  Per sample (ms)  Samples/sec")
    IO.puts(String.duplicate("-", 55))

    for bs <- @batch_sizes do
      batch_input = Nx.Random.uniform(Nx.Random.key(bs), -1.0, 1.0, shape: {bs, @input_dim})

      ExBurn.Model.predict(compiled, batch_input)

      {time_us, _} =
        :timer.tc(fn ->
          ExBurn.Model.predict(compiled, batch_input)
        end)

      total_ms = time_us / 1000
      per_sample_ms = total_ms / bs
      samples_per_sec = bs / (total_ms / 1000)

      IO.puts(
        "#{pad("#{bs}", 11)} #{pad("#{Float.round(total_ms, 2)}", 11)} #{pad("#{Float.round(per_sample_ms, 3)}", 16)} #{Float.round(samples_per_sec, 0)}"
      )
    end

    # Nx.Serving
    IO.puts("\n--- Nx.Serving (batch_size=32, timeout=50ms) ---")

    serving = ExBurn.Serving.build(compiled, batch_size: 32, batch_timeout: 50)

    status =
      ExBurn.Serving.status(ExBurn.Serving.new(compiled, batch_size: 32, batch_timeout: 50))

    IO.puts("  Config: #{inspect(status)}")

    inputs =
      for i <- 1..10, do: Nx.Random.uniform(Nx.Random.key(i), -1.0, 1.0, shape: {1, @input_dim})

    {time_us, results} =
      :timer.tc(fn ->
        Enum.map(inputs, fn input ->
          Nx.Serving.run(serving, input)
        end)
      end)

    IO.puts(
      "  10 requests in #{Float.round(time_us / 1000, 1)}ms (#{Float.round(time_us / 10 / 1000, 1)}ms/req)"
    )

    IO.puts(
      "  Results: #{length(results)} outputs, first shape: #{inspect(Nx.shape(hd(results)))}"
    )

    IO.puts("\n=== Done ===")
  end

  defp pad(str, len) do
    if String.length(str) >= len,
      do: str,
      else: str <> String.duplicate(" ", len - String.length(str))
  end
end

ServingBench.run()
