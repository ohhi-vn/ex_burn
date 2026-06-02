# Benchmark: Tensor Creation Operations
# Run with: mix run bench/tensor_creation_bench.exs
#
# Benchmarks the performance of creating tensors via BurnBridge
# compared to plain Nx tensor creation.

Mix.install([
  {:nx, "~> 0.7"},
  {:ex_burn, path: Path.expand("..", __DIR__)}
])

defmodule TensorCreationBench do
  @sizes [
    {10, 10},
    {100, 100},
    {500, 500},
    {1000, 1000},
    {100, 1000},
    {1000, 100}
  ]

  def run do
    IO.puts("=== Tensor Creation Benchmark ===\n")

    IO.puts(
      "Shape          Nx zeros    Burn zeros  Nx ones     Burn ones   Nx rand     Burn rand"
    )

    IO.puts(String.duplicate("-", 90))

    for {rows, cols} <- @sizes do
      shape = [rows, cols]

      nx_zeros_ms =
        benchmark(fn -> Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {rows, cols}) end)

      burn_zeros_ms = benchmark(fn -> ExBurn.BurnBridge.zeros(shape, :f32) end)
      nx_ones_ms = benchmark(fn -> Nx.broadcast(Nx.tensor(1.0, type: {:f, 32}), {rows, cols}) end)
      burn_ones_ms = benchmark(fn -> ExBurn.BurnBridge.ones(shape, :f32) end)

      nx_rand_ms =
        benchmark(fn -> Nx.Random.uniform(Nx.Random.key(1), 0.0, 1.0, shape: {rows, cols}) end)

      burn_rand_ms = benchmark(fn -> ExBurn.BurnBridge.rand(shape, :f32, 0.0, 1.0) end)

      label = "{#{rows},#{cols}}"

      IO.puts(
        "#{pad(label, 14)} #{fmt(nx_zeros_ms)} #{fmt(burn_zeros_ms)} #{fmt(nx_ones_ms)} #{fmt(burn_ones_ms)} #{fmt(nx_rand_ms)} #{fmt(burn_rand_ms)}"
      )
    end

    IO.puts("")
  end

  defp benchmark(fun, runs \\ 50) do
    fun.()

    times =
      for _ <- 1..runs do
        {time, _} = :timer.tc(fun)
        time / 1000
      end

    Enum.sum(times) / length(times)
  end

  defp fmt(ms) when ms < 1.0, do: "#{Float.round(ms * 1000, 1)} us     "
  defp fmt(ms), do: "#{Float.round(ms, 2)} ms   "

  defp pad(str, len) do
    if String.length(str) >= len,
      do: str,
      else: str <> String.duplicate(" ", len - String.length(str))
  end
end

TensorCreationBench.run()
