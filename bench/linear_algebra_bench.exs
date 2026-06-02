# Benchmark: Linear Algebra Operations
# Run with: mix run bench/linear_algebra_bench.exs
#
# Benchmarks matrix multiplication and transpose comparing
# Nx backend vs BurnBridge direct operations.

Mix.install([
  {:nx, "~> 0.7"},
  {:ex_burn, path: Path.expand("..", __DIR__)}
])

defmodule LinearAlgebraBench do
  @sizes [50, 100, 250, 500, 1000]

  def run do
    IO.puts("=== Linear Algebra Benchmark ===\n")
    IO.puts("Size       Nx matmul   Burn matmul Nx transpose  Burn transpose")
    IO.puts(String.duplicate("-", 80))

    for size <- @sizes do
      nx_a = Nx.Random.uniform(Nx.Random.key(1), -1.0, 1.0, shape: {size, size})
      nx_b = Nx.Random.uniform(Nx.Random.key(2), -1.0, 1.0, shape: {size, size})
      burn_a = ExBurn.BurnBridge.from_nx(nx_a)
      burn_b = ExBurn.BurnBridge.from_nx(nx_b)

      nx_matmul_ms = benchmark(fn -> Nx.dot(nx_a, nx_b) end)
      burn_matmul_ms = benchmark(fn -> ExBurn.BurnBridge.matmul(burn_a, burn_b) end)
      nx_transpose_ms = benchmark(fn -> Nx.transpose(nx_a) end)
      burn_transpose_ms = benchmark(fn -> ExBurn.BurnBridge.transpose(burn_a) end)

      label = "#{size}x#{size}"

      IO.puts(
        "#{pad(label, 10)} #{fmt(nx_matmul_ms)} #{fmt(burn_matmul_ms)} #{fmt(nx_transpose_ms)} #{fmt(burn_transpose_ms)}"
      )
    end

    IO.puts("")
  end

  defp benchmark(fun, runs \\ 20) do
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

LinearAlgebraBench.run()
