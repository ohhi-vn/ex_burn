# Benchmark: Element-wise Arithmetic Operations
# Run with: mix run bench/arithmetic_bench.exs
#
# Benchmarks element-wise arithmetic (add, mul, etc.) comparing
# Nx backend vs BurnBridge direct operations.

Mix.install([
  {:nx, "~> 0.7"},
  {:ex_burn, path: Path.expand("..", __DIR__)}
])

defmodule ArithmeticBench do
  @sizes [100, 500, 1000, 2000]

  def run do
    IO.puts("=== Element-wise Arithmetic Benchmark ===\n")
    IO.puts("Size       Nx add      Burn add    Nx mul      Burn mul    Nx exp      Burn exp")
    IO.puts(String.duplicate("-", 90))

    for size <- @sizes do
      nx_a = Nx.Random.uniform(Nx.Random.key(1), -1.0, 1.0, shape: {size, size})
      nx_b = Nx.Random.uniform(Nx.Random.key(2), -1.0, 1.0, shape: {size, size})
      burn_a = ExBurn.BurnBridge.from_nx(nx_a)
      burn_b = ExBurn.BurnBridge.from_nx(nx_b)

      nx_add_ms = benchmark(fn -> Nx.add(nx_a, nx_b) end)
      burn_add_ms = benchmark(fn -> ExBurn.BurnBridge.add(burn_a, burn_b) end)
      nx_mul_ms = benchmark(fn -> Nx.multiply(nx_a, nx_b) end)
      burn_mul_ms = benchmark(fn -> ExBurn.BurnBridge.mul(burn_a, burn_b) end)
      nx_exp_ms = benchmark(fn -> Nx.exp(nx_a) end)
      burn_exp_ms = benchmark(fn -> ExBurn.BurnBridge.exp(burn_a) end)

      label = "#{size}x#{size}"

      IO.puts(
        "#{pad(label, 10)} #{fmt(nx_add_ms)} #{fmt(burn_add_ms)} #{fmt(nx_mul_ms)} #{fmt(burn_mul_ms)} #{fmt(nx_exp_ms)} #{fmt(burn_exp_ms)}"
      )
    end

    IO.puts("")
  end

  defp benchmark(fun, runs \\ 30) do
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

ArithmeticBench.run()
