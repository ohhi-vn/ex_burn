# Benchmark: Nx <-> Burn Tensor Conversion
# Run with: mix run bench/conversion_bench.exs
#
# Benchmarks the overhead of converting between Nx tensors
# and Burn tensors via BurnBridge.

Mix.install([
  {:nx, "~> 0.7"},
  {:ex_burn, path: Path.expand("..", __DIR__)}
])

defmodule ConversionBench do
  @sizes [
    {10, 10},
    {100, 100},
    {500, 500},
    {1000, 1000},
    {2000, 2000},
    {100, 5000},
    {5000, 100}
  ]

  def run do
    IO.puts("=== Nx <-> Burn Conversion Benchmark ===\n")
    IO.puts("Shape          Nx->Burn    Burn->Nx    Round-trip  Overhead")
    IO.puts(String.duplicate("-", 70))

    for {rows, cols} <- @sizes do
      nx_tensor = Nx.Random.uniform(Nx.Random.key(1), -1.0, 1.0, shape: {rows, cols})

      nx_to_burn_ms = benchmark(fn -> ExBurn.BurnBridge.from_nx(nx_tensor) end)

      burn_tensor = ExBurn.BurnBridge.from_nx(nx_tensor)
      burn_to_nx_ms = benchmark(fn -> ExBurn.BurnBridge.to_nx(burn_tensor) end)

      roundtrip_ms =
        benchmark(fn ->
          bt = ExBurn.BurnBridge.from_nx(nx_tensor)
          ExBurn.BurnBridge.to_nx(bt)
        end)

      overhead = roundtrip_ms - nx_to_burn_ms - burn_to_nx_ms
      overhead_str = if overhead > 0, do: "+#{Float.round(overhead, 2)}ms", else: "negligible"

      label = "{#{rows},#{cols}}"

      IO.puts(
        "#{pad(label, 14)} #{fmt(nx_to_burn_ms)} #{fmt(burn_to_nx_ms)} #{fmt(roundtrip_ms)} #{overhead_str}"
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

ConversionBench.run()
