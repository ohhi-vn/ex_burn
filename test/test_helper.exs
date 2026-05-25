ExUnit.start(exclude: [:nif, :cuda, :metal, :vulkan])

# Shared test fixtures
defmodule ExBurn.TestFixtures do
  @moduledoc "Shared test data and helpers."

  def scalar(val \\ 1.0), do: Nx.tensor(val)
  def vector(vals), do: Nx.tensor(vals)
  def matrix(rows), do: Nx.tensor(rows)

  def assert_close(a, b, tolerance \\ 1.0e-4) do
    diff = Nx.subtract(a, b) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()

    if diff >= tolerance do
      raise ExUnit.AssertionError,
        message: "Expected #{inspect(Nx.to_list(a))} ≈ #{inspect(Nx.to_list(b))} (diff=#{diff})",
        left: a,
        right: b
    end
  end
end
