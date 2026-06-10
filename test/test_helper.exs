nif_loaded? =
  match?({:module, _}, Code.ensure_loaded(ExBurn.Nif)) and
    function_exported?(ExBurn.Nif, :new_tensor, 3)

# Check which GPU backends are actually available at runtime
gpu_available? = nif_loaded? and ExBurn.Nif.gpu_available()

# Only exclude GPU-specific tests when the NIF isn't loaded or no GPU is available
# This allows CUDA/Metal/Vulkan tests to run on machines with the right hardware
gpu_excludes =
  if gpu_available? do
    []
  else
    [:cuda, :metal, :vulkan]
  end

nif_excludes = if nif_loaded?, do: [], else: [:nif]

ExUnit.start(exclude: gpu_excludes ++ nif_excludes)

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
