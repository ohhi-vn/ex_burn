defmodule ExBurn do
  @moduledoc """
  ExBurn — Elixir bridge to the [Burn](https://burn.dev) deep learning framework.

  ExBurn provides a high-level API for tensor computation, neural network
  training, and GPU-accelerated machine learning by delegating to Burn
  via Rust NIFs (Native Implemented Functions).

  ## Architecture

  ```
  Elixir/Axon → Nx.Defn → ExBurn.Backend → ExBurn.Nif (Rustler) → Burn/CubeCL → GPU
  ```

  ## Quick Start

      # Set ExBurn as the default Nx backend
      Nx.default_backend(ExBurn.Backend)

      # Create and manipulate tensors
      t = Nx.tensor([1.0, 2.0, 3.0])
      Nx.add(t, t) |> Nx.to_list()

  ## Modules

  - `ExBurn.Backend` — Nx backend that delegates to Burn via NIF
  - `ExBurn.Nif`     — Rustler NIF stubs for Burn interop
  - `ExBurn.Tensor`  — Tensor conversion utilities between Nx and Burn formats
  - `ExBurn.BurnBridge` — High-level bridge for Burn operations
  - `ExBurn.DalaML`  — Dala ML compiler integration for mobile
  - `ExBurn.CubeclBridge` — Bridge to ExCubecl for GPU execution
  - `ExBurn.Model`   — Model definition and training orchestration
  - `ExBurn.Training` — Training loop implementation
  """

  @doc "Returns the current version of ExBurn."
  @spec version() :: String.t()
  def version, do: Application.spec(:ex_burn, :vsn) |> to_string()

  @doc """
  Returns the default device for tensor operations.

  Currently returns `:gpu` when a compatible GPU backend is available,
  otherwise falls back to `:cpu`.
  """
  @spec default_device() :: :cpu | :gpu
  def default_device do
    if ExBurn.Nif.gpu_available(), do: :gpu, else: :cpu
  end

  @doc """
  Sets the default Nx backend to `ExBurn.Backend`.

  After calling this, all Nx operations will be executed via Burn.
  """
  @spec configure!() :: :ok
  def configure! do
    Nx.default_backend(ExBurn.Backend)
    :ok
  end
end
