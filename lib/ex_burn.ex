defmodule ExBurn do
  @moduledoc """
  ExBurn — Elixir bridge to the [Burn](https://burn.dev) deep learning framework.

  ExBurn provides a high-level API for tensor computation, neural network
  training, and GPU-accelerated machine learning by delegating to Burn
  via Rust NIFs (Native Implemented Functions).

  ## Architecture

  ```
  Elixir/Axon → Nx.Defn → ExBurn.Defn.Compiler → ExBurn.Backend → ExBurn.Nif (Rustler) → Burn/CubeCL → GPU
                                                                                   ↕
                                                                             ExCubecl (GPU buffers, kernels, pipelines)
  ```

  ## Quick Start

      # Set ExBurn as the default Nx backend
      Nx.default_backend(ExBurn.Backend)

      # Create and manipulate tensors
      t = Nx.tensor([1.0, 2.0, 3.0])
      Nx.add(t, t) |> Nx.to_list()

  ## Modules

  - `ExBurn.Defn.Compiler` — `Nx.Defn.Compiler` implementation for GPU-accelerated defn
  - `ExBurn.Backend` — Nx backend that delegates to Burn via NIF
  - `ExBurn.Nif`     — Rustler NIF stubs for Burn interop
  - `ExBurn.Tensor`  — Tensor conversion utilities between Nx and Burn formats
  - `ExBurn.BurnBridge` — High-level bridge for Burn operations and ExCubecl buffers
  - `ExBurn.CubeclBridge` — GPU compute via ExCubecl (buffers, kernels, pipelines)
  - `ExBurn.Model`   — Model definition and training orchestration
  - `ExBurn.Training` — Training loop implementation
  - `ExBurn.Serving` — Nx.Serving integration for batched concurrent inference
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
    if ExBurn.NifHelper.gpu_available(), do: :gpu, else: :cpu
  end

  @doc """
  Returns the name of the active compute device (e.g., "CUDA (NVIDIA GPU)").
  """
  @spec device_name() :: String.t()
  def device_name do
    ExBurn.BurnBridge.device_name()
  end

  @doc """
  Returns a map with device information including GPU availability,
  backend name, and available backends.
  """
  @spec device_info() :: map()
  def device_info do
    ExBurn.BurnBridge.device_info()
  end

  @doc """
  Checks whether an NVIDIA CUDA GPU is available.
  """
  @spec cuda_available?() :: boolean()
  def cuda_available? do
    ExBurn.CubeclBridge.cuda_available?()
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
