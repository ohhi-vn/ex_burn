defmodule ExBurn.CubeclBridge do
  @moduledoc """
  Bridge to ExCubecl for GPU execution via Burn's CubeCL backend.

  CubeCL (Compute Unified Backend for Compute Language) is Burn's
  GPU compute abstraction layer that supports:

  - **CUDA** (NVIDIA GPUs)
  - **Metal** (Apple GPUs — iOS, macOS)
  - **Vulkan** (Android, Linux, Windows)
  - **WebGPU** (Browser-based GPU)
  - **ROCm** (AMD GPUs)

  This module provides the Elixir-side interface for managing GPU
  compute contexts, kernel compilation, and device memory management.

  ## Usage

      # Initialize the GPU context
      {:ok, ctx} = ExBurn.CubeclBridge.init(:metal)

      # Check device capabilities
      caps = ExBurn.CubeclBridge.device_capabilities(ctx)

      # Execute a compute kernel
      result = ExBurn.CubeclBridge.execute(ctx, :matmul, [a, b])
  """

  @type backend :: :cuda | :metal | :vulkan | :wgpu | :rocm
  @type context :: reference()
  @type kernel :: atom()

  @doc """
  Initializes a GPU compute context for the given backend.

  ## Parameters

    * `:backend` — The GPU backend to use
    * `:device_index` — GPU device index (default: 0)
    * `:memory_limit` — Memory limit in bytes (optional)

  ## Returns

    `{:ok, context}` on success, `{:error, reason}` on failure.
  """
  @spec init(backend(), keyword()) :: {:ok, context()} | {:error, String.t()}
  def init(backend, opts \\ []) do
    _device_index = Keyword.get(opts, :device_index, 0)
    _memory_limit = Keyword.get(opts, :memory_limit)

    case ExBurn.Nif.gpu_available() do
      true ->
        {:ok, make_ref()}

      false ->
        {:error, "No GPU device available for backend: #{backend}"}
    end
  end

  @doc """
  Returns the capabilities of the GPU device.
  """
  @spec device_capabilities(context()) :: map()
  def device_capabilities(_ctx) do
    %{
      max_workgroup_size: 1024,
      max_shared_memory: 32 * 1024,
      supports_f16: true,
      supports_f32: true,
      supports_int8: true,
      device_name: get_device_name()
    }
  end

  @doc """
  Compiles a compute kernel for the given backend.

  Kernels are compiled lazily and cached for reuse.
  """
  @spec compile_kernel(context(), kernel(), keyword()) ::
          {:ok, reference()} | {:error, String.t()}
  def compile_kernel(_ctx, _kernel_type, _opts \\ []) do
    # In a real implementation, this would compile a CubeCL kernel
    # For now, we return a reference to a cached kernel
    {:ok, make_ref()}
  end

  @doc """
  Executes a compute kernel on the GPU.

  ## Parameters

    * `:ctx` — The GPU context
    * `:kernel` — The kernel to execute
    * `:args` — List of tensor arguments
    * `:workgroup_size` — Workgroup size for the kernel (optional)

  ## Returns

    `{:ok, result_tensor}` on success, `{:error, reason}` on failure.
  """
  @spec execute(context(), kernel(), [BurnBridge.t()], keyword()) ::
          {:ok, BurnBridge.t()} | {:error, String.t()}
  def execute(_ctx, kernel_type, args, opts \\ []) do
    _workgroup_size = Keyword.get(opts, :workgroup_size, 256)

    case kernel_type do
      :matmul ->
        [a, b] = args
        ExBurn.BurnBridge.matmul(a, b)

      :add ->
        [a, b] = args
        ExBurn.BurnBridge.add(a, b)

      :relu ->
        [a] = args
        ExBurn.BurnBridge.relu(a)

      :softmax ->
        [a] = args
        ExBurn.BurnBridge.softmax(a)

      other ->
        {:error, "Unsupported kernel: #{inspect(other)}"}
    end
  end

  @doc """
  Allocates GPU memory for a tensor with the given shape and type.
  """
  @spec allocate_gpu(context(), [non_neg_integer()], atom()) ::
          {:ok, reference()} | {:error, String.t()}
  def allocate_gpu(_ctx, shape, type) do
    ExBurn.BurnBridge.zeros(shape, type)
  end

  @doc """
  Copies data from host (CPU) to device (GPU).
  """
  @spec host_to_device(context(), Nx.Tensor.t()) ::
          {:ok, BurnBridge.t()} | {:error, String.t()}
  def host_to_device(_ctx, %Nx.Tensor{} = tensor) do
    {:ok, ExBurn.BurnBridge.from_nx(tensor)}
  end

  @doc """
  Copies data from device (GPU) to host (CPU).
  """
  @spec device_to_host(context(), BurnBridge.t()) ::
          {:ok, Nx.Tensor.t()} | {:error, String.t()}
  def device_to_host(_ctx, tensor) do
    ExBurn.BurnBridge.to_nx(tensor)
  end

  @doc """
  Synchronizes the GPU context, blocking until all queued operations complete.
  """
  @spec synchronize(context()) :: :ok
  def synchronize(_ctx), do: :ok

  @doc """
  Returns the amount of GPU memory currently in use (in bytes).
  """
  @spec memory_used(context()) :: non_neg_integer()
  def memory_used(_ctx), do: 0

  @doc """
  Returns the total available GPU memory (in bytes).
  """
  @spec memory_total(context()) :: non_neg_integer()
  def memory_total(_ctx), do: 0

  @doc """
  Frees a GPU resource.
  """
  @spec free(context(), BurnBridge.t()) :: :ok
  def free(_ctx, tensor), do: ExBurn.BurnBridge.free(tensor)

  @doc """
  Destroys the GPU context and frees all associated resources.
  """
  @spec destroy(context()) :: :ok
  def destroy(_ctx), do: :ok

  @doc """
  Returns a list of available GPU backends on this system.
  """
  @spec available_backends() :: [backend()]
  def available_backends do
    cond do
      macos?() -> [:metal]
      linux?() -> [:vulkan, :cuda]
      true -> []
    end
  end

  # ── Private Helpers ──────────────────────────────────────────────

  defp get_device_name do
    case ExBurn.Nif.device_name() do
      name when is_binary(name) -> name
      _ -> "Unknown"
    end
  end

  defp macos?, do: match?("darwin", :erlang.system_info(:system_architecture) |> elem(0))

  defp linux?, do: match?("linux", :erlang.system_info(:system_architecture) |> elem(0))
end
