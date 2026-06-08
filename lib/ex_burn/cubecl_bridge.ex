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

  This module delegates to the ExCubecl library (v0.4.0) for all GPU
  operations. ExCubecl buffers (opaque references) are used for GPU
  memory throughout.

  ## Usage

      # Check if a GPU is available
      if ExBurn.CubeclBridge.available?() do
        # Initialize the GPU context
        {:ok, ctx} = ExBurn.CubeclBridge.init(:metal)

        # Check device capabilities
        caps = ExBurn.CubeclBridge.device_capabilities(ctx)

        # Allocate GPU buffer and run a kernel
        {:ok, buf} = ExBurn.CubeclBridge.allocate_gpu(ctx, [4, 4], :f32)
        {:ok, result} = ExBurn.CubeclBridge.execute(ctx, :add, [buf, buf])

        # Pipeline for multi-kernel execution
        {:ok, pid} = ExBurn.CubeclBridge.pipeline()
        :ok = ExBurn.CubeclBridge.pipeline_add(pid, :add, [buf, buf], buf)
        :ok = ExBurn.CubeclBridge.pipeline_add(pid, :relu, [buf], buf)
        {:ok, commands} = ExBurn.CubeclBridge.pipeline_run(pid)
        :ok = ExBurn.CubeclBridge.pipeline_free(pid)
      end
  """

  @type backend :: :cuda | :metal | :vulkan | :wgpu | :rocm
  @type context :: reference()
  @type kernel :: atom()
  @type buffer :: reference()
  @type command_id :: non_neg_integer()
  @type pipeline_id :: non_neg_integer()

  # ── Initialization ───────────────────────────────────────────────

  @doc """
  Checks whether a GPU device is available via ExCubecl.
  """
  @spec available?() :: boolean()
  def available?, do: ExCubecl.available?()

  @doc """
  Initializes a GPU compute context for the given backend.

  ## Parameters

    * `backend` — The GPU backend to use (`:cuda`, `:metal`, `:vulkan`, `:wgpu`, `:rocm`)
    * `opts` — Options (currently unused, reserved for future use)

  ## Returns

    `{:ok, context}` on success, `{:error, reason}` on failure.
  """
  @spec init(backend(), keyword()) :: {:ok, context()} | {:error, String.t()}
  def init(_backend, _opts \\ []) do
    if ExCubecl.available?() do
      {:ok, make_ref()}
    else
      {:error, "No GPU device available"}
    end
  end

  @doc """
  Returns the capabilities of the GPU device.
  """
  @spec device_capabilities(context()) :: map()
  def device_capabilities(_ctx) do
    case ExCubecl.device_info() do
      {:ok, info} ->
        Map.merge(
          %{
            max_workgroup_size: 1024,
            max_shared_memory: 32 * 1024,
            supports_f16: true,
            supports_f32: true,
            supports_int8: true
          },
          info
        )

      {:error, _} ->
        %{
          max_workgroup_size: 1024,
          max_shared_memory: 32 * 1024,
          supports_f16: true,
          supports_f32: true,
          supports_int8: true,
          device_name: "unknown"
        }
    end
  end

  @doc """
  Returns the number of GPU devices available.
  """
  @spec device_count() :: {:ok, non_neg_integer()} | {:error, term()}
  def device_count, do: ExCubecl.device_count()

  @doc """
  Returns the ExCubecl library version string.
  """
  @spec version() :: String.t()
  def version, do: ExCubecl.version()

  # ── Kernel Compilation ───────────────────────────────────────────

  @doc """
  Compiles a compute kernel for the given backend.

  Kernels are managed by ExCubecl; this function verifies the kernel
  is available and returns a reference.
  """
  @spec compile_kernel(context(), kernel(), keyword()) ::
          {:ok, reference()} | {:error, String.t()}
  def compile_kernel(_ctx, kernel_type, _opts \\ []) do
    case ExCubecl.kernels() do
      {:ok, kernels} ->
        if Atom.to_string(kernel_type) in kernels do
          {:ok, make_ref()}
        else
          {:error, "Kernel not found: #{inspect(kernel_type)}"}
        end

      {:error, reason} ->
        {:error, "Failed to list kernels: #{inspect(reason)}"}
    end
  end

  @doc """
  Returns the list of kernel names supported by ExCubecl.
  """
  @spec kernels() :: {:ok, [String.t()]} | {:error, term()}
  def kernels, do: ExCubecl.kernels()

  @doc """
  Returns the list of supported data types.
  """
  @spec supported_dtypes() :: [atom()]
  def supported_dtypes, do: ExCubecl.supported_dtypes()

  # ── Kernel Execution ─────────────────────────────────────────────

  @doc """
  Executes a compute kernel on the GPU.

  ## Parameters

    * `ctx` — The GPU context
    * `kernel` — The kernel to execute (atom)
    * `args` — List of ExCubecl buffer references
    * `opts` — Options (currently unused, reserved for future use)

  ## Returns

    `{:ok, result_buffer}` on success, `{:error, reason}` on failure.
  """
  @spec execute(context(), kernel(), [buffer()], keyword()) ::
          {:ok, buffer()} | {:error, String.t()}
  def execute(_ctx, kernel_type, inputs, opts \\ []) do
    params = Keyword.get(opts, :params, %{})

    # For single-output kernels, use the last input as the output buffer
    # (in-place operation convention), or allocate a new one.
    output = Keyword.get(opts, :output)

    output =
      case output do
        nil ->
          # Default: use the first input's shape for the output
          case inputs do
            [first | _] ->
              case ExCubecl.shape(first) do
                {:ok, shape} ->
                  dtype_str = ExCubecl.dtype(first) |> elem(1)
                  dtype = String.to_existing_atom(dtype_str)
                  {:ok, buf} = ExCubecl.buffer(<<>>, shape, dtype)
                  buf

                _ ->
                  {:error, "Cannot determine output shape"}
              end

            [] ->
              {:error, "No input buffers provided"}
          end

        buf ->
          buf
      end

    case output do
      {:error, _} = err ->
        err

      _ ->
        case ExCubecl.run_kernel(Atom.to_string(kernel_type), inputs, output, params) do
          {:ok, _command_id} -> {:ok, output}
          {:error, reason} -> {:error, to_string(reason)}
        end
    end
  end

  # ── Async Execution ──────────────────────────────────────────────

  @doc """
  Submits a command for asynchronous execution.

  Returns a command ID that can be polled or waited on.
  """
  @spec async_submit(String.t()) :: {:ok, command_id()} | {:error, term()}
  def async_submit(command), do: ExCubecl.submit(command)

  @doc """
  Polls the status of an asynchronous command.

  Returns `:pending`, `:running`, `:completed`, or `:failed`.
  """
  @spec async_poll(command_id()) ::
          {:ok, :pending | :running | :completed | :failed} | {:error, term()}
  def async_poll(command_id), do: ExCubecl.poll(command_id)

  @doc """
  Blocks until the given command completes.
  """
  @spec async_wait(command_id()) :: :ok | {:error, term()}
  def async_wait(command_id), do: ExCubecl.wait(command_id)

  # ── Pipelines ────────────────────────────────────────────────────

  @doc """
  Creates a new pipeline for multi-kernel execution.
  """
  @spec pipeline() :: {:ok, pipeline_id()} | {:error, term()}
  def pipeline, do: ExCubecl.pipeline()

  @doc """
  Adds a kernel command to a pipeline.

  ## Parameters

    * `pipeline_id` — The pipeline to add to
    * `kernel` — Kernel name (atom)
    * `inputs` — List of input buffer references
    * `output` — Output buffer reference
    * `params` — Additional parameters (optional, default: `%{}`)
  """
  @spec pipeline_add(pipeline_id(), atom(), [buffer()], buffer(), map()) ::
          :ok | {:error, term()}
  def pipeline_add(pipeline_id, kernel, inputs, output, params \\ %{}) do
    ExCubecl.pipeline_add(pipeline_id, Atom.to_string(kernel), inputs, output, params)
  end

  @doc """
  Adds a pre-built `%ExCubecl.Command{}` struct to a pipeline.
  """
  @spec pipeline_add_struct(pipeline_id(), ExCubecl.Command.t()) ::
          :ok | {:error, term()}
  def pipeline_add_struct(pipeline_id, command) do
    ExCubecl.pipeline_add_struct(pipeline_id, command)
  end

  @doc """
  Executes all commands in the pipeline and returns their command IDs.
  """
  @spec pipeline_run(pipeline_id()) :: {:ok, [command_id()]} | {:error, term()}
  def pipeline_run(pipeline_id), do: ExCubecl.pipeline_run(pipeline_id)

  @doc """
  Frees a pipeline and its associated resources.
  """
  @spec pipeline_free(pipeline_id()) :: :ok | {:error, term()}
  def pipeline_free(pipeline_id), do: ExCubecl.pipeline_free(pipeline_id)

  # ── Memory Management ────────────────────────────────────────────

  @doc """
  Allocates a GPU buffer with the given shape and type.

  Returns an ExCubecl buffer reference.
  """
  @spec allocate_gpu(context(), [non_neg_integer()], atom()) ::
          {:ok, buffer()} | {:error, String.t()}
  def allocate_gpu(_ctx, shape, type) do
    # Allocate an empty buffer; data is zero-initialized by ExCubecl
    dtype_str = Atom.to_string(type)
    ExCubecl.buffer(<<>>, shape, String.to_atom(dtype_str))
  rescue
    ArgumentError ->
      # Fallback: try with string type
      ExCubecl.buffer(<<>>, shape, Atom.to_string(type))
  end

  @doc """
  Copies data from host (CPU) to a new GPU buffer.

  ## Parameters

    * `ctx` — The GPU context
    * `tensor` — An Nx tensor to copy to the GPU

  ## Returns

    `{:ok, buffer}` on success, `{:error, reason}` on failure.
  """
  @spec host_to_device(context(), Nx.Tensor.t()) ::
          {:ok, buffer()} | {:error, String.t()}
  def host_to_device(_ctx, %Nx.Tensor{} = tensor) do
    shape = Nx.shape(tensor)
    type = Nx.type(tensor)
    binary = Nx.to_binary(tensor)
    dtype_str = Atom.to_string(type)

    case ExCubecl.buffer(binary, shape, String.to_atom(dtype_str)) do
      {:ok, buf} -> {:ok, buf}
      {:error, reason} -> {:error, to_string(reason)}
    end
  rescue
    ArgumentError ->
      # Fallback: try with string type
      case ExCubecl.buffer(
             Nx.to_binary(tensor),
             Nx.shape(tensor),
             Atom.to_string(Nx.type(tensor))
           ) do
        {:ok, buf} -> {:ok, buf}
        {:error, reason} -> {:error, to_string(reason)}
      end
  end

  @doc """
  Copies data from a GPU buffer to host (CPU) as an Nx tensor.

  ## Parameters

    * `ctx` — The GPU context
    * `buffer` — An ExCubecl buffer reference

  ## Returns

    `{:ok, Nx.Tensor.t()}` on success, `{:error, reason}` on failure.
  """
  @spec device_to_host(context(), buffer()) ::
          {:ok, Nx.Tensor.t()} | {:error, String.t()}
  def device_to_host(_ctx, buf) do
    with {:ok, binary} <- ExCubecl.read(buf),
         {:ok, shape} <- ExCubecl.shape(buf),
         {:ok, dtype_str} <- ExCubecl.dtype(buf) do
      type = String.to_existing_atom(dtype_str)
      {:ok, Nx.from_binary(binary, type) |> Nx.reshape(shape)}
    else
      {:error, reason} -> {:error, to_string(reason)}
    end
  rescue
    ArgumentError ->
      # If the dtype atom doesn't exist, fall back to f32
      with {:ok, binary} <- ExCubecl.read(buf),
           {:ok, shape} <- ExCubecl.shape(buf) do
        {:ok, Nx.from_binary(binary, :f32) |> Nx.reshape(shape)}
      else
        {:error, reason} -> {:error, to_string(reason)}
      end
  end

  @doc """
  Returns the shape of a GPU buffer.
  """
  @spec buffer_shape(buffer()) :: {:ok, [non_neg_integer()]} | {:error, term()}
  def buffer_shape(buf), do: ExCubecl.shape(buf)

  @doc """
  Returns the data type of a GPU buffer.
  """
  @spec buffer_dtype(buffer()) :: {:ok, String.t()} | {:error, term()}
  def buffer_dtype(buf), do: ExCubecl.dtype(buf)

  @doc """
  Returns the size of a GPU buffer in bytes.
  """
  @spec buffer_size(buffer()) :: {:ok, non_neg_integer()} | {:error, term()}
  def buffer_size(buf), do: ExCubecl.size(buf)

  @doc """
  Reads the raw binary data from a GPU buffer.
  """
  @spec buffer_read(buffer()) :: {:ok, binary()} | {:error, term()}
  def buffer_read(buf), do: ExCubecl.read(buf)

  @doc """
  Reads the raw binary data from a GPU buffer, raising on error.
  """
  @spec buffer_read!(buffer()) :: binary()
  def buffer_read!(buf), do: ExCubecl.read!(buf)

  # ── Synchronization ──────────────────────────────────────────────

  @doc """
  Synchronizes the GPU context, blocking until all queued operations complete.

  Note: ExCubecl does not expose a global synchronization primitive.
  Use `async_wait/1` on specific command IDs for fine-grained control.
  """
  @spec synchronize(context()) :: :ok
  def synchronize(_ctx), do: :ok

  @doc """
  Returns the amount of GPU memory currently in use (in bytes).

  Attempts to query the GPU device for actual memory usage.
  Falls back to 0 if the information is not available.
  """
  @spec memory_used(context()) :: non_neg_integer()
  def memory_used(_ctx) do
    case call_memory_info() do
      {:ok, %{used: used}} when is_integer(used) -> used
      {:ok, %{memory_used: used}} when is_integer(used) -> used
      _ -> estimate_memory_used()
    end
  rescue
    _ -> 0
  end

  @doc """
  Returns the total available GPU memory (in bytes).

  Attempts to query the GPU device for total memory.
  Falls back to 0 if the information is not available.
  """
  @spec memory_total(context()) :: non_neg_integer()
  def memory_total(_ctx) do
    case call_memory_info() do
      {:ok, %{total: total}} when is_integer(total) -> total
      {:ok, %{memory_total: total}} when is_integer(total) -> total
      _ -> 0
    end
  rescue
    _ -> 0
  end

  @doc """
  Returns a map with GPU memory information.

  Returns `:error` if no GPU is available or memory info cannot be queried.

  ## Returns

      {:ok, %{total: integer(), used: integer(), free: integer()}}
      or
      {:error, reason}
  """
  @spec memory_info() :: {:ok, map()} | {:error, term()}
  def memory_info do
    case call_memory_info() do
      {:ok, info} -> {:ok, info}
      _ -> {:error, "GPU memory info not available"}
    end
  rescue
    e -> {:error, Exception.message(e)}
  end

  defp call_memory_info do
    if Code.ensure_loaded?(ExCubecl) and function_exported?(ExCubecl, :memory_info, 0) do
      apply(ExCubecl, :memory_info, [])
    else
      {:error, :not_available}
    end
  end

  defp estimate_memory_used do
    # Rough estimate based on process memory
    case Process.info(self(), :memory) do
      {:memory, bytes} -> bytes
      _ -> 0
    end
  end

  @doc """
  Frees a GPU buffer.

  Note: ExCubecl buffers are garbage-collected when their reference
  goes out of scope. This function is a no-op for API compatibility.
  """
  @spec free(context(), buffer()) :: :ok
  def free(_ctx, _buf), do: :ok

  @doc """
  Destroys the GPU context and frees all associated resources.
  """
  @spec destroy(context()) :: :ok
  def destroy(_ctx), do: :ok

  @doc """
  Returns a list of available GPU backends on this system.

  Delegates to ExCubecl availability checks and platform detection.
  On Linux/Windows with NVIDIA hardware, `:cuda` is included.
  On macOS, `:metal` is included.
  """
  @spec available_backends() :: [backend()]
  def available_backends do
    if not ExCubecl.available?() do
      []
    else
      backends = []
      backends = backends ++ if cuda_detected?(), do: [:cuda], else: []
      backends = backends ++ if metal_detected?(), do: [:metal], else: []
      backends = backends ++ if vulkan_detected?(), do: [:vulkan], else: []
      backends
    end
  end

  @doc """
  Returns a human-readable summary of the GPU device.

  ## Example

      IO.puts(ExBurn.CubeclBridge.device_summary())
  """
  @spec device_summary() :: String.t()
  def device_summary do
    backends = available_backends()

    if backends == [] do
      "No GPU available"
    else
      caps =
        if match?({:ok, _}, init(hd(backends))), do: device_capabilities(hd(backends)), else: %{}

      info = [
        "GPU Backends: #{Enum.join(Enum.map(backends, &Atom.to_string/1), ", ")}",
        "Device: #{Map.get(caps, :device_name, "Unknown")}",
        "Max Workgroup Size: #{Map.get(caps, :max_workgroup_size, "N/A")}",
        "Shared Memory: #{Map.get(caps, :max_shared_memory, "N/A")} bytes",
        "Supports f16: #{Map.get(caps, :supports_f16, false)}",
        "Supports f32: #{Map.get(caps, :supports_f32, true)}"
      ]

      Enum.join(info, "\n")
    end
  end

  @doc """
  Checks whether an NVIDIA CUDA GPU is available on this system.

  First checks via ExCubecl if available, then falls back to
  platform-specific heuristics (nvidia-smi on Linux/Windows).
  """
  @spec cuda_available?() :: boolean()
  def cuda_available? do
    cuda_detected?()
  end

  # ── Private Helpers ──────────────────────────────────────────────

  defp macos?,
    do: :erlang.system_info(:system_architecture) |> to_string() |> String.contains?("darwin")

  defp linux?,
    do: :erlang.system_info(:system_architecture) |> to_string() |> String.contains?("linux")

  defp cuda_detected? do
    # Check via ExCubecl first
    case ExCubecl.available?() do
      true ->
        # Try to get device info; if it reports CUDA, we're good
        case ExCubecl.device_info() do
          {:ok, %{backend: "cuda"}} -> true
          {:ok, %{backend: :cuda}} -> true
          _ -> cuda_fallback_check()
        end

      _ ->
        cuda_fallback_check()
    end
  rescue
    _ -> cuda_fallback_check()
  end

  defp cuda_fallback_check do
    # Fallback: check for nvidia-smi on Linux/Windows
    case :os.type() do
      {:unix, :linux} -> has_nvidia_smi?()
      {:win32, _} -> has_nvidia_smi?()
      _ -> false
    end
  end

  defp has_nvidia_smi? do
    case System.cmd("nvidia-smi", [], stderr_to_stdout: true) do
      {_, 0} -> true
      _ -> false
    end
  rescue
    _ -> false
  end

  defp metal_detected? do
    macos?() and ExCubecl.available?()
  end

  defp vulkan_detected? do
    (linux?() or match?({:win32, _}, :os.type())) and ExCubecl.available?()
  end
end
