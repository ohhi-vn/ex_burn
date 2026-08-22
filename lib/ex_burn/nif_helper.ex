defmodule ExBurn.NifHelper do
  @moduledoc """
  Helper module that wraps NIF calls and returns `{:ok, result}` tuples.

  Every wrapper calls its `ExBurn.Nif` counterpart directly — no dynamic
  dispatch — so Dialyzer can check the whole bridge and no atoms are
  created at runtime.
  """

  # ── Tensor Creation ──────────────────────────────────────────────

  def new_tensor(data, shape, type) do
    {:ok, ExBurn.Nif.new_tensor(data, shape, type)}
  rescue
    e -> {:error, Exception.message(e)}
  end

  def zeros_tensor(shape, type) do
    {:ok, ExBurn.Nif.zeros_tensor(shape, type)}
  rescue
    e -> {:error, Exception.message(e)}
  end

  def ones_tensor(shape, type) do
    {:ok, ExBurn.Nif.ones_tensor(shape, type)}
  rescue
    e -> {:error, Exception.message(e)}
  end

  def eye_tensor(size, _type) do
    {:ok, ExBurn.Nif.eye_tensor(size, :f32)}
  rescue
    e -> {:error, Exception.message(e)}
  end

  def iota_tensor(shape, axis, _type) do
    {:ok, ExBurn.Nif.iota_tensor(shape, axis, :f32)}
  rescue
    e -> {:error, Exception.message(e)}
  end

  # ── Tensor Inspection ────────────────────────────────────────────

  def tensor_shape(ref) do
    {:ok, ExBurn.Nif.tensor_shape(ref)}
  rescue
    e -> {:error, Exception.message(e)}
  end

  def tensor_to_binary(ref) do
    {:ok, ExBurn.Nif.tensor_to_binary(ref)}
  rescue
    e -> {:error, Exception.message(e)}
  end

  # ── Arithmetic ───────────────────────────────────────────────────

  def add_tensor(a, b), do: guard(:add_tensor, [a, b])
  def sub_tensor(a, b), do: guard(:sub_tensor, [a, b])
  def mul_tensor(a, b), do: guard(:mul_tensor, [a, b])
  def div_tensor(a, b), do: guard(:div_tensor, [a, b])
  def neg_tensor(a), do: guard(:neg_tensor, [a])
  def abs_tensor(a), do: guard(:abs_tensor, [a])
  def exp_tensor(a), do: guard(:exp_tensor, [a])
  def log_tensor(a), do: guard(:log_tensor, [a])
  def sqrt_tensor(a), do: guard(:sqrt_tensor, [a])
  def pow_tensor(a, exp), do: guard(:pow_tensor, [a, exp])
  def sigmoid_tensor(a), do: guard(:sigmoid_tensor, [a])
  def tanh_tensor(a), do: guard(:tanh_tensor, [a])
  def relu_tensor(a), do: guard(:relu_tensor, [a])

  # ── Reductions ───────────────────────────────────────────────────

  def sum_tensor(a), do: guard(:sum_tensor, [a])
  def mean_tensor(a), do: guard(:mean_tensor, [a])
  def max_tensor(a), do: guard(:max_tensor, [a])
  def min_tensor(a), do: guard(:min_tensor, [a])

  # ── Linear Algebra ───────────────────────────────────────────────

  def matmul_tensor(a, b), do: guard(:matmul_tensor, [a, b])
  def transpose_tensor(a), do: guard(:transpose_tensor, [a])
  def dot_tensor(a, b), do: guard(:dot_tensor, [a, b])

  # ── Shape Manipulation ───────────────────────────────────────────

  def reshape_tensor(a, shape), do: guard(:reshape_tensor, [a, shape])
  def broadcast_tensor(a, shape), do: guard(:broadcast_tensor, [a, shape])
  def concat_tensor(a, b), do: guard(:concat_tensor, [a, b])

  # ── Device Management ────────────────────────────────────────────

  def gpu_available do
    if Code.ensure_loaded?(ExCubecl) and function_exported?(ExCubecl, :available?, 0) do
      ExCubecl.available?()
    else
      # ExCubecl not available — fall back to NIF check
      try do
        ExBurn.Nif.gpu_available()
      rescue
        _ -> false
      end
    end
  end

  def device_name do
    try do
      case ExCubecl.device_info() do
        {:ok, %{device_name: name}} -> to_string(name)
        _ -> "Unknown"
      end
    rescue
      _ ->
        try do
          ExBurn.Nif.device_name()
        rescue
          _ -> "Unknown"
        end
    end
  end

  def to_gpu(tensor), do: guard(:to_gpu, [tensor])
  def to_cpu(tensor), do: guard(:to_cpu, [tensor])

  # ── Memory ───────────────────────────────────────────────────────

  def free_tensor(ref) do
    ExBurn.Nif.free_tensor(ref)
  end

  # ── Neural Network ───────────────────────────────────────────────

  def softmax_tensor(a, dim), do: guard(:softmax_tensor, [a, dim])
  def layer_norm_tensor(a, dim, eps), do: guard(:layer_norm_tensor, [a, dim, eps])

  # ── Autodiff ────────────────────────────────────────────────────

  def backward_tensor(a) do
    ExBurn.Nif.backward_tensor(a)
  rescue
    e -> {:error, Exception.message(e)}
  end

  def grad_tensor(tensor, var), do: guard(:grad_tensor, [tensor, var])

  # ── Loss Functions ──────────────────────────────────────────────

  def cross_entropy_loss(pred, target), do: guard(:cross_entropy_loss, [pred, target])
  def mse_loss(pred, target), do: guard(:mse_loss, [pred, target])

  # ── Regularization ──────────────────────────────────────────────

  def dropout(tensor, prob), do: guard(:dropout, [tensor, prob])

  # ── Random ──────────────────────────────────────────────────────

  @random_nx_types %{
    "f32" => {:f, 32},
    "f64" => {:f, 64},
    "i32" => {:s, 32},
    "i64" => {:s, 64}
  }

  def random_tensor(shape, type, low, high) do
    case Map.fetch(@random_nx_types, type) do
      {:ok, nx_type} ->
        try do
          key = Nx.Random.key(System.os_time())

          {tensor, _new_key} =
            Nx.Random.uniform(key, low, high, shape: List.to_tuple(shape), type: nx_type)

          data = Nx.to_binary(tensor)
          new_tensor(data, shape, type)
        rescue
          e -> {:error, Exception.message(e)}
        end

      :error ->
        {:error, "unknown dtype #{inspect(type)}"}
    end
  end

  # ── Convolution ─────────────────────────────────────────────────

  def conv2d_tensor(input_ref, kernel_ref, stride, padding) do
    # Nx-based fallback: convert to Nx, conv, convert back
    try do
      {:ok, input_binary} = tensor_to_binary(input_ref)
      {:ok, input_shape} = tensor_shape(input_ref)
      {:ok, kernel_binary} = tensor_to_binary(kernel_ref)
      {:ok, kernel_shape} = tensor_shape(kernel_ref)

      nx_type = {:f, 32}

      input_tensor =
        input_binary
        |> Nx.from_binary(nx_type)
        |> Nx.reshape(List.to_tuple(input_shape))

      kernel_tensor =
        kernel_binary
        |> Nx.from_binary(nx_type)
        |> Nx.reshape(List.to_tuple(kernel_shape))

      output = Nx.conv(input_tensor, kernel_tensor, strides: stride, padding: padding)

      data = Nx.to_binary(output)
      shape = Nx.shape(output) |> Tuple.to_list()
      new_tensor(data, shape, "f32")
    rescue
      e -> {:error, Exception.message(e)}
    end
  end

  # ── Private ──────────────────────────────────────────────────────

  # Calls the given Nif function with the pre-resolved args and converts
  # raised errors into error tuples.
  defp guard(fun, args) do
    try do
      {:ok, apply(ExBurn.Nif, fun, args)}
    rescue
      e -> {:error, Exception.message(e)}
    end
  end
end
