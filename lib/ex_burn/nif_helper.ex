defmodule ExBurn.NifHelper do
  @moduledoc """
  Helper module that wraps NIF calls and returns {:ok, result} tuples.
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

  def add_tensor(a, b), do: dispatch("add_tensor", [a, b])
  def sub_tensor(a, b), do: dispatch("sub_tensor", [a, b])
  def mul_tensor(a, b), do: dispatch("mul_tensor", [a, b])
  def div_tensor(a, b), do: dispatch("div_tensor", [a, b])
  def neg_tensor(a), do: dispatch("neg_tensor", a)
  def abs_tensor(a), do: dispatch("abs_tensor", a)
  def exp_tensor(a), do: dispatch("exp_tensor", a)
  def log_tensor(a), do: dispatch("log_tensor", a)
  def sqrt_tensor(a), do: dispatch("sqrt_tensor", a)
  def pow_tensor(a, exp), do: dispatch("pow_tensor", [a, exp])
  def sigmoid_tensor(a), do: dispatch("sigmoid_tensor", a)
  def tanh_tensor(a), do: dispatch("tanh_tensor", a)
  def relu_tensor(a), do: dispatch("relu_tensor", a)

  # ── Reductions ───────────────────────────────────────────────────

  def sum_tensor(a), do: dispatch("sum_tensor", a)
  def sum_tensor(a, _axes), do: sum_tensor(a)
  def mean_tensor(a), do: dispatch("mean_tensor", a)
  def mean_tensor(a, _axes), do: mean_tensor(a)
  def max_tensor(a), do: dispatch("max_tensor", a)
  def max_tensor(a, _axes), do: max_tensor(a)
  def min_tensor(a), do: dispatch("min_tensor", a)
  def min_tensor(a, _axes), do: min_tensor(a)

  # ── Linear Algebra ───────────────────────────────────────────────

  def matmul_tensor(a, b), do: dispatch("matmul_tensor", [a, b])
  def transpose_tensor(a), do: dispatch("transpose_tensor", a)
  def transpose_tensor(a, _dim0, _dim1), do: transpose_tensor(a)
  def dot_tensor(a, b), do: dispatch("dot_tensor", [a, b])

  # ── Shape Manipulation ───────────────────────────────────────────

  def reshape_tensor(a, shape), do: dispatch("reshape_tensor", [a, shape])
  def broadcast_tensor(a, shape), do: dispatch("broadcast_tensor", [a, shape])
  def concat_tensor(a, b), do: dispatch("concat_tensor", [a, b])

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
        %{device_name: name} -> to_string(name)
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

  def to_gpu(tensor), do: dispatch("to_gpu", tensor)
  def to_cpu(tensor), do: dispatch("to_cpu", tensor)

  # ── Memory ───────────────────────────────────────────────────────

  def free_tensor(ref) do
    ExBurn.Nif.free_tensor(ref)
  end

  # ── Neural Network ───────────────────────────────────────────────

  def softmax_tensor(a, dim), do: dispatch("softmax_tensor", [a, dim])
  def layer_norm_tensor(a, dim, eps), do: dispatch("layer_norm_tensor", [a, dim, eps])

  # ── Autodiff ────────────────────────────────────────────────────

  def backward_tensor(a), do: dispatch("backward_tensor", a)
  def grad_tensor(tensor, var), do: dispatch("grad_tensor", [tensor, var])

  # ── Loss Functions ──────────────────────────────────────────────

  def cross_entropy_loss(pred, target), do: dispatch("cross_entropy_loss", [pred, target])
  def mse_loss(pred, target), do: dispatch("mse_loss", [pred, target])

  # ── Regularization ──────────────────────────────────────────────

  def dropout(tensor, prob), do: dispatch("dropout", [tensor, prob])

  # ── Slicing ─────────────────────────────────────────────────────

  def slice_tensor(ref, ranges) do
    # Nx-based fallback: convert to Nx, slice, convert back
    try do
      {:ok, binary} = tensor_to_binary(ref)
      {:ok, shape} = tensor_shape(ref)
      nx_type = {:f, 32}

      tensor =
        binary
        |> Nx.from_binary(nx_type)
        |> Nx.reshape(List.to_tuple(shape))

      sliced =
        Enum.reduce(ranges, tensor, fn {start, stop, step}, acc ->
          Nx.slice(acc, [start], [stop - start], [step])
        end)

      data = Nx.to_binary(sliced)
      shape = Nx.shape(sliced) |> Tuple.to_list()
      new_tensor(data, shape, "f32")
    rescue
      e -> {:error, Exception.message(e)}
    end
  end

  # ── Random ──────────────────────────────────────────────────────

  def random_tensor(shape, type, low, high) do
    try do
      nx_type =
        case type do
          "f32" -> {:f, 32}
          "f64" -> {:f, 64}
          "i32" -> {:s, 32}
          "i64" -> {:s, 64}
          _ -> {:f, 32}
        end

      key = Nx.Random.key(System.os_time())

      {tensor, _new_key} =
        Nx.Random.uniform(key, low, high, shape: List.to_tuple(shape), type: nx_type)

      data = Nx.to_binary(tensor)
      new_tensor(data, shape, type)
    rescue
      e -> {:error, Exception.message(e)}
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

      # Use Nx.conv/4 for 2D convolution
      output = Nx.conv(input_tensor, kernel_tensor, stride: stride, padding: padding)

      data = Nx.to_binary(output)
      shape = Nx.shape(output) |> Tuple.to_list()
      new_tensor(data, shape, "f32")
    rescue
      e -> {:error, Exception.message(e)}
    end
  end

  # ── Private ──────────────────────────────────────────────────────

  defp dispatch(fun, args) when is_list(args) do
    try do
      {:ok, apply(ExBurn.Nif, String.to_atom(fun), args)}
    rescue
      e -> {:error, Exception.message(e)}
    end
  end

  defp dispatch(fun, arg) do
    dispatch(fun, [arg])
  end
end
