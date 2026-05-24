defmodule ExBurn.NifHelper do
  @moduledoc """
  Helper module that wraps NIF calls and returns {:ok, result} tuples.
  """

  # ── Tensor Creation ──────────────────────────────────────────────

  def new_tensor(data, shape, type) do
    {:ok, ExBurn.Nif.nif_new_tensor(data, shape, type)}
  rescue
    e -> {:error, Exception.message(e)}
  end

  def zeros_tensor(shape, type) do
    {:ok, ExBurn.Nif.nif_zeros_tensor(shape, type)}
  rescue
    e -> {:error, Exception.message(e)}
  end

  def ones_tensor(shape, type) do
    {:ok, ExBurn.Nif.nif_ones_tensor(shape, type)}
  rescue
    e -> {:error, Exception.message(e)}
  end

  def eye_tensor(size, _type) do
    {:ok, ExBurn.Nif.nif_eye_tensor(size)}
  rescue
    e -> {:error, Exception.message(e)}
  end

  def iota_tensor(shape, axis, _type) do
    {:ok, ExBurn.Nif.nif_iota_tensor(shape, axis)}
  rescue
    e -> {:error, Exception.message(e)}
  end

  # ── Tensor Inspection ────────────────────────────────────────────

  def tensor_shape(ref) do
    {:ok, ExBurn.Nif.nif_tensor_shape(ref)}
  rescue
    e -> {:error, Exception.message(e)}
  end

  def tensor_to_binary(ref) do
    {:ok, ExBurn.Nif.nif_tensor_to_binary(ref)}
  rescue
    e -> {:error, Exception.message(e)}
  end

  # ── Arithmetic ───────────────────────────────────────────────────

  def add_tensor(a, b), do: dispatch("nif_add_tensor", [a, b])
  def sub_tensor(a, b), do: dispatch("nif_sub_tensor", [a, b])
  def mul_tensor(a, b), do: dispatch("nif_mul_tensor", [a, b])
  def div_tensor(a, b), do: dispatch("nif_div_tensor", [a, b])
  def neg_tensor(a), do: dispatch("nif_neg_tensor", a)
  def abs_tensor(a), do: dispatch("nif_abs_tensor", a)
  def exp_tensor(a), do: dispatch("nif_exp_tensor", a)
  def log_tensor(a), do: dispatch("nif_log_tensor", a)
  def sqrt_tensor(a), do: dispatch("nif_sqrt_tensor", a)
  def pow_tensor(a, exp), do: dispatch("nif_pow_tensor", [a, exp])
  def sigmoid_tensor(a), do: dispatch("nif_sigmoid_tensor", a)
  def tanh_tensor(a), do: dispatch("nif_tanh_tensor", a)
  def relu_tensor(a), do: dispatch("nif_relu_tensor", a)

  # ── Reductions ───────────────────────────────────────────────────

  def sum_tensor(a), do: dispatch("nif_sum_tensor", a)
  def sum_tensor(a, _axes), do: sum_tensor(a)
  def mean_tensor(a), do: dispatch("nif_mean_tensor", a)
  def mean_tensor(a, _axes), do: mean_tensor(a)
  def max_tensor(a), do: dispatch("nif_max_tensor", a)
  def max_tensor(a, _axes), do: max_tensor(a)
  def min_tensor(a), do: dispatch("nif_min_tensor", a)
  def min_tensor(a, _axes), do: min_tensor(a)

  # ── Linear Algebra ───────────────────────────────────────────────

  def matmul_tensor(a, b), do: dispatch("nif_matmul_tensor", [a, b])
  def transpose_tensor(a), do: dispatch("nif_transpose_tensor", a)
  def transpose_tensor(a, _dim0, _dim1), do: transpose_tensor(a)
  def dot_tensor(a, b), do: dispatch("nif_dot_tensor", [a, b])

  # ── Shape Manipulation ───────────────────────────────────────────

  def reshape_tensor(a, shape), do: dispatch("nif_reshape_tensor", [a, shape])
  def broadcast_tensor(a, shape), do: dispatch("nif_broadcast_tensor", [a, shape])
  def concat_tensor(a, b), do: dispatch("nif_concat_tensor", [a, b])

  # ── Device Management ────────────────────────────────────────────

  def gpu_available do
    try do
      ExCubecl.available?()
    rescue
      _ ->
        # Fallback to NIF check if ExCubecl not available
        try do
          ExBurn.Nif.nif_gpu_available()
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
          ExBurn.Nif.nif_device_name()
        rescue
          _ -> "Unknown"
        end
    end
  end

  def to_gpu(tensor), do: dispatch("nif_to_gpu", tensor)
  def to_cpu(tensor), do: dispatch("nif_to_cpu", tensor)

  # ── Memory ───────────────────────────────────────────────────────

  def free_tensor(ref) do
    ExBurn.Nif.nif_free_tensor(ref)
  end

  # ── Neural Network ───────────────────────────────────────────────

  def softmax_tensor(a, dim), do: dispatch("nif_softmax_tensor", [a, dim])
  def layer_norm_tensor(a, _dim, _eps), do: dispatch("nif_layer_norm_tensor", a)

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
