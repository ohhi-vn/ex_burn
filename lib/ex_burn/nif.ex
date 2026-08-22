defmodule ExBurn.Nif do
  @moduledoc """
  Rustler NIF stubs for interfacing with the Burn deep learning framework.

  The Rust side returns bare values for these operations. Fallible operations
  raise on failure (matching the pre-0.22 behaviour).
  """

  use Rustler,
    otp_app: :ex_burn,
    crate: :ex_burn_nif

  @compile {:no_warn_undefined, __MODULE__}

  # The Rust layer currently stores f32 data only. Reject other dtypes
  # *before* crossing the NIF boundary so users get a structured
  # ExBurn.Error instead of a Rust panic (or worse, silent byte
  # reinterpretation).
  defp validate_dtype!(:f32), do: :ok
  defp validate_dtype!("f32"), do: :ok

  defp validate_dtype!(other) do
    raise ExBurn.Error,
      op: :new_tensor,
      reason:
        "dtype #{inspect(other)} is not supported by the NIF yet — tensors are stored as :f32"
  end

  # ── Tensor Creation ──────────────────────────────────────────────

  def new_tensor(data, shape, type) do
    validate_dtype!(type)
    nif_new_tensor(data, shape, normalize_type(type))
  end

  def empty_tensor(shape, type) do
    validate_dtype!(type)
    nif_empty_tensor(shape, normalize_type(type))
  end

  def zeros_tensor(shape, type) do
    validate_dtype!(type)
    nif_zeros_tensor(shape, normalize_type(type))
  end

  def ones_tensor(shape, type) do
    validate_dtype!(type)
    nif_ones_tensor(shape, normalize_type(type))
  end

  defp normalize_type(type) when is_atom(type), do: Atom.to_string(type)
  defp normalize_type(type) when is_binary(type), do: type

  # ── Tensor Inspection ────────────────────────────────────────────

  def tensor_shape(tensor), do: nif_tensor_shape(tensor)
  def tensor_dtype(tensor), do: nif_tensor_dtype(tensor)
  def tensor_to_binary(tensor), do: nif_tensor_to_binary(tensor)
  def tensor_numel(tensor), do: nif_tensor_numel(tensor)

  # ── Element-wise Arithmetic ──────────────────────────────────────

  def add_tensor(a, b), do: nif_add_tensor(a, b)
  def sub_tensor(a, b), do: nif_sub_tensor(a, b)
  def mul_tensor(a, b), do: nif_mul_tensor(a, b)
  def div_tensor(a, b), do: nif_div_tensor(a, b)
  def neg_tensor(a), do: nif_neg_tensor(a)
  def abs_tensor(a), do: nif_abs_tensor(a)
  def exp_tensor(a), do: nif_exp_tensor(a)
  def log_tensor(a), do: nif_log_tensor(a)
  def sqrt_tensor(a), do: nif_sqrt_tensor(a)
  def pow_tensor(a, exp), do: nif_pow_tensor(a, exp)
  def sigmoid_tensor(a), do: nif_sigmoid_tensor(a)
  def tanh_tensor(a), do: nif_tanh_tensor(a)
  def relu_tensor(a), do: nif_relu_tensor(a)

  # ── Reductions ───────────────────────────────────────────────────

  def sum_tensor(a), do: nif_sum_tensor(a)
  def mean_tensor(a), do: nif_mean_tensor(a)
  def max_tensor(a), do: nif_max_tensor(a)
  def min_tensor(a), do: nif_min_tensor(a)

  # ── Linear Algebra ───────────────────────────────────────────────

  def matmul_tensor(a, b), do: nif_matmul_tensor(a, b)
  def transpose_tensor(a), do: nif_transpose_tensor(a)
  def dot_tensor(a, b), do: nif_dot_tensor(a, b)

  # ── Shape Manipulation ───────────────────────────────────────────

  def reshape_tensor(a, shape), do: nif_reshape_tensor(a, shape)
  def broadcast_tensor(a, shape), do: nif_broadcast_tensor(a, shape)
  def concat_tensor(a, b), do: nif_concat_tensor(a, b)

  # ── Autograd / Backward ──────────────────────────────────────────

  def backward_tensor(a), do: nif_backward_tensor(a)
  def grad_tensor(tensor, var), do: nif_grad_tensor(tensor, var)

  # ── Device Management ────────────────────────────────────────────

  def gpu_available, do: nif_gpu_available()
  def device_name, do: nif_device_name()
  def to_gpu(tensor), do: nif_to_gpu(tensor)
  def to_cpu(tensor), do: nif_to_cpu(tensor)

  # ── Memory Management ────────────────────────────────────────────

  def free_tensor(tensor), do: nif_free_tensor(tensor)

  # ── Tensor Creation (cont.) ──────────────────────────────────────

  def eye_tensor(size, type) do
    validate_dtype!(type)
    nif_eye_tensor(size, normalize_type(type))
  end

  def iota_tensor(shape, axis, type) do
    validate_dtype!(type)
    nif_iota_tensor(shape, axis, normalize_type(type))
  end

  # ── Neural Network Operations ────────────────────────────────────

  def softmax_tensor(a, dim), do: nif_softmax_tensor(a, dim)
  def layer_norm_tensor(a, dim, eps), do: nif_layer_norm_tensor(a, dim, eps)

  # ── Loss Functions ──────────────────────────────────────────────

  def cross_entropy_loss(pred, target), do: nif_cross_entropy_loss(pred, target)
  def mse_loss(pred, target), do: nif_mse_loss(pred, target)

  # ── Regularization ──────────────────────────────────────────────

  def dropout(tensor, prob), do: nif_dropout(tensor, prob)

  # ── Raw NIF entry points (registered in Rust) ────────────────────

  @doc """
  Returns the number of NIF functions registered by the Rust library.
  """
  def debug_nif_count, do: :erlang.nif_error(:nif_not_loaded)

  def nif_new_tensor(_data, _shape, _type), do: :erlang.nif_error(:nif_not_loaded)
  def nif_empty_tensor(_shape, _type), do: :erlang.nif_error(:nif_not_loaded)
  def nif_zeros_tensor(_shape, _type), do: :erlang.nif_error(:nif_not_loaded)
  def nif_ones_tensor(_shape, _type), do: :erlang.nif_error(:nif_not_loaded)
  def nif_tensor_shape(_tensor), do: :erlang.nif_error(:nif_not_loaded)
  def nif_tensor_dtype(_tensor), do: :erlang.nif_error(:nif_not_loaded)
  def nif_tensor_to_binary(_tensor), do: :erlang.nif_error(:nif_not_loaded)
  def nif_tensor_numel(_tensor), do: :erlang.nif_error(:nif_not_loaded)
  def nif_add_tensor(_a, _b), do: :erlang.nif_error(:nif_not_loaded)
  def nif_sub_tensor(_a, _b), do: :erlang.nif_error(:nif_not_loaded)
  def nif_mul_tensor(_a, _b), do: :erlang.nif_error(:nif_not_loaded)
  def nif_div_tensor(_a, _b), do: :erlang.nif_error(:nif_not_loaded)
  def nif_neg_tensor(_a), do: :erlang.nif_error(:nif_not_loaded)
  def nif_abs_tensor(_a), do: :erlang.nif_error(:nif_not_loaded)
  def nif_exp_tensor(_a), do: :erlang.nif_error(:nif_not_loaded)
  def nif_log_tensor(_a), do: :erlang.nif_error(:nif_not_loaded)
  def nif_sqrt_tensor(_a), do: :erlang.nif_error(:nif_not_loaded)
  def nif_pow_tensor(_a, _exp), do: :erlang.nif_error(:nif_not_loaded)
  def nif_sigmoid_tensor(_a), do: :erlang.nif_error(:nif_not_loaded)
  def nif_tanh_tensor(_a), do: :erlang.nif_error(:nif_not_loaded)
  def nif_relu_tensor(_a), do: :erlang.nif_error(:nif_not_loaded)
  def nif_sum_tensor(_a), do: :erlang.nif_error(:nif_not_loaded)
  def nif_mean_tensor(_a), do: :erlang.nif_error(:nif_not_loaded)
  def nif_max_tensor(_a), do: :erlang.nif_error(:nif_not_loaded)
  def nif_min_tensor(_a), do: :erlang.nif_error(:nif_not_loaded)
  def nif_matmul_tensor(_a, _b), do: :erlang.nif_error(:nif_not_loaded)
  def nif_transpose_tensor(_a), do: :erlang.nif_error(:nif_not_loaded)
  def nif_dot_tensor(_a, _b), do: :erlang.nif_error(:nif_not_loaded)
  def nif_reshape_tensor(_a, _shape), do: :erlang.nif_error(:nif_not_loaded)
  def nif_broadcast_tensor(_a, _shape), do: :erlang.nif_error(:nif_not_loaded)
  def nif_concat_tensor(_a, _b), do: :erlang.nif_error(:nif_not_loaded)
  def nif_backward_tensor(_a), do: :erlang.nif_error(:nif_not_loaded)
  def nif_grad_tensor(_tensor, _var), do: :erlang.nif_error(:nif_not_loaded)
  def nif_gpu_available, do: :erlang.nif_error(:nif_not_loaded)
  def nif_device_name, do: :erlang.nif_error(:nif_not_loaded)
  def nif_to_gpu(_tensor), do: :erlang.nif_error(:nif_not_loaded)
  def nif_to_cpu(_tensor), do: :erlang.nif_error(:nif_not_loaded)
  def nif_free_tensor(_tensor), do: :erlang.nif_error(:nif_not_loaded)
  def nif_eye_tensor(_size, _type), do: :erlang.nif_error(:nif_not_loaded)
  def nif_iota_tensor(_shape, _axis, _type), do: :erlang.nif_error(:nif_not_loaded)
  def nif_softmax_tensor(_a, _dim), do: :erlang.nif_error(:nif_not_loaded)
  def nif_layer_norm_tensor(_a, _dim, _eps), do: :erlang.nif_error(:nif_not_loaded)
  def nif_cross_entropy_loss(_pred, _target), do: :erlang.nif_error(:nif_not_loaded)
  def nif_mse_loss(_pred, _target), do: :erlang.nif_error(:nif_not_loaded)
  def nif_dropout(_tensor, _prob), do: :erlang.nif_error(:nif_not_loaded)
end
