defmodule ExBurn.Nif do
  @moduledoc """
  Rustler NIF stubs for interfacing with the Burn deep learning framework.
  """

  use Rustler,
    otp_app: :ex_burn,
    crate: :ex_burn_nif

  # ── Tensor Creation ──────────────────────────────────────────────

  def new_tensor(_data, _shape, _type), do: :erlang.nif_error(:nif_not_loaded)
  def empty_tensor(_shape, _type), do: :erlang.nif_error(:nif_not_loaded)
  def zeros_tensor(_shape, _type), do: :erlang.nif_error(:nif_not_loaded)
  def ones_tensor(_shape, _type), do: :erlang.nif_error(:nif_not_loaded)

  # ── Tensor Inspection ────────────────────────────────────────────

  def tensor_shape(_tensor), do: :erlang.nif_error(:nif_not_loaded)
  def tensor_dtype(_tensor), do: :erlang.nif_error(:nif_not_loaded)
  def tensor_to_binary(_tensor), do: :erlang.nif_error(:nif_not_loaded)
  def tensor_numel(_tensor), do: :erlang.nif_error(:nif_not_loaded)

  # ── Element-wise Arithmetic ──────────────────────────────────────

  def add_tensor(_a, _b), do: :erlang.nif_error(:nif_not_loaded)
  def sub_tensor(_a, _b), do: :erlang.nif_error(:nif_not_loaded)
  def mul_tensor(_a, _b), do: :erlang.nif_error(:nif_not_loaded)
  def div_tensor(_a, _b), do: :erlang.nif_error(:nif_not_loaded)
  def neg_tensor(_a), do: :erlang.nif_error(:nif_not_loaded)
  def abs_tensor(_a), do: :erlang.nif_error(:nif_not_loaded)
  def exp_tensor(_a), do: :erlang.nif_error(:nif_not_loaded)
  def log_tensor(_a), do: :erlang.nif_error(:nif_not_loaded)
  def sqrt_tensor(_a), do: :erlang.nif_error(:nif_not_loaded)
  def pow_tensor(_a, _exp), do: :erlang.nif_error(:nif_not_loaded)
  def sigmoid_tensor(_a), do: :erlang.nif_error(:nif_not_loaded)
  def tanh_tensor(_a), do: :erlang.nif_error(:nif_not_loaded)
  def relu_tensor(_a), do: :erlang.nif_error(:nif_not_loaded)

  # ── Reductions ───────────────────────────────────────────────────

  def sum_tensor(_a), do: :erlang.nif_error(:nif_not_loaded)
  def mean_tensor(_a), do: :erlang.nif_error(:nif_not_loaded)
  def max_tensor(_a), do: :erlang.nif_error(:nif_not_loaded)
  def min_tensor(_a), do: :erlang.nif_error(:nif_not_loaded)

  # ── Linear Algebra ───────────────────────────────────────────────

  def matmul_tensor(_a, _b), do: :erlang.nif_error(:nif_not_loaded)
  def transpose_tensor(_a), do: :erlang.nif_error(:nif_not_loaded)
  def dot_tensor(_a, _b), do: :erlang.nif_error(:nif_not_loaded)

  # ── Shape Manipulation ───────────────────────────────────────────

  def reshape_tensor(_a, _shape), do: :erlang.nif_error(:nif_not_loaded)
  def broadcast_tensor(_a, _shape), do: :erlang.nif_error(:nif_not_loaded)
  def concat_tensor(_a, _b), do: :erlang.nif_error(:nif_not_loaded)

  # ── Autograd / Backward ──────────────────────────────────────────

  def backward_tensor(_a), do: :erlang.nif_error(:nif_not_loaded)
  def grad_tensor(_tensor, _var), do: :erlang.nif_error(:nif_not_loaded)

  # ── Device Management ────────────────────────────────────────────

  def gpu_available, do: :erlang.nif_error(:nif_not_loaded)
  def device_name, do: :erlang.nif_error(:nif_not_loaded)
  def to_gpu(_tensor), do: :erlang.nif_error(:nif_not_loaded)
  def to_cpu(_tensor), do: :erlang.nif_error(:nif_not_loaded)

  # ── Memory Management ────────────────────────────────────────────

  def free_tensor(_tensor), do: :erlang.nif_error(:nif_not_loaded)

  # ── Tensor Creation (cont.) ──────────────────────────────────────

  def eye_tensor(_size, _type), do: :erlang.nif_error(:nif_not_loaded)
  def iota_tensor(_shape, _axis, _type), do: :erlang.nif_error(:nif_not_loaded)

  # ── Neural Network Operations ────────────────────────────────────

  def softmax_tensor(_a, _dim), do: :erlang.nif_error(:nif_not_loaded)
  def layer_norm_tensor(_a, _dim, _eps), do: :erlang.nif_error(:nif_not_loaded)

  # ── Loss Functions ──────────────────────────────────────────────

  def cross_entropy_loss(_pred, _target), do: :erlang.nif_error(:nif_not_loaded)
  def mse_loss(_pred, _target), do: :erlang.nif_error(:nif_not_loaded)

  # ── Regularization ──────────────────────────────────────────────

  def dropout(_tensor, _prob), do: :erlang.nif_error(:nif_not_loaded)
end
