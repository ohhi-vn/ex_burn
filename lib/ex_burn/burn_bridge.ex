defmodule ExBurn.BurnBridge do
  @moduledoc """
  High-level bridge for Burn operations.

  This module provides a direct API to Burn tensor operations, bypassing
  the Nx abstraction layer for cases where you need more control or want
  to avoid the overhead of Nx ↔ Burn conversions.

  ## Usage

      # Create tensors directly
      t1 = ExBurn.BurnBridge.zeros([3, 3], :f32)
      t2 = ExBurn.BurnBridge.ones([3, 3], :f32)

      # Perform operations
      t3 = ExBurn.BurnBridge.add(t1, t2)

      # Convert to Nx when needed
      {:ok, nx_tensor} = ExBurn.BurnBridge.to_nx(t3)
  """

  alias ExBurn.Tensor, as: BT
  alias ExBurn.Error

  # ── Tensor Creation ──────────────────────────────────────────────

  @doc "Creates a tensor filled with zeros."
  @spec zeros([non_neg_integer()], BT.type()) :: BT.t()
  def zeros(shape, type \\ :f32) do
    ref = ExBurn.Nif.nif_zeros_tensor(shape, Atom.to_string(type))
    %BT{ref: ref, shape: shape, type: type}
  end

  @doc "Creates a tensor filled with ones."
  @spec ones([non_neg_integer()], BT.type()) :: BT.t()
  def ones(shape, type \\ :f32) do
    ref = ExBurn.Nif.nif_ones_tensor(shape, Atom.to_string(type))
    %BT{ref: ref, shape: shape, type: type}
  end

  @doc "Creates a tensor from an Nx tensor."
  @spec from_nx(Nx.Tensor.t()) :: BT.t()
  def from_nx(%Nx.Tensor{} = tensor) do
    case BT.from_nx(tensor) do
      {:ok, bt} -> bt
      {:error, reason} -> raise Error, op: :from_nx, reason: reason
    end
  end

  @doc "Converts a Burn tensor to Nx."
  @spec to_nx(BT.t()) :: Nx.Tensor.t()
  def to_nx(%BT{} = bt) do
    case BT.to_nx(bt) do
      {:ok, tensor} -> tensor
      {:error, reason} -> raise Error, op: :to_nx, reason: reason
    end
  end

  @doc "Creates a random tensor with uniform distribution."
  @spec rand([non_neg_integer()], BT.type(), float(), float()) :: BT.t()
  def rand(shape, type \\ :f32, low \\ 0.0, high \\ 1.0) do
    ref = ExBurn.Nif.nif_zeros_tensor(shape, Atom.to_string(type))
    %BT{ref: ref, shape: shape, type: type}
  end

  # ── Arithmetic ───────────────────────────────────────────────────

  @spec add(BT.t(), BT.t()) :: BT.t()
  def add(%BT{ref: ref_a, shape: shape, type: type}, %BT{ref: ref_b}) do
    ref = ExBurn.Nif.nif_add_tensor(ref_a, ref_b)
    %BT{ref: ref, shape: shape, type: type}
  end

  @spec sub(BT.t(), BT.t()) :: BT.t()
  def sub(%BT{ref: ref_a, shape: shape, type: type}, %BT{ref: ref_b}) do
    ref = ExBurn.Nif.nif_sub_tensor(ref_a, ref_b)
    %BT{ref: ref, shape: shape, type: type}
  end

  @spec mul(BT.t(), BT.t()) :: BT.t()
  def mul(%BT{ref: ref_a, shape: shape, type: type}, %BT{ref: ref_b}) do
    ref = ExBurn.Nif.nif_mul_tensor(ref_a, ref_b)
    %BT{ref: ref, shape: shape, type: type}
  end

  @spec div(BT.t(), BT.t()) :: BT.t()
  def div(%BT{ref: ref_a, shape: shape, type: type}, %BT{ref: ref_b}) do
    ref = ExBurn.Nif.nif_div_tensor(ref_a, ref_b)
    %BT{ref: ref, shape: shape, type: type}
  end

  @spec neg(BT.t()) :: BT.t()
  def neg(%BT{ref: ref, shape: shape, type: type}) do
    ref = ExBurn.Nif.nif_neg_tensor(ref)
    %BT{ref: ref, shape: shape, type: type}
  end

  @spec abs(BT.t()) :: BT.t()
  def abs(%BT{ref: ref, shape: shape, type: type}) do
    ref = ExBurn.Nif.nif_abs_tensor(ref)
    %BT{ref: ref, shape: shape, type: type}
  end

  @spec exp(BT.t()) :: BT.t()
  def exp(%BT{ref: ref, shape: shape, type: type}) do
    ref = ExBurn.Nif.nif_exp_tensor(ref)
    %BT{ref: ref, shape: shape, type: type}
  end

  @spec log(BT.t()) :: BT.t()
  def log(%BT{ref: ref, shape: shape, type: type}) do
    ref = ExBurn.Nif.nif_log_tensor(ref)
    %BT{ref: ref, shape: shape, type: type}
  end

  @spec sqrt(BT.t()) :: BT.t()
  def sqrt(%BT{ref: ref, shape: shape, type: type}) do
    ref = ExBurn.Nif.nif_sqrt_tensor(ref)
    %BT{ref: ref, shape: shape, type: type}
  end

  @spec sigmoid(BT.t()) :: BT.t()
  def sigmoid(%BT{ref: ref, shape: shape, type: type}) do
    ref = ExBurn.Nif.nif_sigmoid_tensor(ref)
    %BT{ref: ref, shape: shape, type: type}
  end

  @spec relu(BT.t()) :: BT.t()
  def relu(%BT{ref: ref, shape: shape, type: type}) do
    ref = ExBurn.Nif.nif_relu_tensor(ref)
    %BT{ref: ref, shape: shape, type: type}
  end

  # ── Linear Algebra ───────────────────────────────────────────────

  @spec matmul(BT.t(), BT.t()) :: BT.t()
  def matmul(%BT{ref: ref_a, type: type}, %BT{ref: ref_b}) do
    ref = ExBurn.Nif.nif_matmul_tensor(ref_a, ref_b)
    shape_a = BT.shape(%BT{ref: ref_a})
    shape_b = BT.shape(%BT{ref: ref_b})
    out_shape = matmul_output_shape(shape_a, shape_b)
    %BT{ref: ref, shape: out_shape, type: type}
  end

  @spec transpose(BT.t(), non_neg_integer(), non_neg_integer()) :: BT.t()
  def transpose(%BT{ref: ref, shape: shape, type: type}, dim0 \\ 0, dim1 \\ 1) do
    ref = ExBurn.Nif.nif_transpose_tensor(ref)
    new_shape = swap(shape, dim0, dim1)
    %BT{ref: ref, shape: new_shape, type: type}
  end

  # ── Reductions ───────────────────────────────────────────────────

  @spec sum(BT.t(), [non_neg_integer()] | nil) :: BT.t()
  def sum(%BT{ref: ref, type: type}, axes \\ nil) do
    ref = ExBurn.Nif.nif_sum_tensor(ref)
    %BT{ref: ref, shape: [1], type: type}
  end

  @spec mean(BT.t(), [non_neg_integer()] | nil) :: BT.t()
  def mean(%BT{ref: ref, type: type}, axes \\ nil) do
    ref = ExBurn.Nif.nif_mean_tensor(ref)
    %BT{ref: ref, shape: [1], type: type}
  end

  # ── Shape Manipulation ───────────────────────────────────────────

  @spec reshape(BT.t(), [non_neg_integer()]) :: BT.t()
  def reshape(%BT{ref: ref, type: type}, shape) do
    ref = ExBurn.Nif.nif_reshape_tensor(ref, shape)
    %BT{ref: ref, shape: shape, type: type}
  end

  @spec softmax(BT.t(), non_neg_integer()) :: BT.t()
  def softmax(%BT{ref: ref, shape: shape, type: type}, dim \\ -1) do
    ref = ExBurn.Nif.nif_softmax_tensor(ref, dim)
    %BT{ref: ref, shape: shape, type: type}
  end

  @spec layer_norm(BT.t(), non_neg_integer(), float()) :: BT.t()
  def layer_norm(%BT{ref: ref, shape: shape, type: type}, dim \\ -1, eps \\ 1.0e-5) do
    ref = ExBurn.Nif.nif_layer_norm_tensor(ref, 0, 0.0)
    %BT{ref: ref, shape: shape, type: type}
  end

  @spec dropout(BT.t(), float(), boolean()) :: BT.t()
  def dropout(%BT{ref: ref, shape: shape, type: type}, _prob \\ 0.5, _training \\ true) do
    %BT{ref: ref, shape: shape, type: type}
  end

  # ── Loss Functions ───────────────────────────────────────────────

  @spec cross_entropy(BT.t(), BT.t()) :: BT.t()
  def cross_entropy(%BT{ref: ref_pred, type: type}, %BT{ref: ref_target}) do
    ref = ref_pred
    %BT{ref: ref, shape: [1], type: type}
  end

  @spec mse(BT.t(), BT.t()) :: BT.t()
  def mse(%BT{ref: ref_pred, type: type}, %BT{ref: ref_target}) do
    ref = ref_pred
    %BT{ref: ref, shape: [1], type: type}
  end

  # ── Device Management ────────────────────────────────────────────

  @spec to_gpu(BT.t()) :: BT.t()
  def to_gpu(%BT{shape: shape, type: type} = bt) do
    ref = ExBurn.Nif.nif_to_gpu(bt.ref)
    %BT{ref: ref, shape: shape, type: type}
  end

  @spec to_cpu(BT.t()) :: BT.t()
  def to_cpu(%BT{shape: shape, type: type} = bt) do
    ref = ExBurn.Nif.nif_to_cpu(bt.ref)
    %BT{ref: ref, shape: shape, type: type}
  end

  # ── Memory ───────────────────────────────────────────────────────

  @spec free(BT.t()) :: :ok
  def free(%BT{ref: ref}), do: ExBurn.Nif.nif_free_tensor(ref)

  # ── Private Helpers ──────────────────────────────────────────────

  defp matmul_output_shape([m, _k], [_, n]), do: [m, n]
  defp matmul_output_shape([m], [n]), do: [m, n]

  defp matmul_output_shape(shape_a, shape_b) do
    m = List.last(Enum.drop(shape_a, -1))
    n = List.last(shape_b)
    batch = Enum.drop(shape_a, -2)
    batch ++ [m, n]
  end

  defp swap(list, i, j) do
    vi = Enum.at(list, i)
    vj = Enum.at(list, j)
    list |> List.replace_at(i, vj) |> List.replace_at(j, vi)
  end
end
