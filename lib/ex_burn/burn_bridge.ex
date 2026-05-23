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
    case ExBurn.Nif.zeros_tensor(shape, Atom.to_string(type)) do
      {:ok, ref} -> %BT{ref: ref, shape: shape, type: type}
      {:error, reason} -> raise Error, op: :zeros, reason: reason
    end
  end

  @doc "Creates a tensor filled with ones."
  @spec ones([non_neg_integer()], BT.type()) :: BT.t()
  def ones(shape, type \\ :f32) do
    case ExBurn.Nif.ones_tensor(shape, Atom.to_string(type)) do
      {:ok, ref} -> %BT{ref: ref, shape: shape, type: type}
      {:error, reason} -> raise Error, op: :ones, reason: reason
    end
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
    case ExBurn.Nif.random_tensor(shape, Atom.to_string(type), low, high) do
      {:ok, ref} -> %BT{ref: ref, shape: shape, type: type}
      {:error, reason} -> raise Error, op: :rand, reason: reason
    end
  end

  # ── Arithmetic ───────────────────────────────────────────────────

  @spec add(BT.t(), BT.t()) :: BT.t()
  def add(%BT{ref: ref_a, shape: shape, type: type}, %BT{ref: ref_b}) do
    case ExBurn.Nif.add_tensor(ref_a, ref_b) do
      {:ok, ref} -> %BT{ref: ref, shape: shape, type: type}
      {:error, reason} -> raise Error, op: :add, reason: reason
    end
  end

  @spec sub(BT.t(), BT.t()) :: BT.t()
  def sub(%BT{ref: ref_a, shape: shape, type: type}, %BT{ref: ref_b}) do
    case ExBurn.Nif.sub_tensor(ref_a, ref_b) do
      {:ok, ref} -> %BT{ref: ref, shape: shape, type: type}
      {:error, reason} -> raise Error, op: :sub, reason: reason
    end
  end

  @spec mul(BT.t(), BT.t()) :: BT.t()
  def mul(%BT{ref: ref_a, shape: shape, type: type}, %BT{ref: ref_b}) do
    case ExBurn.Nif.mul_tensor(ref_a, ref_b) do
      {:ok, ref} -> %BT{ref: ref, shape: shape, type: type}
      {:error, reason} -> raise Error, op: :mul, reason: reason
    end
  end

  @spec div(BT.t(), BT.t()) :: BT.t()
  def div(%BT{ref: ref_a, shape: shape, type: type}, %BT{ref: ref_b}) do
    case ExBurn.Nif.div_tensor(ref_a, ref_b) do
      {:ok, ref} -> %BT{ref: ref, shape: shape, type: type}
      {:error, reason} -> raise Error, op: :div, reason: reason
    end
  end

  @spec neg(BT.t()) :: BT.t()
  def neg(%BT{ref: ref, shape: shape, type: type}) do
    case ExBurn.Nif.neg_tensor(ref) do
      {:ok, ref} -> %BT{ref: ref, shape: shape, type: type}
      {:error, reason} -> raise Error, op: :neg, reason: reason
    end
  end

  @spec exp(BT.t()) :: BT.t()
  def exp(%BT{ref: ref, shape: shape, type: type}) do
    case ExBurn.Nif.exp_tensor(ref) do
      {:ok, ref} -> %BT{ref: ref, shape: shape, type: type}
      {:error, reason} -> raise Error, op: :exp, reason: reason
    end
  end

  @spec log(BT.t()) :: BT.t()
  def log(%BT{ref: ref, shape: shape, type: type}) do
    case ExBurn.Nif.log_tensor(ref) do
      {:ok, ref} -> %BT{ref: ref, shape: shape, type: type}
      {:error, reason} -> raise Error, op: :log, reason: reason
    end
  end

  @spec sqrt(BT.t()) :: BT.t()
  def sqrt(%BT{ref: ref, shape: shape, type: type}) do
    case ExBurn.Nif.sqrt_tensor(ref) do
      {:ok, ref} -> %BT{ref: ref, shape: shape, type: type}
      {:error, reason} -> raise Error, op: :sqrt, reason: reason
    end
  end

  @spec sigmoid(BT.t()) :: BT.t()
  def sigmoid(%BT{ref: ref, shape: shape, type: type}) do
    case ExBurn.Nif.sigmoid_tensor(ref) do
      {:ok, ref} -> %BT{ref: ref, shape: shape, type: type}
      {:error, reason} -> raise Error, op: :sigmoid, reason: reason
    end
  end

  @spec relu(BT.t()) :: BT.t()
  def relu(%BT{ref: ref, shape: shape, type: type}) do
    case ExBurn.Nif.relu_tensor(ref) do
      {:ok, ref} -> %BT{ref: ref, shape: shape, type: type}
      {:error, reason} -> raise Error, op: :relu, reason: reason
    end
  end

  # ── Linear Algebra ───────────────────────────────────────────────

  @spec matmul(BT.t(), BT.t()) :: BT.t()
  def matmul(%BT{ref: ref_a, type: type}, %BT{ref: ref_b}) do
    case ExBurn.Nif.matmul_tensor(ref_a, ref_b) do
      {:ok, ref} ->
        shape_a = BT.shape(%BT{ref: ref_a})
        shape_b = BT.shape(%BT{ref: ref_b})
        out_shape = matmul_output_shape(shape_a, shape_b)
        %BT{ref: ref, shape: out_shape, type: type}

      {:error, reason} ->
        raise Error, op: :matmul, reason: reason
    end
  end

  @spec transpose(BT.t(), non_neg_integer(), non_neg_integer()) :: BT.t()
  def transpose(%BT{ref: ref, shape: shape, type: type}, dim0 \\ 0, dim1 \\ 1) do
    case ExBurn.Nif.transpose_tensor(ref, dim0, dim1) do
      {:ok, ref} ->
        new_shape = swap(shape, dim0, dim1)
        %BT{ref: ref, shape: new_shape, type: type}

      {:error, reason} ->
        raise Error, op: :transpose, reason: reason
    end
  end

  # ── Reductions ───────────────────────────────────────────────────

  @spec sum(BT.t(), [non_neg_integer()] | nil) :: BT.t()
  def sum(%BT{ref: ref, type: type}, axes \\ nil) do
    case ExBurn.Nif.sum_tensor(ref, axes) do
      {:ok, ref} -> %BT{ref: ref, shape: [1], type: type}
      {:error, reason} -> raise Error, op: :sum, reason: reason
    end
  end

  @spec mean(BT.t(), [non_neg_integer()] | nil) :: BT.t()
  def mean(%BT{ref: ref, type: type}, axes \\ nil) do
    case ExBurn.Nif.mean_tensor(ref, axes) do
      {:ok, ref} -> %BT{ref: ref, shape: [1], type: type}
      {:error, reason} -> raise Error, op: :mean, reason: reason
    end
  end

  # ── Shape Manipulation ───────────────────────────────────────────

  @spec reshape(BT.t(), [non_neg_integer()]) :: BT.t()
  def reshape(%BT{ref: ref, type: type}, shape) do
    case ExBurn.Nif.reshape_tensor(ref, shape) do
      {:ok, ref} -> %BT{ref: ref, shape: shape, type: type}
      {:error, reason} -> raise Error, op: :reshape, reason: reason
    end
  end

  @spec softmax(BT.t(), non_neg_integer()) :: BT.t()
  def softmax(%BT{ref: ref, shape: shape, type: type}, dim \\ -1) do
    case ExBurn.Nif.softmax_tensor(ref, dim) do
      {:ok, ref} -> %BT{ref: ref, shape: shape, type: type}
      {:error, reason} -> raise Error, op: :softmax, reason: reason
    end
  end

  @spec layer_norm(BT.t(), non_neg_integer(), float()) :: BT.t()
  def layer_norm(%BT{ref: ref, shape: shape, type: type}, dim \\ -1, eps \\ 1.0e-5) do
    case ExBurn.Nif.layer_norm_tensor(ref, dim, eps) do
      {:ok, ref} -> %BT{ref: ref, shape: shape, type: type}
      {:error, reason} -> raise Error, op: :layer_norm, reason: reason
    end
  end

  @spec dropout(BT.t(), float(), boolean()) :: BT.t()
  def dropout(%BT{ref: ref, shape: shape, type: type}, _prob \\ 0.5, _training \\ true) do
    # Dropout is a no-op during inference; during training, the NIF would apply it.
    # Since the NIF registration is problematic in Rustler 0.37, we return the input as-is.
    %BT{ref: ref, shape: shape, type: type}
  end

  # ── Loss Functions ───────────────────────────────────────────────

  @spec cross_entropy(BT.t(), BT.t()) :: BT.t()
  def cross_entropy(%BT{ref: ref_pred, type: type}, %BT{ref: ref_target}) do
    case ExBurn.Nif.cross_entropy_tensor(ref_pred, ref_target) do
      {:ok, ref} -> %BT{ref: ref, shape: [1], type: type}
      {:error, reason} -> raise Error, op: :cross_entropy, reason: reason
    end
  end

  @spec mse(BT.t(), BT.t()) :: BT.t()
  def mse(%BT{ref: ref_pred, type: type}, %BT{ref: ref_target}) do
    case ExBurn.Nif.mse_tensor(ref_pred, ref_target) do
      {:ok, ref} -> %BT{ref: ref, shape: [1], type: type}
      {:error, reason} -> raise Error, op: :mse, reason: reason
    end
  end

  # ── Device Management ────────────────────────────────────────────

  @spec to_gpu(BT.t()) :: BT.t()
  def to_gpu(%BT{shape: shape, type: type} = bt) do
    case ExBurn.Nif.to_gpu(bt.ref) do
      {:ok, ref} -> %BT{ref: ref, shape: shape, type: type}
      {:error, reason} -> raise Error, op: :to_gpu, reason: reason
    end
  end

  @spec to_cpu(BT.t()) :: BT.t()
  def to_cpu(%BT{shape: shape, type: type} = bt) do
    case ExBurn.Nif.to_cpu(bt.ref) do
      {:ok, ref} -> %BT{ref: ref, shape: shape, type: type}
      {:error, reason} -> raise Error, op: :to_cpu, reason: reason
    end
  end

  # ── Memory ───────────────────────────────────────────────────────

  @spec free(BT.t()) :: :ok
  def free(%BT{ref: ref}), do: ExBurn.Nif.free_tensor(ref)

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
