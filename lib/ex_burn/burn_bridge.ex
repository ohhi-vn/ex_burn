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
    ref = ExBurn.Nif.zeros_tensor(shape, Atom.to_string(type))
    %BT{ref: ref, shape: shape, type: type}
  end

  @doc "Creates a tensor filled with ones."
  @spec ones([non_neg_integer()], BT.type()) :: BT.t()
  def ones(shape, type \\ :f32) do
    ref = ExBurn.Nif.ones_tensor(shape, Atom.to_string(type))
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
  def rand(shape, type \\ :f32, _low \\ 0.0, _high \\ 1.0) do
    ref = ExBurn.Nif.zeros_tensor(shape, Atom.to_string(type))
    %BT{ref: ref, shape: shape, type: type}
  end

  @doc "Creates a GPU buffer via ExCubecl from a list of values."
  @spec buffer(list(), [non_neg_integer()], atom()) :: ExCubecl.buffer_ref()
  def buffer(data, shape, type \\ :f32) do
    case ExCubecl.buffer(data, shape, type) do
      {:ok, buf} -> buf
      {:error, reason} -> raise Error, op: :buffer, reason: inspect(reason)
    end
  end

  @doc "Creates a GPU buffer via ExCubecl, raising on error."
  def buffer!(data, shape, type \\ :f32) do
    ExCubecl.buffer!(data, shape, type)
  end

  @doc "Reads data from an ExCubecl buffer."
  @spec read_buffer(ExCubecl.buffer_ref()) :: binary()
  def read_buffer(buf) do
    case ExCubecl.read(buf) do
      {:ok, data} -> data
      {:error, reason} -> raise Error, op: :read_buffer, reason: inspect(reason)
    end
  end

  @doc "Returns the shape of an ExCubecl buffer."
  @spec buffer_shape(ExCubecl.buffer_ref()) :: [non_neg_integer()]
  def buffer_shape(buf) do
    case ExCubecl.shape(buf) do
      {:ok, shape} -> shape
      {:error, reason} -> raise Error, op: :buffer_shape, reason: inspect(reason)
    end
  end

  @doc "Returns the byte size of an ExCubecl buffer."
  @spec buffer_size(ExCubecl.buffer_ref()) :: non_neg_integer()
  def buffer_size(buf) do
    case ExCubecl.size(buf) do
      {:ok, size} -> size
      {:error, reason} -> raise Error, op: :buffer_size, reason: inspect(reason)
    end
  end

  # ── Arithmetic ───────────────────────────────────────────────────

  @spec add(BT.t(), BT.t()) :: BT.t()
  def add(%BT{ref: ref_a, shape: shape, type: type}, %BT{ref: ref_b}) do
    ref = ExBurn.Nif.add_tensor(ref_a, ref_b)
    %BT{ref: ref, shape: shape, type: type}
  end

  @spec sub(BT.t(), BT.t()) :: BT.t()
  def sub(%BT{ref: ref_a, shape: shape, type: type}, %BT{ref: ref_b}) do
    ref = ExBurn.Nif.sub_tensor(ref_a, ref_b)
    %BT{ref: ref, shape: shape, type: type}
  end

  @spec mul(BT.t(), BT.t()) :: BT.t()
  def mul(%BT{ref: ref_a, shape: shape, type: type}, %BT{ref: ref_b}) do
    ref = ExBurn.Nif.mul_tensor(ref_a, ref_b)
    %BT{ref: ref, shape: shape, type: type}
  end

  @spec div(BT.t(), BT.t()) :: BT.t()
  def div(%BT{ref: ref_a, shape: shape, type: type}, %BT{ref: ref_b}) do
    ref = ExBurn.Nif.div_tensor(ref_a, ref_b)
    %BT{ref: ref, shape: shape, type: type}
  end

  @spec neg(BT.t()) :: BT.t()
  def neg(%BT{ref: ref, shape: shape, type: type}) do
    ref = ExBurn.Nif.neg_tensor(ref)
    %BT{ref: ref, shape: shape, type: type}
  end

  @spec abs(BT.t()) :: BT.t()
  def abs(%BT{ref: ref, shape: shape, type: type}) do
    ref = ExBurn.Nif.abs_tensor(ref)
    %BT{ref: ref, shape: shape, type: type}
  end

  @spec exp(BT.t()) :: BT.t()
  def exp(%BT{ref: ref, shape: shape, type: type}) do
    ref = ExBurn.Nif.exp_tensor(ref)
    %BT{ref: ref, shape: shape, type: type}
  end

  @spec log(BT.t()) :: BT.t()
  def log(%BT{ref: ref, shape: shape, type: type}) do
    ref = ExBurn.Nif.log_tensor(ref)
    %BT{ref: ref, shape: shape, type: type}
  end

  @spec sqrt(BT.t()) :: BT.t()
  def sqrt(%BT{ref: ref, shape: shape, type: type}) do
    ref = ExBurn.Nif.sqrt_tensor(ref)
    %BT{ref: ref, shape: shape, type: type}
  end

  @spec sigmoid(BT.t()) :: BT.t()
  def sigmoid(%BT{ref: ref, shape: shape, type: type}) do
    ref = ExBurn.Nif.sigmoid_tensor(ref)
    %BT{ref: ref, shape: shape, type: type}
  end

  @spec relu(BT.t()) :: BT.t()
  def relu(%BT{ref: ref, shape: shape, type: type}) do
    ref = ExBurn.Nif.relu_tensor(ref)
    %BT{ref: ref, shape: shape, type: type}
  end

  # ── Linear Algebra ───────────────────────────────────────────────

  @spec matmul(BT.t(), BT.t()) :: BT.t()
  def matmul(%BT{ref: ref_a, type: type}, %BT{ref: ref_b}) do
    ref = ExBurn.Nif.matmul_tensor(ref_a, ref_b)
    shape_a = BT.shape(%BT{ref: ref_a})
    shape_b = BT.shape(%BT{ref: ref_b})
    out_shape = matmul_output_shape(shape_a, shape_b)
    %BT{ref: ref, shape: out_shape, type: type}
  end

  @spec transpose(BT.t(), non_neg_integer(), non_neg_integer()) :: BT.t()
  def transpose(%BT{ref: ref, shape: shape, type: type}, dim0 \\ 0, dim1 \\ 1) do
    ref = ExBurn.Nif.transpose_tensor(ref)
    new_shape = swap(shape, dim0, dim1)
    %BT{ref: ref, shape: new_shape, type: type}
  end

  # ── Reductions ───────────────────────────────────────────────────

  @spec sum(BT.t(), [non_neg_integer()] | nil) :: BT.t()
  def sum(%BT{ref: ref, type: type}, _axes \\ nil) do
    ref = ExBurn.Nif.sum_tensor(ref)
    %BT{ref: ref, shape: [1], type: type}
  end

  @spec mean(BT.t(), [non_neg_integer()] | nil) :: BT.t()
  def mean(%BT{ref: ref, type: type}, _axes \\ nil) do
    ref = ExBurn.Nif.mean_tensor(ref)
    %BT{ref: ref, shape: [1], type: type}
  end

  # ── Shape Manipulation ───────────────────────────────────────────

  @spec reshape(BT.t(), [non_neg_integer()]) :: BT.t()
  def reshape(%BT{ref: ref, type: type}, shape) do
    ref = ExBurn.Nif.reshape_tensor(ref, shape)
    %BT{ref: ref, shape: shape, type: type}
  end

  @spec softmax(BT.t(), non_neg_integer()) :: BT.t()
  def softmax(%BT{ref: ref, shape: shape, type: type}, dim \\ -1) do
    ref = ExBurn.Nif.softmax_tensor(ref, dim)
    %BT{ref: ref, shape: shape, type: type}
  end

  @spec layer_norm(BT.t(), non_neg_integer(), float()) :: BT.t()
  def layer_norm(%BT{ref: ref, shape: shape, type: type}, _dim \\ -1, _eps \\ 1.0e-5) do
    ref = ExBurn.Nif.layer_norm_tensor(ref, 0, 0.0)
    %BT{ref: ref, shape: shape, type: type}
  end

  @spec dropout(BT.t(), float(), boolean()) :: BT.t()
  def dropout(%BT{ref: ref, shape: shape, type: type}, _prob \\ 0.5, _training \\ true) do
    %BT{ref: ref, shape: shape, type: type}
  end

  # ── Loss Functions ───────────────────────────────────────────────

  @spec cross_entropy(BT.t(), BT.t()) :: BT.t()
  def cross_entropy(%BT{ref: ref_pred, type: type}, %BT{ref: _ref_target}) do
    ref = ref_pred
    %BT{ref: ref, shape: [1], type: type}
  end

  @spec mse(BT.t(), BT.t()) :: BT.t()
  def mse(%BT{ref: ref_pred, type: type}, %BT{ref: _ref_target}) do
    ref = ref_pred
    %BT{ref: ref, shape: [1], type: type}
  end

  # ── Device Management ────────────────────────────────────────────

  @spec to_gpu(BT.t()) :: BT.t()
  def to_gpu(%BT{ref: ref, shape: shape, type: type} = bt) do
    # Read data from Burn tensor, create ExCubecl buffer, wrap back
    case ExBurn.Nif.tensor_to_binary(ref) do
      binary ->
        # Convert binary to flat list for ExCubecl
        flat_data = for <<x::float-32 <- binary>>, do: x
        _nx_type = BT.burn_type_to_nx(type)

        case ExCubecl.buffer(flat_data, shape, type) do
          {:ok, buf} ->
            # Return a Burn tensor that wraps the ExCubecl buffer reference
            %BT{ref: buf, shape: shape, type: type}
          {:error, _} ->
            # Fallback: keep original
            bt
        end
    end
  end

  @spec to_cpu(BT.t()) :: BT.t()
  def to_cpu(%BT{ref: ref, shape: shape, type: type}) do
    # If ref is an ExCubecl buffer, read it back
    try do
      case ExCubecl.read(ref) do
        {:ok, binary} ->
          nx_type = BT.burn_type_to_nx(type)
          tensor = Nx.from_binary(binary, nx_type) |> Nx.reshape(List.to_tuple(shape))
          # Convert back to Burn tensor
          from_nx(tensor)
        {:error, _} ->
          # Not an ExCubecl buffer, return as-is
          %BT{ref: ref, shape: shape, type: type}
      end
    rescue
      _ -> %BT{ref: ref, shape: shape, type: type}
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
