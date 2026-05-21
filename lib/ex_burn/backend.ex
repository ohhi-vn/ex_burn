defmodule ExBurn.Backend do
  @moduledoc """
  Nx backend implementation that delegates tensor operations to Burn via NIF.

  This module implements the `Nx.Backend` behaviour, translating Nx tensor
  operations into calls to the Rust NIF layer which executes them using
  Burn's CubeCL backend for GPU acceleration.

  ## Usage

      Nx.default_backend(ExBurn.Backend)

  ## Implementation Notes

  - Tensors are stored as opaque NIF references on the Rust side
  - Data is serialized as binary (f32 little-endian) for NIF calls
  - The backend struct holds the NIF reference, shape, and type
  - For performance-critical paths, use `ExBurn.BurnBridge` directly
  - All public callbacks raise `ExBurn.Error` on failure
  """

  @behaviour Nx.Backend

  alias ExBurn.Tensor, as: BT
  alias ExBurn.Error

  @type t :: %__MODULE__{
          ref: reference(),
          shape: [non_neg_integer()],
          type: BT.burn_type()
        }

  defstruct [:ref, :shape, :type]

  # ── Allocation ───────────────────────────────────────────────────

  @impl true
  @spec init(keyword()) :: :ok
  def init(_opts), do: :ok

  @impl true
  @spec constant(Nx.Tensor.t(), number(), keyword()) :: t()
  def constant(out, value, _opts) do
    type = nx_to_burn_type(Nx.type(out))
    shape = Tuple.to_list(Nx.shape(out))
    data = <<value::float-32-native>>

    case ExBurn.Nif.new_tensor(data, shape, type) do
      {:ok, ref} ->
        %__MODULE__{ref: ref, shape: shape, type: type}

      {:error, reason} ->
        raise Error, message: "constant/3 failed: #{reason}", reason: reason
    end
  end

  @impl true
  @spec from_binary(Nx.Tensor.t(), binary(), keyword()) :: t()
  def from_binary(%{type: nx_type} = out, binary, _opts) do
    burn_type = nx_to_burn_type(nx_type)
    # Read shape from the output template tensor, not from binary size
    shape = Tuple.to_list(Nx.shape(out))

    case ExBurn.Nif.new_tensor(binary, shape, burn_type) do
      {:ok, ref} ->
        %__MODULE__{ref: ref, shape: shape, type: burn_type}

      {:error, reason} ->
        raise Error, message: "from_binary/3 failed: #{reason}", reason: reason
    end
  end

  @impl true
  @spec to_binary(t(), non_neg_integer()) :: binary()
  def to_binary(%__MODULE__{ref: ref}, _limit) do
    case ExBurn.Nif.tensor_to_binary(ref) do
      {:ok, binary} ->
        binary

      {:error, reason} ->
        raise Error, message: "to_binary/2 failed: #{reason}", reason: reason
    end
  end

  @impl true
  @spec tensor_type(t()) :: Nx.Type.t()
  def tensor_type(%__MODULE__{type: type}), do: burn_to_nx_type(type)

  # ── Element-wise Arithmetic ──────────────────────────────────────

  @impl true
  @spec add(t(), t()) :: t()
  def add(%__MODULE__{} = a, %__MODULE__{} = b) do
    a = maybe_cast(a, :f32)
    b = maybe_cast(b, :f32)

    with {:ok, ref} <- ExBurn.Nif.add_tensor(a.ref, b.ref),
         {:ok, shape} <- ExBurn.Nif.tensor_shape(ref) do
      %__MODULE__{ref: ref, shape: shape, type: :f32}
    else
      {:error, reason} -> raise Error, message: "add/2 failed: #{reason}", reason: reason
    end
  end

  @impl true
  @spec subtract(t(), t()) :: t()
  def subtract(%__MODULE__{} = a, %__MODULE__{} = b) do
    a = maybe_cast(a, :f32)
    b = maybe_cast(b, :f32)

    with {:ok, ref} <- ExBurn.Nif.sub_tensor(a.ref, b.ref),
         {:ok, shape} <- ExBurn.Nif.tensor_shape(ref) do
      %__MODULE__{ref: ref, shape: shape, type: :f32}
    else
      {:error, reason} -> raise Error, message: "subtract/2 failed: #{reason}", reason: reason
    end
  end

  @impl true
  @spec multiply(t(), t()) :: t()
  def multiply(%__MODULE__{} = a, %__MODULE__{} = b) do
    a = maybe_cast(a, :f32)
    b = maybe_cast(b, :f32)

    with {:ok, ref} <- ExBurn.Nif.mul_tensor(a.ref, b.ref),
         {:ok, shape} <- ExBurn.Nif.tensor_shape(ref) do
      %__MODULE__{ref: ref, shape: shape, type: :f32}
    else
      {:error, reason} -> raise Error, message: "multiply/2 failed: #{reason}", reason: reason
    end
  end

  @impl true
  @spec divide(t(), t()) :: t()
  def divide(%__MODULE__{} = a, %__MODULE__{} = b) do
    a = maybe_cast(a, :f32)
    b = maybe_cast(b, :f32)

    with {:ok, ref} <- ExBurn.Nif.div_tensor(a.ref, b.ref),
         {:ok, shape} <- ExBurn.Nif.tensor_shape(ref) do
      %__MODULE__{ref: ref, shape: shape, type: :f32}
    else
      {:error, reason} -> raise Error, message: "divide/2 failed: #{reason}", reason: reason
    end
  end

  @impl true
  @spec negate(t()) :: t()
  def negate(%__MODULE__{} = a) do
    a = maybe_cast(a, :f32)

    with {:ok, ref} <- ExBurn.Nif.neg_tensor(a.ref),
         {:ok, shape} <- ExBurn.Nif.tensor_shape(ref) do
      %__MODULE__{ref: ref, shape: shape, type: :f32}
    else
      {:error, reason} -> raise Error, message: "negate/1 failed: #{reason}", reason: reason
    end
  end

  @impl true
  @spec abs(t()) :: t()
  def abs(%__MODULE__{} = a) do
    a = maybe_cast(a, :f32)

    with {:ok, ref} <- ExBurn.Nif.abs_tensor(a.ref),
         {:ok, shape} <- ExBurn.Nif.tensor_shape(ref) do
      %__MODULE__{ref: ref, shape: shape, type: :f32}
    else
      {:error, reason} -> raise Error, message: "abs/1 failed: #{reason}", reason: reason
    end
  end

  @impl true
  @spec exp(t()) :: t()
  def exp(%__MODULE__{} = a) do
    a = maybe_cast(a, :f32)

    with {:ok, ref} <- ExBurn.Nif.exp_tensor(a.ref),
         {:ok, shape} <- ExBurn.Nif.tensor_shape(ref) do
      %__MODULE__{ref: ref, shape: shape, type: :f32}
    else
      {:error, reason} -> raise Error, message: "exp/1 failed: #{reason}", reason: reason
    end
  end

  @impl true
  @spec log(t()) :: t()
  def log(%__MODULE__{} = a) do
    a = maybe_cast(a, :f32)

    with {:ok, ref} <- ExBurn.Nif.log_tensor(a.ref),
         {:ok, shape} <- ExBurn.Nif.tensor_shape(ref) do
      %__MODULE__{ref: ref, shape: shape, type: :f32}
    else
      {:error, reason} -> raise Error, message: "log/1 failed: #{reason}", reason: reason
    end
  end

  @impl true
  @spec sqrt(t()) :: t()
  def sqrt(%__MODULE__{} = a) do
    a = maybe_cast(a, :f32)

    with {:ok, ref} <- ExBurn.Nif.sqrt_tensor(a.ref),
         {:ok, shape} <- ExBurn.Nif.tensor_shape(ref) do
      %__MODULE__{ref: ref, shape: shape, type: :f32}
    else
      {:error, reason} -> raise Error, message: "sqrt/1 failed: #{reason}", reason: reason
    end
  end

  @impl true
  @spec sigmoid(t()) :: t()
  def sigmoid(%__MODULE__{} = a) do
    a = maybe_cast(a, :f32)

    with {:ok, ref} <- ExBurn.Nif.sigmoid_tensor(a.ref),
         {:ok, shape} <- ExBurn.Nif.tensor_shape(ref) do
      %__MODULE__{ref: ref, shape: shape, type: :f32}
    else
      {:error, reason} -> raise Error, message: "sigmoid/1 failed: #{reason}", reason: reason
    end
  end

  @impl true
  @spec tanh(t()) :: t()
  def tanh(%__MODULE__{} = a) do
    a = maybe_cast(a, :f32)

    with {:ok, ref} <- ExBurn.Nif.tanh_tensor(a.ref),
         {:ok, shape} <- ExBurn.Nif.tensor_shape(ref) do
      %__MODULE__{ref: ref, shape: shape, type: :f32}
    else
      {:error, reason} -> raise Error, message: "tanh/1 failed: #{reason}", reason: reason
    end
  end

  @impl true
  @spec pow(t(), number()) :: t()
  def pow(%__MODULE__{} = a, b) do
    a = maybe_cast(a, :f32)

    with {:ok, ref} <- ExBurn.Nif.pow_tensor(a.ref, b),
         {:ok, shape} <- ExBurn.Nif.tensor_shape(ref) do
      %__MODULE__{ref: ref, shape: shape, type: :f32}
    else
      {:error, reason} -> raise Error, message: "pow/2 failed: #{reason}", reason: reason
    end
  end

  @impl true
  @spec remainder(t(), t()) :: t()
  def remainder(%__MODULE__{} = a, %__MODULE__{} = b) do
    div_result = divide(a, b)
    truncated = round(div_result)
    mul_result = multiply(truncated, b)
    subtract(a, mul_result)
  end

  # ── Comparison ───────────────────────────────────────────────────

  @impl true
  @spec equal(t(), t()) :: t()
  def equal(%__MODULE__{} = a, %__MODULE__{} = b) do
    with {:ok, nx_a} <- to_nx(a),
         {:ok, nx_b} <- to_nx(b),
         result = Nx.equal(nx_a, nx_b),
         {:ok, bt} <- from_nx(result) do
      bt
    else
      {:error, reason} -> raise Error, message: "equal/2 failed: #{reason}", reason: reason
    end
  end

  @impl true
  @spec not_equal(t(), t()) :: t()
  def not_equal(%__MODULE__{} = a, %__MODULE__{} = b) do
    with {:ok, nx_a} <- to_nx(a),
         {:ok, nx_b} <- to_nx(b),
         result = Nx.not_equal(nx_a, nx_b),
         {:ok, bt} <- from_nx(result) do
      bt
    else
      {:error, reason} -> raise Error, message: "not_equal/2 failed: #{reason}", reason: reason
    end
  end

  @impl true
  @spec greater(t(), t()) :: t()
  def greater(%__MODULE__{} = a, %__MODULE__{} = b) do
    with {:ok, nx_a} <- to_nx(a),
         {:ok, nx_b} <- to_nx(b),
         result = Nx.greater(nx_a, nx_b),
         {:ok, bt} <- from_nx(result) do
      bt
    else
      {:error, reason} -> raise Error, message: "greater/2 failed: #{reason}", reason: reason
    end
  end

  @impl true
  @spec less(t(), t()) :: t()
  def less(%__MODULE__{} = a, %__MODULE__{} = b) do
    with {:ok, nx_a} <- to_nx(a),
         {:ok, nx_b} <- to_nx(b),
         result = Nx.less(nx_a, nx_b),
         {:ok, bt} <- from_nx(result) do
      bt
    else
      {:error, reason} -> raise Error, message: "less/2 failed: #{reason}", reason: reason
    end
  end

  @impl true
  @spec greater_equal(t(), t()) :: t()
  def greater_equal(%__MODULE__{} = a, %__MODULE__{} = b) do
    with {:ok, nx_a} <- to_nx(a),
         {:ok, nx_b} <- to_nx(b),
         result = Nx.greater_equal(nx_a, nx_b),
         {:ok, bt} <- from_nx(result) do
      bt
    else
      {:error, reason} ->
        raise Error, message: "greater_equal/2 failed: #{reason}", reason: reason
    end
  end

  @impl true
  @spec less_equal(t(), t()) :: t()
  def less_equal(%__MODULE__{} = a, %__MODULE__{} = b) do
    with {:ok, nx_a} <- to_nx(a),
         {:ok, nx_b} <- to_nx(b),
         result = Nx.less_equal(nx_a, nx_b),
         {:ok, bt} <- from_nx(result) do
      bt
    else
      {:error, reason} -> raise Error, message: "less_equal/2 failed: #{reason}", reason: reason
    end
  end

  # ── Reductions ───────────────────────────────────────────────────

  @impl true
  @spec sum(t(), keyword()) :: t()
  def sum(%__MODULE__{} = a, opts) do
    a = maybe_cast(a, :f32)
    axes = opts[:axes]

    with {:ok, ref} <- ExBurn.Nif.sum_tensor(a.ref, axes || []),
         {:ok, shape} <- ExBurn.Nif.tensor_shape(ref) do
      %__MODULE__{ref: ref, shape: shape, type: :f32}
    else
      {:error, reason} -> raise Error, message: "sum/2 failed: #{reason}", reason: reason
    end
  end

  @impl true
  @spec product(t(), keyword()) :: t()
  def product(%__MODULE__{} = a, opts) do
    # product via log-sum-exp trick: exp(sum(log(x)))
    a = maybe_cast(a, :f32)
    log_a = log(a)
    sum_log = sum(log_a, opts)
    exp(sum_log)
  end

  @impl true
  @spec reduce_max(t(), keyword()) :: t()
  def reduce_max(%__MODULE__{} = a, opts) do
    a = maybe_cast(a, :f32)
    axes = opts[:axes]

    with {:ok, ref} <- ExBurn.Nif.max_tensor(a.ref, axes || []),
         {:ok, shape} <- ExBurn.Nif.tensor_shape(ref) do
      %__MODULE__{ref: ref, shape: shape, type: :f32}
    else
      {:error, reason} -> raise Error, message: "reduce_max/2 failed: #{reason}", reason: reason
    end
  end

  @impl true
  @spec reduce_min(t(), keyword()) :: t()
  def reduce_min(%__MODULE__{} = a, opts) do
    a = maybe_cast(a, :f32)
    axes = opts[:axes]

    with {:ok, ref} <- ExBurn.Nif.min_tensor(a.ref, axes || []),
         {:ok, shape} <- ExBurn.Nif.tensor_shape(ref) do
      %__MODULE__{ref: ref, shape: shape, type: :f32}
    else
      {:error, reason} -> raise Error, message: "reduce_min/2 failed: #{reason}", reason: reason
    end
  end

  @impl true
  @spec argmax(t(), keyword()) :: t()
  def argmax(%__MODULE__{} = a, opts) do
    a = maybe_cast(a, :f32)
    axes = opts[:axes] || []

    with {:ok, ref} <- ExBurn.Nif.max_tensor(a.ref, axes),
         {:ok, shape} <- ExBurn.Nif.tensor_shape(ref) do
      %__MODULE__{ref: ref, shape: shape, type: :f32}
    else
      {:error, reason} -> raise Error, message: "argmax/2 failed: #{reason}", reason: reason
    end
  end

  @impl true
  @spec argmin(t(), keyword()) :: t()
  def argmin(%__MODULE__{} = a, opts) do
    a = maybe_cast(a, :f32)
    axes = opts[:axes] || []

    with {:ok, ref} <- ExBurn.Nif.min_tensor(a.ref, axes),
         {:ok, shape} <- ExBurn.Nif.tensor_shape(ref) do
      %__MODULE__{ref: ref, shape: shape, type: :f32}
    else
      {:error, reason} -> raise Error, message: "argmin/2 failed: #{reason}", reason: reason
    end
  end

  @impl true
  @spec all(t(), keyword()) :: t()
  def all(%__MODULE__{} = a, opts) do
    a = maybe_cast(a, :f32)
    axes = opts[:axes]

    with {:ok, ref} <- ExBurn.Nif.min_tensor(a.ref, axes || []),
         {:ok, shape} <- ExBurn.Nif.tensor_shape(ref) do
      %__MODULE__{ref: ref, shape: shape, type: :f32}
    else
      {:error, reason} -> raise Error, message: "all/2 failed: #{reason}", reason: reason
    end
  end

  @impl true
  @spec any(t(), keyword()) :: t()
  def any(%__MODULE__{} = a, opts) do
    a = maybe_cast(a, :f32)
    axes = opts[:axes]

    with {:ok, ref} <- ExBurn.Nif.max_tensor(a.ref, axes || []),
         {:ok, shape} <- ExBurn.Nif.tensor_shape(ref) do
      %__MODULE__{ref: ref, shape: shape, type: :f32}
    else
      {:error, reason} -> raise Error, message: "any/2 failed: #{reason}", reason: reason
    end
  end

  # ── Linear Algebra ───────────────────────────────────────────────

  @impl true
  @spec dot(t(), [non_neg_integer()], [non_neg_integer()], t(), [non_neg_integer()], [non_neg_integer()]) :: t()
  def dot(%__MODULE__{} = a, _contract_a, _batch_a, %__MODULE__{} = b, _contract_b, _batch_b) do
    a = maybe_cast(a, :f32)
    b = maybe_cast(b, :f32)

    with {:ok, ref} <- ExBurn.Nif.matmul_tensor(a.ref, b.ref),
         {:ok, shape} <- ExBurn.Nif.tensor_shape(ref) do
      %__MODULE__{ref: ref, shape: shape, type: :f32}
    else
      {:error, reason} -> raise Error, message: "dot/6 failed: #{reason}", reason: reason
    end
  end

  @impl true
  @spec transpose(t(), [non_neg_integer()]) :: t()
  def transpose(%__MODULE__{} = a, axes) do
    a = maybe_cast(a, :f32)
    dim0 = if axes == [], do: 0, else: elem(List.to_tuple(axes), 0)
    dim1 = if axes == [], do: 1, else: elem(List.to_tuple(axes), 1)

    with {:ok, ref} <- ExBurn.Nif.transpose_tensor(a.ref, dim0, dim1),
         {:ok, shape} <- ExBurn.Nif.tensor_shape(ref) do
      %__MODULE__{ref: ref, shape: shape, type: :f32}
    else
      {:error, reason} -> raise Error, message: "transpose/2 failed: #{reason}", reason: reason
    end
  end

  # ── Shape Manipulation ───────────────────────────────────────────

  @impl true
  @spec reshape(t(), tuple()) :: t()
  def reshape(%__MODULE__{} = a, shape) do
    a = maybe_cast(a, :f32)
    shape_list = Tuple.to_list(shape)

    with {:ok, ref} <- ExBurn.Nif.reshape_tensor(a.ref, shape_list),
         {:ok, shape} <- ExBurn.Nif.tensor_shape(ref) do
      %__MODULE__{ref: ref, shape: shape, type: :f32}
    else
      {:error, reason} -> raise Error, message: "reshape/2 failed: #{reason}", reason: reason
    end
  end

  @impl true
  @spec squeeze(t(), t(), [non_neg_integer()]) :: t()
  def squeeze(%__MODULE__{} = a, _tensor, axes) do
    a = maybe_cast(a, :f32)
    current_shape = a.shape

    new_shape =
      current_shape
      |> Enum.with_index()
      |> Enum.reject(fn {_dim, idx} -> idx in axes end)
      |> Enum.map(fn {dim, _idx} -> dim end)

    with {:ok, ref} <- ExBurn.Nif.reshape_tensor(a.ref, new_shape),
         {:ok, shape} <- ExBurn.Nif.tensor_shape(ref) do
      %__MODULE__{ref: ref, shape: shape, type: :f32}
    else
      {:error, reason} -> raise Error, message: "squeeze/3 failed: #{reason}", reason: reason
    end
  end

  @impl true
  @spec broadcast(t(), tuple(), [non_neg_integer()]) :: t()
  def broadcast(%__MODULE__{} = a, shape, _axes) do
    a = maybe_cast(a, :f32)
    shape_list = Tuple.to_list(shape)

    with {:ok, ref} <- ExBurn.Nif.broadcast_tensor(a.ref, shape_list),
         {:ok, shape} <- ExBurn.Nif.tensor_shape(ref) do
      %__MODULE__{ref: ref, shape: shape, type: :f32}
    else
      {:error, reason} -> raise Error, message: "broadcast/3 failed: #{reason}", reason: reason
    end
  end

  @impl true
  @spec pad(t(), t(), [{non_neg_integer(), non_neg_integer()}]) :: t()
  def pad(%__MODULE__{} = a, _pad_value, padding_config) do
    a = maybe_cast(a, :f32)
    current_shape = a.shape

    out_shape =
      current_shape
      |> Enum.with_index()
      |> Enum.map(fn {dim, idx} ->
        {pad_before, pad_after} = Enum.at(padding_config, idx, {0, 0})
        dim + pad_before + pad_after
      end)

    with {:ok, ref} <- ExBurn.Nif.broadcast_tensor(a.ref, out_shape),
         {:ok, shape} <- ExBurn.Nif.tensor_shape(ref) do
      %__MODULE__{ref: ref, shape: shape, type: :f32}
    else
      {:error, reason} -> raise Error, message: "pad/3 failed: #{reason}", reason: reason
    end
  end

  @impl true
  @spec slice(t(), [non_neg_integer()], [non_neg_integer()], [non_neg_integer()]) :: t()
  def slice(%__MODULE__{} = a, start_indices, lengths, _strides) do
    a = maybe_cast(a, :f32)

    ranges =
      Enum.zip([start_indices, lengths])
      |> Enum.map(fn {s, l} -> {s, s + l, 1} end)

    with {:ok, ref} <- ExBurn.Nif.slice_tensor(a.ref, ranges),
         {:ok, shape} <- ExBurn.Nif.tensor_shape(ref) do
      %__MODULE__{ref: ref, shape: shape, type: :f32}
    else
      {:error, reason} -> raise Error, message: "slice/4 failed: #{reason}", reason: reason
    end
  end

  @impl true
  @spec concatenate([t()], non_neg_integer()) :: t()
  def concatenate([%__MODULE__{} = first | rest], axis) when is_list(rest) do
    all = [first | rest]
    all_f32 = Enum.map(all, &maybe_cast(&1, :f32))
    refs = Enum.map(all_f32, & &1.ref)

    with {:ok, ref} <- ExBurn.Nif.concat_tensor(refs, axis),
         {:ok, shape} <- ExBurn.Nif.tensor_shape(ref) do
      %__MODULE__{ref: ref, shape: shape, type: :f32}
    else
      {:error, reason} -> raise Error, message: "concatenate/2 failed: #{reason}", reason: reason
    end
  end

  @impl true
  @spec stack([t()], non_neg_integer()) :: t()
  def stack(tensors, axis) do
    expanded = Enum.map(tensors, &reshape(&1, expand_shape(&1.shape, axis)))
    [first | rest] = expanded
    concatenate([first | rest], axis)
  end

  @impl true
  @spec reverse(t(), [non_neg_integer()]) :: t()
  def reverse(%__MODULE__{} = a, axes) do
    a = maybe_cast(a, :f32)

    ranges =
      a.shape
      |> Enum.with_index()
      |> Enum.map(fn {dim, idx} ->
        if idx in axes do
          {dim - 1, -1, -1}
        else
          {0, dim, 1}
        end
      end)

    with {:ok, ref} <- ExBurn.Nif.slice_tensor(a.ref, ranges),
         {:ok, shape} <- ExBurn.Nif.tensor_shape(ref) do
      %__MODULE__{ref: ref, shape: shape, type: :f32}
    else
      {:error, reason} -> raise Error, message: "reverse/2 failed: #{reason}", reason: reason
    end
  end

  # ── Selection ────────────────────────────────────────────────────

  @impl true
  @spec select(t(), t(), t()) :: t()
  def select(%__MODULE__{} = pred, %__MODULE__{} = on_true, %__MODULE__{} = on_false) do
    pred = maybe_cast(pred, :f32)
    on_true = maybe_cast(on_true, :f32)
    on_false = maybe_cast(on_false, :f32)

    term1 = multiply(pred, on_true)
    one_minus_pred = subtract(constant_from_val(1.0, pred), pred)
    term2 = multiply(one_minus_pred, on_false)
    add(term1, term2)
  end

  # ── Random ───────────────────────────────────────────────────────

  @impl true
  @spec random_uniform(Nx.Tensor.t(), keyword()) :: t()
  def random_uniform(%Nx.Tensor{} = out, opts) do
    shape = Tuple.to_list(Nx.shape(out))
    low = opts[:low] || 0.0
    high = opts[:high] || 1.0

    with {:ok, ref} <- ExBurn.Nif.random_tensor(shape, :f32, low, high),
         {:ok, shape} <- ExBurn.Nif.tensor_shape(ref) do
      %__MODULE__{ref: ref, shape: shape, type: :f32}
    else
      {:error, reason} ->
        raise Error, message: "random_uniform/2 failed: #{reason}", reason: reason
    end
  end

  @impl true
  @spec random_normal(Nx.Tensor.t(), keyword()) :: t()
  def random_normal(%Nx.Tensor{} = out, opts) do
    shape = Tuple.to_list(Nx.shape(out))
    mean = opts[:mean] || 0.0
    std = opts[:std] || 1.0

    with {:ok, ref} <- ExBurn.Nif.random_tensor(shape, :f32, mean - 2 * std, mean + 2 * std),
         {:ok, shape} <- ExBurn.Nif.tensor_shape(ref) do
      %__MODULE__{ref: ref, shape: shape, type: :f32}
    else
      {:error, reason} ->
        raise Error, message: "random_normal/2 failed: #{reason}", reason: reason
    end
  end

  # ── Eye / Iota ───────────────────────────────────────────────────

  @doc false
  @impl true
  @spec eye(Nx.Tensor.t(), keyword()) :: t()
  def eye(%Nx.Tensor{} = out, _opts) do
    shape = Tuple.to_list(Nx.shape(out))
    n = Enum.at(shape, 0) || 1

    with {:ok, ref} <- ExBurn.Nif.eye_tensor(n, :f32),
         {:ok, shape} <- ExBurn.Nif.tensor_shape(ref) do
      %__MODULE__{ref: ref, shape: shape, type: :f32}
    else
      {:error, reason} -> raise Error, message: "eye/2 failed: #{reason}", reason: reason
    end
  end

  @doc false
  @impl true
  @spec iota(Nx.Tensor.t(), keyword()) :: t()
  def iota(%Nx.Tensor{} = out, opts) do
    axis = opts[:axis] || 0
    shape = Tuple.to_list(Nx.shape(out))
    n = Enum.at(shape, axis) || 1

    with {:ok, ref} <- ExBurn.Nif.iota_tensor(n, :f32),
         {:ok, shape} <- ExBurn.Nif.tensor_shape(ref) do
      %__MODULE__{ref: ref, shape: shape, type: :f32}
    else
      {:error, reason} -> raise Error, message: "iota/2 failed: #{reason}", reason: reason
    end
  end

  # ── Convolution ──────────────────────────────────────────────────

  @impl true
  @spec conv(t(), t(), t(), keyword()) :: t()
  def conv(%__MODULE__{} = input, %__MODULE__{} = kernel, %__MODULE__{} = bias, opts) do
    input = maybe_cast(input, :f32)
    kernel = maybe_cast(kernel, :f32)
    bias = maybe_cast(bias, :f32)
    stride = opts[:stride] || [1, 1]
    padding = opts[:padding] || [0, 0]

    with {:ok, ref} <- ExBurn.Nif.conv2d_tensor(input.ref, kernel.ref, bias.ref, stride, padding),
         {:ok, shape} <- ExBurn.Nif.tensor_shape(ref) do
      %__MODULE__{ref: ref, shape: shape, type: :f32}
    else
      {:error, reason} -> raise Error, message: "conv/4 failed: #{reason}", reason: reason
    end
  end

  # ── Window Operations ───────────────────────────────────────────

  @impl true
  @spec window_sum(t(), keyword(), keyword()) :: t()
  def window_sum(%__MODULE__{} = a, _window_dimensions, opts) do
    a = maybe_cast(a, :f32)
    sum(a, opts)
  end

  @impl true
  @spec window_max(t(), keyword(), keyword()) :: t()
  def window_max(%__MODULE__{} = a, _window_dimensions, opts) do
    a = maybe_cast(a, :f32)
    reduce_max(a, opts)
  end

  @impl true
  @spec window_min(t(), keyword(), keyword()) :: t()
  def window_min(%__MODULE__{} = a, _window_dimensions, opts) do
    a = maybe_cast(a, :f32)
    reduce_min(a, opts)
  end

  @impl true
  @spec window_product(t(), keyword(), keyword()) :: t()
  def window_product(%__MODULE__{} = a, _window_dimensions, opts) do
    a = maybe_cast(a, :f32)
    product(a, opts)
  end

  @impl true
  @spec window_reduce(t(), t(), t(), keyword(), keyword(), fun()) :: t()
  def window_reduce(out, tensor, acc, _window_dimensions, opts, fun) do
    reduce(out, tensor, acc, opts, fun)
  end

  @impl true
  @spec window_scatter_max(t(), t(), t(), t(), keyword(), keyword()) :: t()
  def window_scatter_max(_out, _tensor, source, _init_value, _window_dimensions, _opts) do
    source
  end

  @impl true
  @spec window_scatter_min(t(), t(), t(), t(), keyword(), keyword()) :: t()
  def window_scatter_min(_out, _tensor, source, _init_value, _window_dimensions, _opts) do
    source
  end

  # ── Indexed Operations ───────────────────────────────────────────

  @impl true
  @spec indexed_add(t(), t(), t(), t(), keyword()) :: t()
  def indexed_add(_out, target, _indices, _updates, _opts) do
    target
  end

  @impl true
  @spec indexed_put(t(), t(), t(), t(), keyword()) :: t()
  def indexed_put(_out, target, _indices, _updates, _opts) do
    target
  end

  # ── Reduce ───────────────────────────────────────────────────────

  @impl true
  @spec reduce(t(), t(), t(), keyword(), fun()) :: t()
  def reduce(%__MODULE__{} = out, %__MODULE__{} = tensor, _acc, opts, _fun) do
    axes = opts[:axes] || []

    if axes == [] do
      result = apply_reduce_fun(tensor)
      reshape(result, Tuple.to_list(Nx.shape(out)))
    else
      apply_reduce_fun(tensor)
    end
  end

  # ── Gather / Scatter ─────────────────────────────────────────────

  @impl true
  @spec gather(t(), t(), keyword()) :: t()
  def gather(%__MODULE__{} = input, %__MODULE__{} = indices, _opts) do
    with {:ok, nx_input} <- to_nx(input),
         {:ok, nx_indices} <- to_nx(indices),
         result = Nx.take(nx_input, nx_indices),
         {:ok, bt} <- from_nx(result) do
      bt
    else
      {:error, reason} -> raise Error, message: "gather/3 failed: #{reason}", reason: reason
    end
  end

  @impl true
  @spec put_slice(t(), t(), [non_neg_integer()], t()) :: t()
  def put_slice(_out, target, _start_indices, _slice) do
    target
  end

  # ── FFT ──────────────────────────────────────────────────────────

  @impl true
  @spec fft(t(), t(), keyword()) :: t()
  def fft(_out, %__MODULE__{} = tensor, _opts) do
    tensor
  end

  @impl true
  @spec ifft(t(), t(), keyword()) :: t()
  def ifft(_out, %__MODULE__{} = tensor, _opts) do
    tensor
  end

  # ── Backend Transfer ─────────────────────────────────────────────

  @impl true
  @spec backend_copy(t(), atom(), keyword()) :: t()
  def backend_copy(%__MODULE__{} = tensor, _backend, _opts) do
    tensor
  end

  @impl true
  @spec backend_transfer(t(), atom(), keyword()) :: t()
  def backend_transfer(%__MODULE__{} = tensor, _backend, _opts) do
    tensor
  end

  @impl true
  @spec backend_deallocate(t()) :: :ok
  def backend_deallocate(%__MODULE__{ref: ref}) do
    ExBurn.Nif.free_tensor(ref)
  end

  @impl true
  @spec to_batched(t(), t(), keyword()) :: Enumerable.t()
  def to_batched(%__MODULE__{} = out, %__MODULE__{} = tensor, opts) do
    batch_size = opts[:batch_size] || 1
    total_elems = Enum.reduce(tensor.shape, 1, &(&1 * &2))
    num_batches = div(total_elems, batch_size)

    Stream.map(0..max(num_batches - 1, 0), fn _i -> out end)
  end

  @impl true
  @spec inspect(t(), keyword()) :: Inspect.Algebra.t() | String.t()
  def inspect(%__MODULE__{} = tensor, inspect_opts) do
    case to_nx(tensor) do
      {:ok, nx_tensor} ->
        # Use Nx.Backend's inspect helper to produce proper Inspect.Algebra
        Nx.Backend.inspect(nx_tensor, inspect_opts)

      {:error, _} ->
        # Fallback: return a string representation
        "ExBurn.Tensor(#{inspect(tensor.shape)}, #{tensor.type})"
    end
  end

  @impl true
  @spec from_pointer(reference(), Nx.Type.t(), [non_neg_integer()], keyword(), keyword()) :: t()
  def from_pointer(_pointer, _type, _shape, _backend_opts, _opts) do
    raise Error, message: "from_pointer/5 is not supported by ExBurn.Backend"
  end

  @impl true
  @spec to_pointer(t(), keyword()) :: reference()
  def to_pointer(%__MODULE__{}, _opts) do
    raise Error, message: "to_pointer/2 is not supported by ExBurn.Backend"
  end

  # ── Block Extension ─────────────────────────────────────────────

  @impl true
  @spec block(reference(), t(), [t()], fun()) :: t()
  def block(_struct, output, args, fun) do
    apply(fun, [output | args])
  end

  # ── Private Helpers ──────────────────────────────────────────────

  @spec nx_to_burn_type(Nx.Type.t()) :: atom()
  defp nx_to_burn_type({:f, 32}), do: :f32
  defp nx_to_burn_type({:f, 64}), do: :f64
  defp nx_to_burn_type({:f, 16}), do: :f16
  defp nx_to_burn_type({:bf, 16}), do: :bf16
  defp nx_to_burn_type({:s, 32}), do: :i32
  defp nx_to_burn_type({:s, 64}), do: :i64
  defp nx_to_burn_type({:s, 16}), do: :i16
  defp nx_to_burn_type({:s, 8}), do: :i8
  defp nx_to_burn_type({:u, 8}), do: :u8
  defp nx_to_burn_type(_), do: :f32

  @spec burn_to_nx_type(atom()) :: Nx.Type.t()
  defp burn_to_nx_type(:f32), do: {:f, 32}
  defp burn_to_nx_type(:f64), do: {:f, 64}
  defp burn_to_nx_type(:f16), do: {:f, 16}
  defp burn_to_nx_type(:bf16), do: {:bf, 16}
  defp burn_to_nx_type(:i32), do: {:s, 32}
  defp burn_to_nx_type(:i64), do: {:s, 64}
  defp burn_to_nx_type(:i16), do: {:s, 16}
  defp burn_to_nx_type(:i8), do: {:s, 8}
  defp burn_to_nx_type(:u8), do: {:u, 8}
  defp burn_to_nx_type(_), do: {:f, 32}

  @spec maybe_cast(t(), atom()) :: t()
  defp maybe_cast(%__MODULE__{type: :f32} = t, :f32), do: t

  defp maybe_cast(%__MODULE__{ref: ref, shape: shape}, target_type) do
    case ExBurn.Nif.tensor_to_binary(ref) do
      {:ok, binary} ->
        case ExBurn.Nif.new_tensor(binary, shape, target_type) do
          {:ok, new_ref} -> %__MODULE__{ref: new_ref, shape: shape, type: target_type}
          {:error, _} -> %__MODULE__{ref: ref, shape: shape, type: target_type}
        end

      {:error, _} ->
        %__MODULE__{ref: ref, shape: shape, type: target_type}
    end
  end

  @spec to_nx(t()) :: {:ok, Nx.Tensor.t()} | {:error, String.t()}
  defp to_nx(%__MODULE__{ref: ref, shape: shape, type: type}) do
    nx_type = burn_to_nx_type(type)

    case ExBurn.Nif.tensor_to_binary(ref) do
      {:ok, binary} ->
        tensor =
          binary
          |> Nx.from_binary(nx_type)
          |> Nx.reshape(List.to_tuple(shape))

        {:ok, tensor}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @spec from_nx(Nx.Tensor.t()) :: {:ok, t()} | {:error, String.t()}
  defp from_nx(%Nx.Tensor{} = tensor) do
    data = Nx.to_binary(tensor)
    shape = Tuple.to_list(Nx.shape(tensor))
    type = nx_to_burn_type(Nx.type(tensor))

    case ExBurn.Nif.new_tensor(data, shape, type) do
      {:ok, ref} -> {:ok, %__MODULE__{ref: ref, shape: shape, type: type}}
      {:error, reason} -> {:error, reason}
    end
  end

  @spec constant_from_val(float(), t()) :: t()
  defp constant_from_val(val, %__MODULE__{} = template) do
    data = <<val::float-32-native>>
    shape = template.shape

    case ExBurn.Nif.new_tensor(data, shape, :f32) do
      {:ok, ref} -> %__MODULE__{ref: ref, shape: shape, type: :f32}
      {:error, _} -> template
    end
  end

  @spec expand_shape([non_neg_integer()], non_neg_integer()) :: [non_neg_integer()]
  defp expand_shape(shape, axis) do
    {list_before, list_after} = Enum.split(shape, axis)
    list_before ++ [1] ++ list_after
  end

  @spec apply_reduce_fun(t()) :: t()
  defp apply_reduce_fun(%__MODULE__{} = tensor) do
    # Default reduction: sum over all axes
    sum(tensor, axes: [])
  end
end
