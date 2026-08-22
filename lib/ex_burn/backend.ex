defmodule ExBurn.Backend do
  @moduledoc """
  Nx backend implementation that delegates tensor operations to Burn via NIF.

  This module implements the `Nx.Backend` behaviour, translating Nx tensor
  operations into calls to the Rust NIF layer which executes them using
  Burn's CubeCL backend for GPU acceleration.

  ## Architecture

  ```
  Axon model
     ↓
  Nx.Defn graph
     ↓
  ExBurn.Backend (Nx.Backend behaviour)
     ↓
  ExBurn.Nif (Rustler NIF) ←→ ExCubecl (GPU buffers, kernels, pipelines)
     ↓
  Burn Autodiff<CubeCL> (Rust)
     ↓
  CubeCL kernels
     ↓
  Metal (iOS) / Vulkan (Android) / CUDA → GPU
  ```

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

  # The callbacks intentionally pattern-match on the *data* struct
  # (%__MODULE__{}) rather than the full %Nx.Tensor{}, and return the data
  # struct directly — Nx wraps it into a tensor at runtime. This diverges
  # from the behaviour's nominal %Nx.Tensor{} specs, which Dialyzer flags
  # for every callback. Suppress at module level with this rationale;
  # revisit if the backend is ever reworked to build full tensors.
  @dialyzer {:nowarn_function,
             window_sum: 4,
             window_max: 4,
             window_min: 4,
             window_product: 4,
             window_reduce: 6,
             window_scatter_max: 6,
             window_scatter_min: 6,
             indexed_add: 5,
             indexed_put: 5,
             put_slice: 4,
             fft: 3,
             ifft: 3,
             sort: 3,
             argsort: 3,
             triangular_solve: 4,
             to_batched: 3}

  alias ExBurn.Tensor, as: BT
  alias ExBurn.Error
  alias ExBurn.NifHelper, as: Nif

  @type t :: %__MODULE__{
          ref: reference(),
          shape: [non_neg_integer()],
          type: BT.burn_type()
        }

  defstruct [:ref, :shape, :type]

  alias Nx.Tensor, as: T

  @dialyzer {:nowarn_function, block: 4}

  # Wraps a backend struct into an Nx.Tensor, per the Nx.Backend contract.
  defp wrap(out, %__MODULE__{} = data) do
    names =
      case out do
        %T{names: names} when is_list(names) -> names
        _ -> List.duplicate(nil, length(data.shape))
      end

    data = harmonize_shape(out, data)

    %T{
      data: data,
      shape: List.to_tuple(data.shape),
      type: burn_to_nx_type(data.type),
      names: names
    }
  end

  # The Rust layer materializes scalar results as rank-1 tensors; align the
  # stored shape with the declared output shape when element counts match.
  defp harmonize_shape(%T{} = out, %__MODULE__{shape: shape} = data) do
    declared = out |> Nx.shape() |> Tuple.to_list()

    if declared != shape and Enum.product(declared) == Enum.product(shape) do
      %{data | shape: declared}
    else
      data
    end
  end

  defp harmonize_shape(_out, data), do: data

  # ── Allocation ───────────────────────────────────────────────────

  @impl true
  @spec init(keyword()) :: :ok
  def init(_opts), do: :ok

  @impl true
  @spec constant(Nx.Tensor.t(), number(), keyword()) :: t()
  def constant(out, value, _opts) do
    # The NIF layer only supports f32 storage; always encode as f32.
    type = :f32
    shape = Tuple.to_list(Nx.shape(out))

    # The Rust side validates byte size against the declared shape, so the
    # scalar value must be replicated to fill every element.
    data = :binary.copy(<<value * 1.0::float-32-native>>, Enum.product(shape))

    case Nif.new_tensor(data, shape, Atom.to_string(type)) do
      {:ok, ref} ->
        wrap(out, %__MODULE__{ref: ref, shape: shape, type: type})

      {:error, reason} ->
        raise Error, op: :constant, reason: reason
    end
  end

  @impl true
  @spec from_binary(Nx.Tensor.t(), binary(), keyword()) :: t()
  def from_binary(%{type: nx_type} = out, binary, _opts) do
    burn_type = nx_to_burn_type(nx_type)
    shape = Tuple.to_list(Nx.shape(out))

    # The NIF stores f32 only: re-encode through Nx so the dtype
    # conversion is value-preserving instead of a byte reinterpretation.
    data = encode_as_f32(binary, nx_type)

    case Nif.new_tensor(data, shape, Atom.to_string(burn_type)) do
      {:ok, ref} ->
        wrap(out, %__MODULE__{ref: ref, shape: shape, type: burn_type})

      {:error, reason} ->
        raise Error, op: :from_binary, reason: reason
    end
  end

  defp encode_as_f32(binary, {:f, 32}), do: binary

  defp encode_as_f32(binary, nx_type) do
    binary
    |> Nx.from_binary(nx_type)
    |> Nx.as_type({:f, 32})
    |> Nx.to_binary()
  end

  @impl true
  @spec to_binary(t(), non_neg_integer()) :: binary()
  def to_binary(%T{data: %__MODULE__{ref: ref}}, _limit) do
    case Nif.tensor_to_binary(ref) do
      {:ok, binary} ->
        binary

      {:error, reason} ->
        raise Error, op: :to_binary, reason: reason
    end
  end

  @spec tensor_type(t()) :: Nx.Type.t()
  def tensor_type(%__MODULE__{type: type}), do: burn_to_nx_type(type)

  @doc "Returns the total number of elements in the backend tensor."
  @spec parameter_count(t()) :: non_neg_integer()
  def parameter_count(%__MODULE__{shape: shape}), do: Enum.product(shape)

  @doc "Returns a human-readable string representation of the tensor."
  @spec describe(t()) :: String.t()
  def describe(%__MODULE__{shape: shape, type: type}) do
    "ExBurn.Tensor<shape: #{inspect(shape)}, type: #{type}, elements: #{Enum.product(shape)}>"
  end

  # ── Element-wise Arithmetic ──────────────────────────────────────

  # Binary ops dispatched to the NIF with type promotion.
  @binary_nif_ops [
    add: :add_tensor,
    subtract: :sub_tensor,
    multiply: :mul_tensor,
    divide: :div_tensor,
    pow: :pow_tensor
  ]

  for {nx_op, nif} <- @binary_nif_ops do
    @impl true
    def unquote(nx_op)(out, %T{data: %__MODULE__{} = a}, %T{data: %__MODULE__{} = b}) do
      out_type = result_type(a.type, b.type)
      a = maybe_cast(a, out_type)
      b = maybe_cast(b, out_type)

      with {:ok, ref} <- Nif.unquote(nif)(a.ref, b.ref),
           {:ok, shape} <- Nif.tensor_shape(ref) do
        wrap(out, %__MODULE__{ref: ref, shape: shape, type: out_type})
      else
        {:error, reason} -> raise Error, op: unquote(nx_op), reason: reason
      end
    end

    # Fallback: one side may be a scalar/constant on the default backend.
    # Convert both sides to this backend, then dispatch to the NIF path.
    @impl true
    def unquote(nx_op)(out, %T{} = a, %T{} = b) do
      a = Nx.backend_transfer(a, __MODULE__)
      b = Nx.backend_transfer(b, __MODULE__)
      unquote(nx_op)(out, a, b)
    end
  end

  # Unary ops dispatched to the NIF (always f32).
  @unary_nif_ops [
    negate: :neg_tensor,
    abs: :abs_tensor,
    exp: :exp_tensor,
    log: :log_tensor,
    sqrt: :sqrt_tensor,
    sigmoid: :sigmoid_tensor,
    tanh: :tanh_tensor
  ]

  for {nx_op, nif} <- @unary_nif_ops do
    @impl true
    def unquote(nx_op)(out, %T{data: %__MODULE__{} = a}) do
      a = maybe_cast(a, :f32)

      with {:ok, ref} <- Nif.unquote(nif)(a.ref),
           {:ok, shape} <- Nif.tensor_shape(ref) do
        wrap(out, %__MODULE__{ref: ref, shape: shape, type: :f32})
      else
        {:error, reason} -> raise Error, op: unquote(nx_op), reason: reason
      end
    end

    @impl true
    def unquote(nx_op)(out, %T{} = a) do
      a = Nx.backend_transfer(a, __MODULE__)
      unquote(nx_op)(out, a)
    end
  end

  @impl true
  @spec remainder(Nx.Tensor.t(), t(), t()) :: t()
  def remainder(_out, %T{data: %__MODULE__{} = a}, %T{data: %__MODULE__{} = b}) do
    # Truncated remainder: r = a - trunc(a / b) * b
    nx_fallback(:remainder, fn ->
      nx_a = to_nx!(a)
      nx_b = to_nx!(b)

      ratio = Nx.divide(nx_a, nx_b)

      truncated =
        Nx.select(Nx.greater(ratio, 0.0), Nx.floor(ratio), Nx.ceil(ratio))

      Nx.subtract(nx_a, Nx.multiply(truncated, nx_b))
    end)
  end

  # Ops computed via Nx round-trip (no direct NIF support yet).
  # Each entry: {callback name, arity, Nx function, error op tag}.
  @nx_binary_ops [
    atan2: :atan2,
    min: :min,
    max: :max,
    bitwise_and: :bitwise_and,
    bitwise_or: :bitwise_or,
    bitwise_xor: :bitwise_xor,
    left_shift: :left_shift,
    right_shift: :right_shift,
    equal: :equal,
    not_equal: :not_equal,
    greater: :greater,
    less: :less,
    greater_equal: :greater_equal,
    less_equal: :less_equal,
    logical_and: :logical_and,
    logical_or: :logical_or,
    logical_xor: :logical_xor
  ]

  for {nx_op, err_op} <- @nx_binary_ops do
    @impl true
    def unquote(nx_op)(out, %T{data: %__MODULE__{} = a}, %T{data: %__MODULE__{} = b}) do
      unquote(nx_op)(out, a, b)
    end

    @impl true
    def unquote(nx_op)(_out, %__MODULE__{} = a, %__MODULE__{} = b) do
      with {:ok, nx_a} <- to_nx(a),
           {:ok, nx_b} <- to_nx(b),
           result = Nx.unquote(nx_op)(nx_a, nx_b),
           {:ok, bt} <- from_nx(result) do
        bt
      else
        {:error, reason} -> raise Error, op: unquote(err_op), reason: reason
      end
    end
  end

  @nx_unary_ops [
    acos: :acos,
    acosh: :acosh,
    asin: :asin,
    asinh: :asinh,
    atan: :atan,
    atanh: :atanh,
    cbrt: :cbrt,
    ceil: :ceil,
    conjugate: :conjugate,
    cos: :cos,
    cosh: :cosh,
    erf: :erf,
    erfc: :erfc,
    erf_inv: :erf_inv,
    expm1: :expm1,
    floor: :floor,
    log1p: :log1p,
    rsqrt: :rsqrt,
    sin: :sin,
    sinh: :sinh,
    tan: :tan,
    bitwise_not: :bitwise_not,
    round: :round,
    sign: :sign,
    count_leading_zeros: :count_leading_zeros,
    population_count: :population_count,
    real: :real,
    imag: :imag,
    is_nan: :is_nan,
    is_infinity: :is_infinity
  ]

  for {nx_op, err_op} <- @nx_unary_ops do
    @impl true
    def unquote(nx_op)(out, %T{data: %__MODULE__{} = a}) do
      unquote(nx_op)(out, a)
    end

    @impl true
    def unquote(nx_op)(_out, %__MODULE__{} = a) do
      with {:ok, nx_a} <- to_nx(a),
           result = Nx.unquote(nx_op)(nx_a),
           {:ok, bt} <- from_nx(result) do
        bt
      else
        {:error, reason} -> raise Error, op: unquote(err_op), reason: reason
      end
    end
  end

  @impl true
  @spec quotient(Nx.Tensor.t(), t(), t()) :: t()
  def quotient(_out, %T{data: %__MODULE__{} = a}, %T{data: %__MODULE__{} = b}) do
    # Truncated integer-style division, computed exactly via Nx.
    nx_fallback(:quotient, fn ->
      Nx.floor(Nx.divide(to_nx!(a), to_nx!(b)))
    end)
  end

  # ── Reductions ───────────────────────────────────────────────────

  @impl true
  @spec sum(Nx.Tensor.t(), t(), keyword()) :: t()
  def sum(out, %T{data: %__MODULE__{} = a}, opts) do
    axes = opts[:axes] || []

    if full_reduction?(a, axes) do
      a = maybe_cast(a, :f32)

      with {:ok, ref} <- Nif.sum_tensor(a.ref) do
        shape = Tuple.to_list(Nx.shape(out))
        wrap(out, %__MODULE__{ref: ref, shape: shape, type: :f32})
      else
        {:error, reason} -> raise Error, op: :sum, reason: reason
      end
    else
      nx_fallback(:sum, fn -> Nx.sum(to_nx!(a), axes: axes, keep_axes: false) end)
    end
  end

  @impl true
  @spec product(Nx.Tensor.t(), t(), keyword()) :: t()
  def product(_out, %T{data: %__MODULE__{} = a}, opts) do
    # exp(sum(log(x))) is only valid for strictly positive inputs, so
    # compute the product exactly via Nx instead.
    nx_fallback(:product, fn ->
      Nx.product(to_nx!(a), axes: opts[:axes], keep_axes: false)
    end)
  end

  @impl true
  @spec reduce_max(Nx.Tensor.t(), t(), keyword()) :: t()
  def reduce_max(out, %T{data: %__MODULE__{} = a}, opts) do
    axes = opts[:axes] || []

    if full_reduction?(a, axes) do
      a = maybe_cast(a, :f32)

      with {:ok, ref} <- Nif.max_tensor(a.ref) do
        shape = Tuple.to_list(Nx.shape(out))
        wrap(out, %__MODULE__{ref: ref, shape: shape, type: :f32})
      else
        {:error, reason} -> raise Error, op: :reduce_max, reason: reason
      end
    else
      nx_fallback(:reduce_max, fn -> Nx.reduce_max(to_nx!(a), axes: axes, keep_axes: false) end)
    end
  end

  @impl true
  @spec reduce_min(Nx.Tensor.t(), t(), keyword()) :: t()
  def reduce_min(out, %T{data: %__MODULE__{} = a}, opts) do
    axes = opts[:axes] || []

    if full_reduction?(a, axes) do
      a = maybe_cast(a, :f32)

      with {:ok, ref} <- Nif.min_tensor(a.ref) do
        shape = Tuple.to_list(Nx.shape(out))
        wrap(out, %__MODULE__{ref: ref, shape: shape, type: :f32})
      else
        {:error, reason} -> raise Error, op: :reduce_min, reason: reason
      end
    else
      nx_fallback(:reduce_min, fn -> Nx.reduce_min(to_nx!(a), axes: axes, keep_axes: false) end)
    end
  end

  @impl true
  @spec argmax(Nx.Tensor.t(), t(), keyword()) :: t()
  def argmax(_out, %T{data: %__MODULE__{} = a}, opts) do
    # Must return *indices*, not values — compute via Nx.
    nx_fallback(:argmax, fn ->
      Nx.argmax(to_nx!(a),
        axis: opts[:axis],
        tie_break: opts[:tie_break] || :low,
        keep_axis: opts[:keep_axis] || false
      )
    end)
  end

  @impl true
  @spec argmin(Nx.Tensor.t(), t(), keyword()) :: t()
  def argmin(_out, %T{data: %__MODULE__{} = a}, opts) do
    nx_fallback(:argmin, fn ->
      Nx.argmin(to_nx!(a),
        axis: opts[:axis],
        tie_break: opts[:tie_break] || :low,
        keep_axis: opts[:keep_axis] || false
      )
    end)
  end

  @impl true
  @spec all(Nx.Tensor.t(), t(), keyword()) :: t()
  def all(_out, %T{data: %__MODULE__{} = a}, opts) do
    nx_fallback(:all, fn -> Nx.all(to_nx!(a), axes: opts[:axes], keep_axes: false) end)
  end

  @impl true
  @spec any(Nx.Tensor.t(), t(), keyword()) :: t()
  def any(_out, %T{data: %__MODULE__{} = a}, opts) do
    nx_fallback(:any, fn -> Nx.any(to_nx!(a), axes: opts[:axes], keep_axes: false) end)
  end

  # A reduction is "full" when no axes are given or every axis is reduced.
  defp full_reduction?(%__MODULE__{shape: shape}, axes),
    do: axes == [] or length(axes) >= length(shape)

  # Runs `fun` against the plain-Nx view of the tensor and uploads the
  # result back to Burn. Used for ops without a direct NIF implementation.
  defp nx_fallback(op, fun) do
    case from_nx(fun.()) do
      {:ok, bt} -> bt
      {:error, reason} -> raise Error, op: op, reason: reason
    end
  end

  defp to_nx!(%__MODULE__{} = tensor) do
    case to_nx(tensor) do
      {:ok, nx} -> nx
      {:error, reason} -> raise Error, op: :to_binary, reason: reason
    end
  end

  # ── Linear Algebra ───────────────────────────────────────────────

  @impl true
  @spec dot(
          Nx.Tensor.t(),
          t(),
          [non_neg_integer()],
          [non_neg_integer()],
          t(),
          [non_neg_integer()],
          [non_neg_integer()]
        ) :: t()
  def dot(out, %T{data: %__MODULE__{} = a}, ca, ba, %T{data: %__MODULE__{} = b}, cb, bb),
    do: dot_bare(out, a, ca, ba, b, cb, bb)

  def dot(out, %__MODULE__{} = a, ca, ba, %__MODULE__{} = b, cb, bb),
    do: dot_bare(out, a, ca, ba, b, cb, bb)

  defp dot_bare(
         out,
         %__MODULE__{} = a,
         contract_a,
         batch_a,
         %__MODULE__{} = b,
         contract_b,
         batch_b
       ) do
    fast_path? =
      batch_a == [] and batch_b == [] and
        length(a.shape) >= 2 and length(b.shape) >= 2 and
        contract_a == [length(a.shape) - 1] and contract_b == [0]

    if fast_path? do
      # Fast path: plain inner-product-style matmul on the GPU.
      a = maybe_cast(a, :f32)
      b = maybe_cast(b, :f32)

      with {:ok, ref} <- Nif.matmul_tensor(a.ref, b.ref),
           {:ok, shape} <- Nif.tensor_shape(ref) do
        wrap(out, %__MODULE__{ref: ref, shape: shape, type: :f32})
      else
        {:error, reason} -> raise Error, op: :dot, reason: reason
      end
    else
      # General contraction — compute exactly via Nx.
      nx_fallback(:dot, fn ->
        Nx.dot(
          to_nx!(a),
          contract_a,
          batch_a,
          to_nx!(b),
          contract_b,
          batch_b
        )
      end)
    end
  end

  @impl true
  @spec transpose(Nx.Tensor.t(), t(), [non_neg_integer()]) :: t()
  def transpose(out, %T{data: %__MODULE__{} = a}, axes) do
    a = maybe_cast(a, :f32)
    rank = length(a.shape)
    default_axes = Enum.to_list((rank - 1)..0//-1)

    cond do
      rank == 2 and (axes == [1, 0] or axes == [] or axes == nil) ->
        with {:ok, ref} <- Nif.transpose_tensor(a.ref),
             {:ok, shape} <- Nif.tensor_shape(ref) do
          wrap(out, %__MODULE__{ref: ref, shape: shape, type: :f32})
        else
          {:error, reason} -> raise Error, op: :transpose, reason: reason
        end

      true ->
        nx_fallback(:transpose, fn -> Nx.transpose(to_nx!(a), axes: axes || default_axes) end)
    end
  end

  # ── Shape Manipulation ───────────────────────────────────────────

  @impl true
  @spec reshape(Nx.Tensor.t(), t()) :: t()
  def reshape(out, %T{data: %__MODULE__{} = a}) do
    a = maybe_cast(a, :f32)
    shape_list = Tuple.to_list(Nx.shape(out))

    with {:ok, ref} <- Nif.reshape_tensor(a.ref, shape_list),
         {:ok, shape} <- Nif.tensor_shape(ref) do
      wrap(out, %__MODULE__{ref: ref, shape: shape, type: :f32})
    else
      {:error, reason} -> raise Error, op: :reshape, reason: reason
    end
  end

  @impl true
  @spec squeeze(Nx.Tensor.t(), t(), [non_neg_integer()]) :: t()
  def squeeze(out, %T{data: %__MODULE__{} = a}, axes) do
    a = maybe_cast(a, :f32)
    current_shape = a.shape

    new_shape =
      current_shape
      |> Enum.with_index()
      |> Enum.reject(fn {_dim, idx} -> idx in axes end)
      |> Enum.map(fn {dim, _idx} -> dim end)

    with {:ok, ref} <- Nif.reshape_tensor(a.ref, new_shape),
         {:ok, shape} <- Nif.tensor_shape(ref) do
      wrap(out, %__MODULE__{ref: ref, shape: shape, type: :f32})
    else
      {:error, reason} -> raise Error, op: :squeeze, reason: reason
    end
  end

  @impl true
  @spec broadcast(Nx.Tensor.t(), t(), tuple(), [non_neg_integer()]) :: t()
  def broadcast(out, %T{data: %__MODULE__{} = a}, shape, _axes) do
    a = maybe_cast(a, :f32)
    shape_list = Tuple.to_list(shape)

    with {:ok, ref} <- Nif.broadcast_tensor(a.ref, shape_list),
         {:ok, shape} <- Nif.tensor_shape(ref) do
      wrap(out, %__MODULE__{ref: ref, shape: shape, type: :f32})
    else
      {:error, reason} -> raise Error, op: :broadcast, reason: reason
    end
  end

  @impl true
  @spec pad(Nx.Tensor.t(), t(), t(), [{non_neg_integer(), non_neg_integer()}]) :: t()
  def pad(out, %T{data: %__MODULE__{} = a}, %T{data: %__MODULE__{} = pv}, cfg),
    do: pad_bare(out, a, pv, cfg)

  def pad(out, %__MODULE__{} = a, %__MODULE__{} = pv, cfg), do: pad_bare(out, a, pv, cfg)

  defp pad_bare(_out, %__MODULE__{} = a, pad_value, padding_config) do
    # Convert to Nx, pad using Nx.pad/3, then convert back.
    # Nx.pad handles arbitrary padding configurations correctly.
    with {:ok, nx_a} <- to_nx(a),
         {:ok, nx_pad_val} <- to_nx(maybe_cast(pad_value, :f32)),
         result = Nx.pad(nx_a, nx_pad_val, padding_config),
         {:ok, bt} <- from_nx(result) do
      bt
    else
      {:error, reason} -> raise Error, op: :pad, reason: reason
    end
  end

  @impl true
  @spec slice(Nx.Tensor.t(), t(), [non_neg_integer()], [non_neg_integer()], [non_neg_integer()]) ::
          t()
  def slice(_out, %T{data: %__MODULE__{} = a}, start_indices, lengths, strides) do
    a = maybe_cast(a, :f32)

    nx_fallback(:slice, fn ->
      to_nx!(a) |> Nx.slice(start_indices, lengths, strides: strides)
    end)
  end

  @impl true
  @spec concatenate(Nx.Tensor.t(), [t()], non_neg_integer()) :: t()
  def concatenate(_out, tensors, axis) do
    all_f32 = Enum.map(tensors, &maybe_cast(&1, :f32))
    one_dimensional? = Enum.all?(all_f32, &(length(&1.shape) == 1))

    if axis == 0 and one_dimensional? do
      # Fast path: 1D concatenation runs on the GPU via the NIF.
      [first | rest] = all_f32

      Enum.reduce(rest, first, fn b, acc ->
        with {:ok, ref} <- Nif.concat_tensor(acc.ref, b.ref),
             {:ok, shape} <- Nif.tensor_shape(ref) do
          %__MODULE__{ref: ref, shape: shape, type: :f32}
        else
          {:error, reason} -> raise Error, op: :concatenate, reason: reason
        end
      end)
    else
      # General case — compute exactly via Nx for correct axis handling.
      nx_fallback(:concatenate, fn ->
        all_f32 |> Enum.map(&to_nx!/1) |> Nx.concatenate(axis: axis)
      end)
    end
  end

  @impl true
  @spec stack(Nx.Tensor.t(), [t()], non_neg_integer()) :: t()
  def stack(out, tensors, axis) do
    expanded = Enum.map(tensors, &stack_expand(&1, axis))
    concatenate(out, expanded, axis)
  end

  # Reshapes one input to insert the stacked axis. Accepts both wrapped
  # tensors and bare backend structs; concatenate/3 expects bare structs.
  defp stack_expand(%T{data: %__MODULE__{} = data}, axis), do: stack_expand(data, axis)

  defp stack_expand(%__MODULE__{shape: shape, type: type} = data, axis) do
    wrapper = %T{
      data: data,
      type: burn_to_nx_type(type),
      shape: List.to_tuple(shape),
      names: List.duplicate(nil, length(shape))
    }

    new_out = Nx.broadcast(Nx.tensor(0.0, type: :f32), List.to_tuple(expand_shape(shape, axis)))

    case reshape(new_out, wrapper) do
      %T{data: %__MODULE__{} = reshaped} -> reshaped
    end
  end

  @impl true
  @spec reverse(Nx.Tensor.t(), t(), [non_neg_integer()]) :: t()
  def reverse(_out, %T{data: %__MODULE__{} = a}, axes) do
    a = maybe_cast(a, :f32)

    nx_fallback(:reverse, fn -> Nx.reverse(to_nx!(a), axes: axes) end)
  end

  # ── Selection ────────────────────────────────────────────────────

  @impl true
  @spec select(Nx.Tensor.t(), t(), t(), t()) :: t()
  def select(_out, %T{data: %__MODULE__{} = pred}, %T{data: %__MODULE__{} = on_true}, %T{
        data: %__MODULE__{} = on_false
      }) do
    # Exact semantics via Nx (the previous arithmetic encoding only worked
    # for 0/1 float predicates of identical shapes).
    nx_fallback(:select, fn ->
      Nx.select(to_nx!(pred), to_nx!(on_true), to_nx!(on_false))
    end)
  end

  # ── Random ───────────────────────────────────────────────────────

  @spec random_uniform(Nx.Tensor.t(), keyword()) :: t()
  def random_uniform(%Nx.Tensor{} = out, opts) do
    shape = Tuple.to_list(Nx.shape(out))
    type = nx_to_burn_type(Nx.type(out))
    low = opts[:low] || 0.0
    high = opts[:high] || 1.0

    with {:ok, ref} <- ExBurn.NifHelper.random_tensor(shape, Atom.to_string(type), low, high),
         {:ok, shape} <- Nif.tensor_shape(ref) do
      wrap(out, %__MODULE__{ref: ref, shape: shape, type: type})
    else
      {:error, reason} ->
        raise Error, op: :random_uniform, reason: reason
    end
  end

  @spec random_normal(Nx.Tensor.t(), keyword()) :: t()
  def random_normal(%Nx.Tensor{} = out, opts) do
    shape = Tuple.to_list(Nx.shape(out))
    nx_type = Nx.type(out)
    mean = opts[:mean] || 0.0
    std = opts[:std] || 1.0

    # Use Nx.Random.normal for proper normal distribution
    key = Nx.Random.key(System.os_time())
    {nx_tensor, _} = Nx.Random.normal(key, mean, std, shape: List.to_tuple(shape), type: nx_type)

    case from_nx(nx_tensor) do
      {:ok, bt} -> bt
      {:error, reason} -> raise Error, op: :random_normal, reason: reason
    end
  end

  # ── Eye / Iota ───────────────────────────────────────────────────

  @doc false
  @impl true
  @spec eye(Nx.Tensor.t(), keyword()) :: t()
  def eye(%Nx.Tensor{} = out, _opts) do
    shape = Tuple.to_list(Nx.shape(out))
    n = Enum.at(shape, 0) || 1

    with {:ok, ref} <- Nif.eye_tensor(n, :f32),
         {:ok, shape} <- Nif.tensor_shape(ref) do
      wrap(out, %__MODULE__{ref: ref, shape: shape, type: :f32})
    else
      {:error, reason} -> raise Error, op: :eye, reason: reason
    end
  end

  @doc false
  @impl true
  @spec iota(Nx.Tensor.t(), axis :: non_neg_integer() | nil, keyword()) :: t()
  def iota(%Nx.Tensor{} = out, axis, _opts) do
    shape = Tuple.to_list(Nx.shape(out))
    axis_idx = axis || 0

    with {:ok, ref} <- Nif.iota_tensor(shape, axis_idx, :f32),
         {:ok, shape} <- Nif.tensor_shape(ref) do
      wrap(out, %__MODULE__{ref: ref, shape: shape, type: :f32})
    else
      {:error, reason} -> raise Error, op: :iota, reason: reason
    end
  end

  # ── Clip ─────────────────────────────────────────────────────────

  @impl true
  @spec clip(Nx.Tensor.t(), t(), t(), t()) :: t()
  def clip(out, %T{data: %__MODULE__{} = a}, %T{data: %__MODULE__{} = min}, %T{
        data: %__MODULE__{} = max
      }),
      do: clip_bare(out, a, min, max)

  def clip(out, %__MODULE__{} = a, %__MODULE__{} = min, %__MODULE__{} = max),
    do: clip_bare(out, a, min, max)

  defp clip_bare(_out, %__MODULE__{} = a, %__MODULE__{} = min, %__MODULE__{} = max) do
    with {:ok, nx_a} <- to_nx(a),
         {:ok, nx_min} <- to_nx(min),
         {:ok, nx_max} <- to_nx(max),
         result = Nx.clip(nx_a, nx_min, nx_max),
         {:ok, bt} <- from_nx(result) do
      bt
    else
      {:error, reason} -> raise Error, op: :clip, reason: reason
    end
  end

  # ── Convolution ──────────────────────────────────────────────────

  @impl true
  @spec conv(Nx.Tensor.t(), t(), t(), keyword()) :: t()
  def conv(out, %T{data: %__MODULE__{} = i}, %T{data: %__MODULE__{} = k}, opts),
    do: conv_bare(out, i, k, opts)

  def conv(out, %__MODULE__{} = i, %__MODULE__{} = k, opts), do: conv_bare(out, i, k, opts)

  defp conv_bare(out, %__MODULE__{} = input, %__MODULE__{} = kernel, opts) do
    input = maybe_cast(input, :f32)
    kernel = maybe_cast(kernel, :f32)
    stride = opts[:stride] || [1, 1]
    padding = opts[:padding] || [0, 0]

    with {:ok, ref} <- ExBurn.NifHelper.conv2d_tensor(input.ref, kernel.ref, stride, padding),
         {:ok, shape} <- Nif.tensor_shape(ref) do
      wrap(out, %__MODULE__{ref: ref, shape: shape, type: :f32})
    else
      {:error, reason} -> raise Error, op: :conv, reason: reason
    end
  end

  # ── Window Operations ───────────────────────────────────────────
  #
  # These were previously silent no-ops that returned wrong data. They now
  # raise so callers get a clear signal instead of corrupted results.

  @impl true
  def window_sum(_out, _t, _window_dimensions, _opts), do: not_implemented(:window_sum)
  @impl true
  def window_max(_out, _t, _window_dimensions, _opts), do: not_implemented(:window_max)
  @impl true
  def window_min(_out, _t, _window_dimensions, _opts), do: not_implemented(:window_min)
  @impl true
  def window_product(_out, _t, _window_dimensions, _opts), do: not_implemented(:window_product)

  @impl true
  def window_reduce(_out, _t, _acc, _window_dimensions, _opts, _fun),
    do: not_implemented(:window_reduce)

  @impl true
  def window_scatter_max(_out, _t, _source, _init_value, _window_dimensions, _opts),
    do: not_implemented(:window_scatter_max)

  @impl true
  def window_scatter_min(_out, _t, _source, _init_value, _window_dimensions, _opts),
    do: not_implemented(:window_scatter_min)

  # ── Indexed Operations ───────────────────────────────────────────

  @impl true
  def indexed_add(_out, _target, _indices, _updates, _opts), do: not_implemented(:indexed_add)
  @impl true
  def indexed_put(_out, _target, _indices, _updates, _opts), do: not_implemented(:indexed_put)

  # ── Reduce ───────────────────────────────────────────────────────

  @impl true
  @spec reduce(Nx.Tensor.t(), t(), t(), keyword(), fun()) :: t()
  def reduce(_out, %T{data: %__MODULE__{} = tensor}, acc, opts, fun) do
    axes = opts[:axes] || []
    acc_value = scalar_value(acc)

    # An empty axes list means "no reduction" in Nx — omit it for the
    # full-reduction case.
    inner_opts =
      if axes == [],
        do: [keep_axes: false],
        else: [axes: axes, keep_axes: false]

    inner = Nx.reduce(to_nx!(tensor), acc_value, inner_opts, fun)
    nx_fallback(:reduce, fn -> inner end)
  end

  # Reducer accumulators arrive as bare structs (compiler path), wrapped
  # tensors (Nx dispatch), or plain numbers.
  defp scalar_value(%__MODULE__{} = data), do: data |> to_nx!() |> Nx.to_number()
  defp scalar_value(%T{data: %__MODULE__{} = data}), do: data |> to_nx!() |> Nx.to_number()
  defp scalar_value(n) when is_number(n), do: n

  # ── Gather / Scatter ─────────────────────────────────────────────

  @impl true
  @spec gather(Nx.Tensor.t(), t(), t(), keyword()) :: t()
  def gather(out, %T{data: %__MODULE__{} = i}, %T{data: %__MODULE__{} = idx}, opts),
    do: gather_bare(out, i, idx, opts)

  def gather(out, %__MODULE__{} = i, %__MODULE__{} = idx, opts),
    do: gather_bare(out, i, idx, opts)

  defp gather_bare(_out, %__MODULE__{} = input, %__MODULE__{} = indices, _opts) do
    with {:ok, nx_input} <- to_nx(input),
         {:ok, nx_indices} <- to_nx(indices),
         result = Nx.take(nx_input, nx_indices),
         {:ok, bt} <- from_nx(result) do
      bt
    else
      {:error, reason} -> raise Error, op: :gather, reason: reason
    end
  end

  @impl true
  @spec put_slice(Nx.Tensor.t(), t(), t(), [non_neg_integer()]) :: t()
  def put_slice(_out, _target, _start_indices, _slice), do: not_implemented(:put_slice)

  # ── FFT ──────────────────────────────────────────────────────────

  @impl true
  def fft(_out, _tensor, _opts), do: not_implemented(:fft)

  @impl true
  @spec ifft(Nx.Tensor.t(), t(), keyword()) :: t()
  def ifft(_out, %__MODULE__{} = tensor, _opts) do
    tensor
  end

  # ── Sort ─────────────────────────────────────────────────────────

  @impl true
  def sort(_out, _a, _opts), do: not_implemented(:sort)

  @impl true
  def argsort(_out, _a, _opts), do: not_implemented(:argsort)

  # ── Triangular Solve ─────────────────────────────────────────────

  @impl true
  def triangular_solve(_out, _a, _b, _opts), do: not_implemented(:triangular_solve)

  # ── Backend Transfer ─────────────────────────────────────────────

  @impl true
  @spec backend_copy(t(), atom(), keyword()) :: t()
  def backend_copy(%T{data: %__MODULE__{}} = t, _backend, _opts), do: t
  def backend_copy(%__MODULE__{} = data, _backend, _opts), do: data

  @impl true
  @spec backend_transfer(t(), atom(), keyword()) :: t()
  def backend_transfer(%T{data: %__MODULE__{}} = t, _backend, _opts), do: t
  def backend_transfer(%__MODULE__{} = data, _backend, _opts), do: data

  @impl true
  @spec backend_deallocate(t()) :: :ok
  def backend_deallocate(%__MODULE__{ref: ref}) do
    Nif.free_tensor(ref)
  end

  @impl true
  @spec to_batched(Nx.Tensor.t(), t(), keyword()) :: Enumerable.t()
  def to_batched(_out, _tensor, _opts), do: not_implemented(:to_batched)

  # Raises a structured error for callbacks that have no implementation.
  # Returning wrong data silently is worse than failing loudly.
  defp not_implemented(op) do
    raise Error,
      op: op,
      reason: "not implemented by ExBurn.Backend — use Nx.BinaryBackend for this operation"
  end

  @impl true
  @spec as_type(Nx.Tensor.t(), t()) :: t()
  def as_type(out, %__MODULE__{} = a) do
    out_type = nx_to_burn_type(Nx.type(out))
    maybe_cast(a, out_type)
  end

  @impl true
  @spec bitcast(Nx.Tensor.t(), t()) :: t()
  def bitcast(_out, %__MODULE__{} = a) do
    a
  end

  @impl true
  @spec inspect(Nx.Tensor.t(), keyword()) :: Inspect.Algebra.t() | String.t()
  def inspect(%T{data: %__MODULE__{} = tensor}, inspect_opts) do
    # Nx dispatches with %Inspect.Opts{} (a map), while direct callers may
    # pass a keyword list — support both instead of crashing on structs.
    limit =
      if is_map(inspect_opts),
        do: Map.get(inspect_opts, :limit, 50),
        else: Keyword.get(inspect_opts, :limit, 50)

    preview =
      if Enum.product(tensor.shape ++ [1]) <= limit do
        case to_nx(tensor) do
          {:ok, nx_tensor} ->
            values = Nx.to_flat_list(nx_tensor) |> Enum.take(limit)
            " data: #{inspect(values)}"

          {:error, _} ->
            ""
        end
      else
        ""
      end

    Inspect.Algebra.string(
      "#ExBurn.Tensor<shape: #{inspect(tensor.shape)}, type: #{tensor.type}#{preview}>"
    )
  end

  @impl true
  @spec from_pointer(reference(), Nx.Type.t(), [non_neg_integer()], keyword(), keyword()) :: t()
  def from_pointer(_pointer, _type, _shape, _backend_opts, _opts) do
    raise Error, op: :from_pointer, reason: "not supported by ExBurn.Backend"
  end

  @impl true
  @spec to_pointer(t(), keyword()) :: reference()
  def to_pointer(%__MODULE__{}, _opts) do
    raise Error, op: :to_pointer, reason: "not supported by ExBurn.Backend"
  end

  # ── Block Extension ─────────────────────────────────────────────

  @impl true
  @spec block(reference(), t() | tuple(), [term()], fun()) :: t() | tuple()
  def block(_struct, output, args, fun) do
    apply(fun, [output | args])
  end

  # ── Private Helpers ──────────────────────────────────────────────

  @spec nx_to_burn_type(Nx.Type.t()) :: BT.burn_type()
  defp nx_to_burn_type(_type) do
    # The NIF layer stores f32 data only, so the backend downcasts every
    # dtype at this boundary instead of crashing deep in Rust. This is a
    # documented limitation — see README "Known limitations".
    :f32
  end

  @spec burn_to_nx_type(BT.burn_type()) :: Nx.Type.t()
  defp burn_to_nx_type(type), do: BT.burn_to_nx(type)

  @spec maybe_cast(t(), atom()) :: t()
  defp maybe_cast(%__MODULE__{type: :f32} = t, :f32), do: t

  defp maybe_cast(%__MODULE__{ref: ref, shape: shape}, target_type) do
    case Nif.tensor_to_binary(ref) do
      {:ok, binary} ->
        case Nif.new_tensor(binary, shape, Atom.to_string(target_type)) do
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

    case Nif.tensor_to_binary(ref) do
      {:ok, binary} ->
        # Pin the host-side view to BinaryBackend: with a flipped global
        # default backend, leaving it on this backend would make every
        # nx_fallback body re-enter these callbacks recursively.
        tensor =
          binary
          |> Nx.from_binary(nx_type)
          |> Nx.reshape(List.to_tuple(shape))
          |> pin_to_binary_backend()

        {:ok, tensor}

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp pin_to_binary_backend(%T{data: %Nx.BinaryBackend{}} = t), do: t
  defp pin_to_binary_backend(t), do: Nx.backend_transfer(t, Nx.BinaryBackend)

  @spec from_nx(Nx.Tensor.t()) :: {:ok, t()} | {:error, String.t()}
  defp from_nx(%Nx.Tensor{} = tensor) do
    shape = Tuple.to_list(Nx.shape(tensor))
    type = nx_to_burn_type(Nx.type(tensor))

    # Cast to f32 (value-preserving) since the NIF stores f32 only.
    data =
      tensor
      |> Nx.as_type({:f, 32})
      |> Nx.to_binary()

    case Nif.new_tensor(data, shape, Atom.to_string(type)) do
      {:ok, ref} -> {:ok, %__MODULE__{ref: ref, shape: shape, type: type}}
      {:error, reason} -> {:error, reason}
    end
  end

  @type_precedence %{
    f64: 10,
    f32: 9,
    f16: 8,
    bf16: 7,
    i64: 6,
    i32: 5,
    i16: 4,
    i8: 3,
    u8: 2
  }

  defp result_type(type_a, type_b) do
    pa = Map.get(@type_precedence, type_a, 0)
    pb = Map.get(@type_precedence, type_b, 0)
    if pb > pa, do: type_b, else: type_a
  end

  @spec expand_shape([non_neg_integer()], non_neg_integer()) :: [non_neg_integer()]
  defp expand_shape(shape, axis) do
    {list_before, list_after} = Enum.split(shape, axis)
    list_before ++ [1] ++ list_after
  end

  defimpl Nx.Container do
    # A Burn tensor resource holds no traversable Nx tensor children —
    # treat it as an opaque leaf.
    def unwrap(_data, _fun), do: []

    def reduce(_data, acc, _fun), do: acc

    def traverse(data, acc, _fun), do: {data, acc}

    def serialize(_data), do: raise("cannot serialize ExBurn.Backend data directly")
  end
end
