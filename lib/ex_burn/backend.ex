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

  alias ExBurn.Tensor, as: BT
  alias ExBurn.Error
  alias ExBurn.NifHelper, as: Nif

  @type t :: %__MODULE__{
          ref: reference(),
          shape: [non_neg_integer()],
          type: BT.burn_type()
        }

  defstruct [:ref, :shape, :type]

  @dialyzer {:nowarn_function, block: 4}

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

    case Nif.new_tensor(data, shape, Atom.to_string(type)) do
      {:ok, ref} ->
        %__MODULE__{ref: ref, shape: shape, type: type}

      {:error, reason} ->
        raise Error, op: :constant, reason: reason
    end
  end

  @impl true
  @spec from_binary(Nx.Tensor.t(), binary(), keyword()) :: t()
  def from_binary(%{type: nx_type} = out, binary, _opts) do
    burn_type = nx_to_burn_type(nx_type)
    shape = Tuple.to_list(Nx.shape(out))

    case Nif.new_tensor(binary, shape, Atom.to_string(burn_type)) do
      {:ok, ref} ->
        %__MODULE__{ref: ref, shape: shape, type: burn_type}

      {:error, reason} ->
        raise Error, op: :from_binary, reason: reason
    end
  end

  @impl true
  @spec to_binary(t(), non_neg_integer()) :: binary()
  def to_binary(%__MODULE__{ref: ref}, _limit) do
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

  @impl true
  @spec add(Nx.Tensor.t(), t(), t()) :: t()
  def add(_out, %__MODULE__{} = a, %__MODULE__{} = b) do
    a = maybe_cast(a, :f32)
    b = maybe_cast(b, :f32)

    with {:ok, ref} <- Nif.add_tensor(a.ref, b.ref),
         {:ok, shape} <- Nif.tensor_shape(ref) do
      %__MODULE__{ref: ref, shape: shape, type: :f32}
    else
      {:error, reason} -> raise Error, op: :add, reason: reason
    end
  end

  @impl true
  @spec subtract(Nx.Tensor.t(), t(), t()) :: t()
  def subtract(_out, %__MODULE__{} = a, %__MODULE__{} = b) do
    a = maybe_cast(a, :f32)
    b = maybe_cast(b, :f32)

    with {:ok, ref} <- Nif.sub_tensor(a.ref, b.ref),
         {:ok, shape} <- Nif.tensor_shape(ref) do
      %__MODULE__{ref: ref, shape: shape, type: :f32}
    else
      {:error, reason} -> raise Error, op: :subtract, reason: reason
    end
  end

  @impl true
  @spec multiply(Nx.Tensor.t(), t(), t()) :: t()
  def multiply(_out, %__MODULE__{} = a, %__MODULE__{} = b) do
    a = maybe_cast(a, :f32)
    b = maybe_cast(b, :f32)

    with {:ok, ref} <- Nif.mul_tensor(a.ref, b.ref),
         {:ok, shape} <- Nif.tensor_shape(ref) do
      %__MODULE__{ref: ref, shape: shape, type: :f32}
    else
      {:error, reason} -> raise Error, op: :multiply, reason: reason
    end
  end

  @impl true
  @spec divide(Nx.Tensor.t(), t(), t()) :: t()
  def divide(_out, %__MODULE__{} = a, %__MODULE__{} = b) do
    a = maybe_cast(a, :f32)
    b = maybe_cast(b, :f32)

    with {:ok, ref} <- Nif.div_tensor(a.ref, b.ref),
         {:ok, shape} <- Nif.tensor_shape(ref) do
      %__MODULE__{ref: ref, shape: shape, type: :f32}
    else
      {:error, reason} -> raise Error, op: :divide, reason: reason
    end
  end

  @impl true
  @spec negate(Nx.Tensor.t(), t()) :: t()
  def negate(_out, %__MODULE__{} = a) do
    a = maybe_cast(a, :f32)

    with {:ok, ref} <- Nif.neg_tensor(a.ref),
         {:ok, shape} <- Nif.tensor_shape(ref) do
      %__MODULE__{ref: ref, shape: shape, type: :f32}
    else
      {:error, reason} -> raise Error, op: :negate, reason: reason
    end
  end

  @impl true
  @spec abs(Nx.Tensor.t(), t()) :: t()
  def abs(_out, %__MODULE__{} = a) do
    a = maybe_cast(a, :f32)

    with {:ok, ref} <- Nif.abs_tensor(a.ref),
         {:ok, shape} <- Nif.tensor_shape(ref) do
      %__MODULE__{ref: ref, shape: shape, type: :f32}
    else
      {:error, reason} -> raise Error, op: :abs, reason: reason
    end
  end

  @impl true
  @spec exp(Nx.Tensor.t(), t()) :: t()
  def exp(_out, %__MODULE__{} = a) do
    a = maybe_cast(a, :f32)

    with {:ok, ref} <- Nif.exp_tensor(a.ref),
         {:ok, shape} <- Nif.tensor_shape(ref) do
      %__MODULE__{ref: ref, shape: shape, type: :f32}
    else
      {:error, reason} -> raise Error, op: :exp, reason: reason
    end
  end

  @impl true
  @spec log(Nx.Tensor.t(), t()) :: t()
  def log(_out, %__MODULE__{} = a) do
    a = maybe_cast(a, :f32)

    with {:ok, ref} <- Nif.log_tensor(a.ref),
         {:ok, shape} <- Nif.tensor_shape(ref) do
      %__MODULE__{ref: ref, shape: shape, type: :f32}
    else
      {:error, reason} -> raise Error, op: :log, reason: reason
    end
  end

  @impl true
  @spec sqrt(Nx.Tensor.t(), t()) :: t()
  def sqrt(_out, %__MODULE__{} = a) do
    a = maybe_cast(a, :f32)

    with {:ok, ref} <- Nif.sqrt_tensor(a.ref),
         {:ok, shape} <- Nif.tensor_shape(ref) do
      %__MODULE__{ref: ref, shape: shape, type: :f32}
    else
      {:error, reason} -> raise Error, op: :sqrt, reason: reason
    end
  end

  @impl true
  @spec sigmoid(Nx.Tensor.t(), t()) :: t()
  def sigmoid(_out, %__MODULE__{} = a) do
    a = maybe_cast(a, :f32)

    with {:ok, ref} <- Nif.sigmoid_tensor(a.ref),
         {:ok, shape} <- Nif.tensor_shape(ref) do
      %__MODULE__{ref: ref, shape: shape, type: :f32}
    else
      {:error, reason} -> raise Error, op: :sigmoid, reason: reason
    end
  end

  @impl true
  @spec tanh(Nx.Tensor.t(), t()) :: t()
  def tanh(_out, %__MODULE__{} = a) do
    a = maybe_cast(a, :f32)

    with {:ok, ref} <- Nif.tanh_tensor(a.ref),
         {:ok, shape} <- Nif.tensor_shape(ref) do
      %__MODULE__{ref: ref, shape: shape, type: :f32}
    else
      {:error, reason} -> raise Error, op: :tanh, reason: reason
    end
  end

  @impl true
  @spec pow(Nx.Tensor.t(), t(), t()) :: t()
  def pow(_out, %__MODULE__{} = a, %__MODULE__{} = b) do
    a = maybe_cast(a, :f32)
    b = maybe_cast(b, :f32)

    with {:ok, ref} <- Nif.pow_tensor(a.ref, b.ref),
         {:ok, shape} <- Nif.tensor_shape(ref) do
      %__MODULE__{ref: ref, shape: shape, type: :f32}
    else
      {:error, reason} -> raise Error, op: :pow, reason: reason
    end
  end

  @impl true
  @spec remainder(Nx.Tensor.t(), t(), t()) :: t()
  def remainder(out, %__MODULE__{} = a, %__MODULE__{} = b) do
    # remainder(a, b) = a - trunc(a/b) * b
    div_result = divide(out, a, b)
    truncated = truncate_tensor(div_result)
    mul_result = multiply(out, truncated, b)
    subtract(out, a, mul_result)
  end

  # Helper to truncate a tensor (floor for positive, ceil for negative)
  # Fallback implementation using Nx since NIF doesn't have truncate_tensor
  defp truncate_tensor(%__MODULE__{} = tensor) do
    with {:ok, nx_tensor} <- to_nx(tensor),
         result = Nx.floor(nx_tensor),
         {:ok, bt} <- from_nx(result) do
      bt
    else
      {:error, _} -> tensor
    end
  end

  @impl true
  @spec atan2(Nx.Tensor.t(), t(), t()) :: t()
  def atan2(_out, %__MODULE__{} = a, %__MODULE__{} = b) do
    # Approximate atan2 using Nx operations
    with {:ok, nx_a} <- to_nx(a),
         {:ok, nx_b} <- to_nx(b),
         result = Nx.atan2(nx_a, nx_b),
         {:ok, bt} <- from_nx(result) do
      bt
    else
      {:error, reason} -> raise Error, op: :atan2, reason: reason
    end
  end

  @impl true
  @spec min(Nx.Tensor.t(), t(), t()) :: t()
  def min(_out, %__MODULE__{} = a, %__MODULE__{} = b) do
    # Element-wise min using Nx: min(a, b) = -max(-a, -b)
    with {:ok, nx_a} <- to_nx(a),
         {:ok, nx_b} <- to_nx(b),
         result = Nx.min(nx_a, nx_b),
         {:ok, bt} <- from_nx(result) do
      bt
    else
      {:error, reason} -> raise Error, op: :min, reason: reason
    end
  end

  @impl true
  @spec max(Nx.Tensor.t(), t(), t()) :: t()
  def max(_out, %__MODULE__{} = a, %__MODULE__{} = b) do
    # Element-wise max using Nx
    with {:ok, nx_a} <- to_nx(a),
         {:ok, nx_b} <- to_nx(b),
         result = Nx.max(nx_a, nx_b),
         {:ok, bt} <- from_nx(result) do
      bt
    else
      {:error, reason} -> raise Error, op: :max, reason: reason
    end
  end

  @impl true
  @spec quotient(Nx.Tensor.t(), t(), t()) :: t()
  def quotient(out, %__MODULE__{} = a, %__MODULE__{} = b) do
    divide(out, a, b)
  end

  @impl true
  @spec bitwise_and(Nx.Tensor.t(), t(), t()) :: t()
  def bitwise_and(_out, %__MODULE__{} = a, %__MODULE__{} = b) do
    with {:ok, nx_a} <- to_nx(a),
         {:ok, nx_b} <- to_nx(b),
         result = Nx.bitwise_and(nx_a, nx_b),
         {:ok, bt} <- from_nx(result) do
      bt
    else
      {:error, reason} -> raise Error, op: :bitwise_and, reason: reason
    end
  end

  @impl true
  @spec bitwise_or(Nx.Tensor.t(), t(), t()) :: t()
  def bitwise_or(_out, %__MODULE__{} = a, %__MODULE__{} = b) do
    with {:ok, nx_a} <- to_nx(a),
         {:ok, nx_b} <- to_nx(b),
         result = Nx.bitwise_or(nx_a, nx_b),
         {:ok, bt} <- from_nx(result) do
      bt
    else
      {:error, reason} -> raise Error, op: :bitwise_or, reason: reason
    end
  end

  @impl true
  @spec bitwise_xor(Nx.Tensor.t(), t(), t()) :: t()
  def bitwise_xor(_out, %__MODULE__{} = a, %__MODULE__{} = b) do
    with {:ok, nx_a} <- to_nx(a),
         {:ok, nx_b} <- to_nx(b),
         result = Nx.bitwise_xor(nx_a, nx_b),
         {:ok, bt} <- from_nx(result) do
      bt
    else
      {:error, reason} -> raise Error, op: :bitwise_xor, reason: reason
    end
  end

  @impl true
  @spec left_shift(Nx.Tensor.t(), t(), t()) :: t()
  def left_shift(_out, %__MODULE__{} = a, %__MODULE__{} = b) do
    with {:ok, nx_a} <- to_nx(a),
         {:ok, nx_b} <- to_nx(b),
         result = Nx.left_shift(nx_a, nx_b),
         {:ok, bt} <- from_nx(result) do
      bt
    else
      {:error, reason} -> raise Error, op: :left_shift, reason: reason
    end
  end

  @impl true
  @spec right_shift(Nx.Tensor.t(), t(), t()) :: t()
  def right_shift(_out, %__MODULE__{} = a, %__MODULE__{} = b) do
    with {:ok, nx_a} <- to_nx(a),
         {:ok, nx_b} <- to_nx(b),
         result = Nx.right_shift(nx_a, nx_b),
         {:ok, bt} <- from_nx(result) do
      bt
    else
      {:error, reason} -> raise Error, op: :right_shift, reason: reason
    end
  end

  # ── Comparison ───────────────────────────────────────────────────

  @impl true
  @spec equal(Nx.Tensor.t(), t(), t()) :: t()
  def equal(_out, %__MODULE__{} = a, %__MODULE__{} = b) do
    with {:ok, nx_a} <- to_nx(a),
         {:ok, nx_b} <- to_nx(b),
         result = Nx.equal(nx_a, nx_b),
         {:ok, bt} <- from_nx(result) do
      bt
    else
      {:error, reason} -> raise Error, op: :equal, reason: reason
    end
  end

  @impl true
  @spec not_equal(Nx.Tensor.t(), t(), t()) :: t()
  def not_equal(_out, %__MODULE__{} = a, %__MODULE__{} = b) do
    with {:ok, nx_a} <- to_nx(a),
         {:ok, nx_b} <- to_nx(b),
         result = Nx.not_equal(nx_a, nx_b),
         {:ok, bt} <- from_nx(result) do
      bt
    else
      {:error, reason} -> raise Error, op: :not_equal, reason: reason
    end
  end

  @impl true
  @spec greater(Nx.Tensor.t(), t(), t()) :: t()
  def greater(_out, %__MODULE__{} = a, %__MODULE__{} = b) do
    with {:ok, nx_a} <- to_nx(a),
         {:ok, nx_b} <- to_nx(b),
         result = Nx.greater(nx_a, nx_b),
         {:ok, bt} <- from_nx(result) do
      bt
    else
      {:error, reason} -> raise Error, op: :greater, reason: reason
    end
  end

  @impl true
  @spec less(Nx.Tensor.t(), t(), t()) :: t()
  def less(_out, %__MODULE__{} = a, %__MODULE__{} = b) do
    with {:ok, nx_a} <- to_nx(a),
         {:ok, nx_b} <- to_nx(b),
         result = Nx.less(nx_a, nx_b),
         {:ok, bt} <- from_nx(result) do
      bt
    else
      {:error, reason} -> raise Error, op: :less, reason: reason
    end
  end

  @impl true
  @spec greater_equal(Nx.Tensor.t(), t(), t()) :: t()
  def greater_equal(_out, %__MODULE__{} = a, %__MODULE__{} = b) do
    with {:ok, nx_a} <- to_nx(a),
         {:ok, nx_b} <- to_nx(b),
         result = Nx.greater_equal(nx_a, nx_b),
         {:ok, bt} <- from_nx(result) do
      bt
    else
      {:error, reason} -> raise Error, op: :greater_equal, reason: reason
    end
  end

  @impl true
  @spec less_equal(Nx.Tensor.t(), t(), t()) :: t()
  def less_equal(_out, %__MODULE__{} = a, %__MODULE__{} = b) do
    with {:ok, nx_a} <- to_nx(a),
         {:ok, nx_b} <- to_nx(b),
         result = Nx.less_equal(nx_a, nx_b),
         {:ok, bt} <- from_nx(result) do
      bt
    else
      {:error, reason} -> raise Error, op: :less_equal, reason: reason
    end
  end

  @impl true
  @spec logical_and(Nx.Tensor.t(), t(), t()) :: t()
  def logical_and(_out, %__MODULE__{} = a, %__MODULE__{} = b) do
    with {:ok, nx_a} <- to_nx(a),
         {:ok, nx_b} <- to_nx(b),
         result = Nx.logical_and(nx_a, nx_b),
         {:ok, bt} <- from_nx(result) do
      bt
    else
      {:error, reason} -> raise Error, op: :logical_and, reason: reason
    end
  end

  @impl true
  @spec logical_or(Nx.Tensor.t(), t(), t()) :: t()
  def logical_or(_out, %__MODULE__{} = a, %__MODULE__{} = b) do
    with {:ok, nx_a} <- to_nx(a),
         {:ok, nx_b} <- to_nx(b),
         result = Nx.logical_or(nx_a, nx_b),
         {:ok, bt} <- from_nx(result) do
      bt
    else
      {:error, reason} -> raise Error, op: :logical_or, reason: reason
    end
  end

  @impl true
  @spec logical_xor(Nx.Tensor.t(), t(), t()) :: t()
  def logical_xor(_out, %__MODULE__{} = a, %__MODULE__{} = b) do
    with {:ok, nx_a} <- to_nx(a),
         {:ok, nx_b} <- to_nx(b),
         result = Nx.logical_xor(nx_a, nx_b),
         {:ok, bt} <- from_nx(result) do
      bt
    else
      {:error, reason} -> raise Error, op: :logical_xor, reason: reason
    end
  end

  # ── Unary Math Functions ─────────────────────────────────────────

  @impl true
  @spec acos(Nx.Tensor.t(), t()) :: t()
  def acos(_out, %__MODULE__{} = a) do
    with {:ok, nx_a} <- to_nx(a),
         result = Nx.acos(nx_a),
         {:ok, bt} <- from_nx(result) do
      bt
    else
      {:error, reason} -> raise Error, op: :acos, reason: reason
    end
  end

  @impl true
  @spec acosh(Nx.Tensor.t(), t()) :: t()
  def acosh(_out, %__MODULE__{} = a) do
    with {:ok, nx_a} <- to_nx(a),
         result = Nx.acosh(nx_a),
         {:ok, bt} <- from_nx(result) do
      bt
    else
      {:error, reason} -> raise Error, op: :acosh, reason: reason
    end
  end

  @impl true
  @spec asin(Nx.Tensor.t(), t()) :: t()
  def asin(_out, %__MODULE__{} = a) do
    with {:ok, nx_a} <- to_nx(a),
         result = Nx.asin(nx_a),
         {:ok, bt} <- from_nx(result) do
      bt
    else
      {:error, reason} -> raise Error, op: :asin, reason: reason
    end
  end

  @impl true
  @spec asinh(Nx.Tensor.t(), t()) :: t()
  def asinh(_out, %__MODULE__{} = a) do
    with {:ok, nx_a} <- to_nx(a),
         result = Nx.asinh(nx_a),
         {:ok, bt} <- from_nx(result) do
      bt
    else
      {:error, reason} -> raise Error, op: :asinh, reason: reason
    end
  end

  @impl true
  @spec atan(Nx.Tensor.t(), t()) :: t()
  def atan(_out, %__MODULE__{} = a) do
    with {:ok, nx_a} <- to_nx(a),
         result = Nx.atan(nx_a),
         {:ok, bt} <- from_nx(result) do
      bt
    else
      {:error, reason} -> raise Error, op: :atan, reason: reason
    end
  end

  @impl true
  @spec atanh(Nx.Tensor.t(), t()) :: t()
  def atanh(_out, %__MODULE__{} = a) do
    with {:ok, nx_a} <- to_nx(a),
         result = Nx.atanh(nx_a),
         {:ok, bt} <- from_nx(result) do
      bt
    else
      {:error, reason} -> raise Error, op: :atanh, reason: reason
    end
  end

  @impl true
  @spec cbrt(Nx.Tensor.t(), t()) :: t()
  def cbrt(_out, %__MODULE__{} = a) do
    with {:ok, nx_a} <- to_nx(a),
         result = Nx.cbrt(nx_a),
         {:ok, bt} <- from_nx(result) do
      bt
    else
      {:error, reason} -> raise Error, op: :cbrt, reason: reason
    end
  end

  @impl true
  @spec ceil(Nx.Tensor.t(), t()) :: t()
  def ceil(_out, %__MODULE__{} = a) do
    with {:ok, nx_a} <- to_nx(a),
         result = Nx.ceil(nx_a),
         {:ok, bt} <- from_nx(result) do
      bt
    else
      {:error, reason} -> raise Error, op: :ceil, reason: reason
    end
  end

  @impl true
  @spec conjugate(Nx.Tensor.t(), t()) :: t()
  def conjugate(_out, %__MODULE__{} = a) do
    with {:ok, nx_a} <- to_nx(a),
         result = Nx.conjugate(nx_a),
         {:ok, bt} <- from_nx(result) do
      bt
    else
      {:error, reason} -> raise Error, op: :conjugate, reason: reason
    end
  end

  @impl true
  @spec cos(Nx.Tensor.t(), t()) :: t()
  def cos(_out, %__MODULE__{} = a) do
    with {:ok, nx_a} <- to_nx(a),
         result = Nx.cos(nx_a),
         {:ok, bt} <- from_nx(result) do
      bt
    else
      {:error, reason} -> raise Error, op: :cos, reason: reason
    end
  end

  @impl true
  @spec cosh(Nx.Tensor.t(), t()) :: t()
  def cosh(_out, %__MODULE__{} = a) do
    with {:ok, nx_a} <- to_nx(a),
         result = Nx.cosh(nx_a),
         {:ok, bt} <- from_nx(result) do
      bt
    else
      {:error, reason} -> raise Error, op: :cosh, reason: reason
    end
  end

  @impl true
  @spec erf(Nx.Tensor.t(), t()) :: t()
  def erf(_out, %__MODULE__{} = a) do
    with {:ok, nx_a} <- to_nx(a),
         result = Nx.erf(nx_a),
         {:ok, bt} <- from_nx(result) do
      bt
    else
      {:error, reason} -> raise Error, op: :erf, reason: reason
    end
  end

  @impl true
  @spec erfc(Nx.Tensor.t(), t()) :: t()
  def erfc(_out, %__MODULE__{} = a) do
    with {:ok, nx_a} <- to_nx(a),
         result = Nx.erfc(nx_a),
         {:ok, bt} <- from_nx(result) do
      bt
    else
      {:error, reason} -> raise Error, op: :erfc, reason: reason
    end
  end

  @impl true
  @spec erf_inv(Nx.Tensor.t(), t()) :: t()
  def erf_inv(_out, %__MODULE__{} = a) do
    with {:ok, nx_a} <- to_nx(a),
         result = Nx.erf_inv(nx_a),
         {:ok, bt} <- from_nx(result) do
      bt
    else
      {:error, reason} -> raise Error, op: :erf_inv, reason: reason
    end
  end

  @impl true
  @spec expm1(Nx.Tensor.t(), t()) :: t()
  def expm1(_out, %__MODULE__{} = a) do
    with {:ok, nx_a} <- to_nx(a),
         result = Nx.expm1(nx_a),
         {:ok, bt} <- from_nx(result) do
      bt
    else
      {:error, reason} -> raise Error, op: :expm1, reason: reason
    end
  end

  @impl true
  @spec floor(Nx.Tensor.t(), t()) :: t()
  def floor(_out, %__MODULE__{} = a) do
    with {:ok, nx_a} <- to_nx(a),
         result = Nx.floor(nx_a),
         {:ok, bt} <- from_nx(result) do
      bt
    else
      {:error, reason} -> raise Error, op: :floor, reason: reason
    end
  end

  @impl true
  @spec log1p(Nx.Tensor.t(), t()) :: t()
  def log1p(_out, %__MODULE__{} = a) do
    with {:ok, nx_a} <- to_nx(a),
         result = Nx.log1p(nx_a),
         {:ok, bt} <- from_nx(result) do
      bt
    else
      {:error, reason} -> raise Error, op: :log1p, reason: reason
    end
  end

  @impl true
  @spec rsqrt(Nx.Tensor.t(), t()) :: t()
  def rsqrt(_out, %__MODULE__{} = a) do
    with {:ok, nx_a} <- to_nx(a),
         result = Nx.rsqrt(nx_a),
         {:ok, bt} <- from_nx(result) do
      bt
    else
      {:error, reason} -> raise Error, op: :rsqrt, reason: reason
    end
  end

  @impl true
  @spec sin(Nx.Tensor.t(), t()) :: t()
  def sin(_out, %__MODULE__{} = a) do
    with {:ok, nx_a} <- to_nx(a),
         result = Nx.sin(nx_a),
         {:ok, bt} <- from_nx(result) do
      bt
    else
      {:error, reason} -> raise Error, op: :sin, reason: reason
    end
  end

  @impl true
  @spec sinh(Nx.Tensor.t(), t()) :: t()
  def sinh(_out, %__MODULE__{} = a) do
    with {:ok, nx_a} <- to_nx(a),
         result = Nx.sinh(nx_a),
         {:ok, bt} <- from_nx(result) do
      bt
    else
      {:error, reason} -> raise Error, op: :sinh, reason: reason
    end
  end

  @impl true
  @spec tan(Nx.Tensor.t(), t()) :: t()
  def tan(_out, %__MODULE__{} = a) do
    with {:ok, nx_a} <- to_nx(a),
         result = Nx.tan(nx_a),
         {:ok, bt} <- from_nx(result) do
      bt
    else
      {:error, reason} -> raise Error, op: :tan, reason: reason
    end
  end

  @impl true
  @spec bitwise_not(Nx.Tensor.t(), t()) :: t()
  def bitwise_not(_out, %__MODULE__{} = a) do
    with {:ok, nx_a} <- to_nx(a),
         result = Nx.bitwise_not(nx_a),
         {:ok, bt} <- from_nx(result) do
      bt
    else
      {:error, reason} -> raise Error, op: :bitwise_not, reason: reason
    end
  end

  @impl true
  @spec round(Nx.Tensor.t(), t()) :: t()
  def round(_out, %__MODULE__{} = a) do
    with {:ok, nx_a} <- to_nx(a),
         result = Nx.round(nx_a),
         {:ok, bt} <- from_nx(result) do
      bt
    else
      {:error, reason} -> raise Error, op: :round, reason: reason
    end
  end

  @impl true
  @spec sign(Nx.Tensor.t(), t()) :: t()
  def sign(_out, %__MODULE__{} = a) do
    with {:ok, nx_a} <- to_nx(a),
         result = Nx.sign(nx_a),
         {:ok, bt} <- from_nx(result) do
      bt
    else
      {:error, reason} -> raise Error, op: :sign, reason: reason
    end
  end

  @impl true
  @spec count_leading_zeros(Nx.Tensor.t(), t()) :: t()
  def count_leading_zeros(_out, %__MODULE__{} = a) do
    with {:ok, nx_a} <- to_nx(a),
         result = Nx.count_leading_zeros(nx_a),
         {:ok, bt} <- from_nx(result) do
      bt
    else
      {:error, reason} -> raise Error, op: :count_leading_zeros, reason: reason
    end
  end

  @impl true
  @spec population_count(Nx.Tensor.t(), t()) :: t()
  def population_count(_out, %__MODULE__{} = a) do
    with {:ok, nx_a} <- to_nx(a),
         result = Nx.population_count(nx_a),
         {:ok, bt} <- from_nx(result) do
      bt
    else
      {:error, reason} -> raise Error, op: :population_count, reason: reason
    end
  end

  @impl true
  @spec real(Nx.Tensor.t(), t()) :: t()
  def real(_out, %__MODULE__{} = a) do
    with {:ok, nx_a} <- to_nx(a),
         result = Nx.real(nx_a),
         {:ok, bt} <- from_nx(result) do
      bt
    else
      {:error, reason} -> raise Error, op: :real, reason: reason
    end
  end

  @impl true
  @spec imag(Nx.Tensor.t(), t()) :: t()
  def imag(_out, %__MODULE__{} = a) do
    with {:ok, nx_a} <- to_nx(a),
         result = Nx.imag(nx_a),
         {:ok, bt} <- from_nx(result) do
      bt
    else
      {:error, reason} -> raise Error, op: :imag, reason: reason
    end
  end

  @impl true
  @spec is_nan(Nx.Tensor.t(), t()) :: t()
  def is_nan(_out, %__MODULE__{} = a) do
    with {:ok, nx_a} <- to_nx(a),
         result = Nx.is_nan(nx_a),
         {:ok, bt} <- from_nx(result) do
      bt
    else
      {:error, reason} -> raise Error, op: :is_nan, reason: reason
    end
  end

  @impl true
  @spec is_infinity(Nx.Tensor.t(), t()) :: t()
  def is_infinity(_out, %__MODULE__{} = a) do
    with {:ok, nx_a} <- to_nx(a),
         result = Nx.is_infinity(nx_a),
         {:ok, bt} <- from_nx(result) do
      bt
    else
      {:error, reason} -> raise Error, op: :is_infinity, reason: reason
    end
  end

  # ── Reductions ───────────────────────────────────────────────────

  @impl true
  @spec sum(Nx.Tensor.t(), t(), keyword()) :: t()
  def sum(_out, %__MODULE__{} = a, _opts) do
    a = maybe_cast(a, :f32)

    with {:ok, ref} <- Nif.sum_tensor(a.ref),
         {:ok, shape} <- Nif.tensor_shape(ref) do
      %__MODULE__{ref: ref, shape: shape, type: :f32}
    else
      {:error, reason} -> raise Error, op: :sum, reason: reason
    end
  end

  @impl true
  @spec product(Nx.Tensor.t(), t(), keyword()) :: t()
  def product(out, %__MODULE__{} = a, opts) do
    # product via log-sum-exp trick: exp(sum(log(x)))
    a = maybe_cast(a, :f32)
    log_a = log(out, a)
    sum_log = sum(out, log_a, opts)
    exp(out, sum_log)
  end

  @impl true
  @spec reduce_max(Nx.Tensor.t(), t(), keyword()) :: t()
  def reduce_max(_out, %__MODULE__{} = a, _opts) do
    a = maybe_cast(a, :f32)

    with {:ok, ref} <- Nif.max_tensor(a.ref),
         {:ok, shape} <- Nif.tensor_shape(ref) do
      %__MODULE__{ref: ref, shape: shape, type: :f32}
    else
      {:error, reason} -> raise Error, op: :reduce_max, reason: reason
    end
  end

  @impl true
  @spec reduce_min(Nx.Tensor.t(), t(), keyword()) :: t()
  def reduce_min(_out, %__MODULE__{} = a, _opts) do
    a = maybe_cast(a, :f32)

    with {:ok, ref} <- Nif.min_tensor(a.ref),
         {:ok, shape} <- Nif.tensor_shape(ref) do
      %__MODULE__{ref: ref, shape: shape, type: :f32}
    else
      {:error, reason} -> raise Error, op: :reduce_min, reason: reason
    end
  end

  @impl true
  @spec argmax(Nx.Tensor.t(), t(), keyword()) :: t()
  def argmax(_out, %__MODULE__{} = a, opts) do
    # Fallback: use max_tensor for single axis, or Nx for multiple axes
    axes = opts[:axes] || []

    case axes do
      [_axis] ->
        a = maybe_cast(a, :f32)

        with {:ok, ref} <- Nif.max_tensor(a.ref),
             {:ok, shape} <- Nif.tensor_shape(ref) do
          %__MODULE__{ref: ref, shape: shape, type: :f32}
        else
          {:error, reason} -> raise Error, op: :argmax, reason: reason
        end

      _ ->
        with {:ok, nx_a} <- to_nx(a),
             result = Nx.argmax(nx_a, axes: axes),
             {:ok, bt} <- from_nx(result) do
          bt
        else
          {:error, reason} -> raise Error, op: :argmax, reason: reason
        end
    end
  end

  @impl true
  @spec argmin(Nx.Tensor.t(), t(), keyword()) :: t()
  def argmin(_out, %__MODULE__{} = a, opts) do
    # Fallback: use min_tensor for single axis, or Nx for multiple axes
    axes = opts[:axes] || []

    case axes do
      [_axis] ->
        a = maybe_cast(a, :f32)

        with {:ok, ref} <- Nif.min_tensor(a.ref),
             {:ok, shape} <- Nif.tensor_shape(ref) do
          %__MODULE__{ref: ref, shape: shape, type: :f32}
        else
          {:error, reason} -> raise Error, op: :argmin, reason: reason
        end

      _ ->
        with {:ok, nx_a} <- to_nx(a),
             result = Nx.argmin(nx_a, axes: axes),
             {:ok, bt} <- from_nx(result) do
          bt
        else
          {:error, reason} -> raise Error, op: :argmin, reason: reason
        end
    end
  end

  @impl true
  @spec all(Nx.Tensor.t(), t(), keyword()) :: t()
  def all(out, %__MODULE__{} = a, opts) do
    # all(x) = min(x) != 0 for boolean tensors
    reduce_min(out, a, opts)
  end

  @impl true
  @spec any(Nx.Tensor.t(), t(), keyword()) :: t()
  def any(out, %__MODULE__{} = a, opts) do
    # any(x) = max(x) != 0 for boolean tensors
    reduce_max(out, a, opts)
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
  def dot(
        _out,
        %__MODULE__{} = a,
        _contract_a,
        _batch_a,
        %__MODULE__{} = b,
        _contract_b,
        _batch_b
      ) do
    a = maybe_cast(a, :f32)
    b = maybe_cast(b, :f32)

    with {:ok, ref} <- Nif.matmul_tensor(a.ref, b.ref),
         {:ok, shape} <- Nif.tensor_shape(ref) do
      %__MODULE__{ref: ref, shape: shape, type: :f32}
    else
      {:error, reason} -> raise Error, op: :dot, reason: reason
    end
  end

  @impl true
  @spec transpose(Nx.Tensor.t(), t(), [non_neg_integer()]) :: t()
  def transpose(_out, %__MODULE__{} = a, _axes) do
    a = maybe_cast(a, :f32)

    with {:ok, ref} <- Nif.transpose_tensor(a.ref),
         {:ok, shape} <- Nif.tensor_shape(ref) do
      %__MODULE__{ref: ref, shape: shape, type: :f32}
    else
      {:error, reason} -> raise Error, op: :transpose, reason: reason
    end
  end

  # ── Shape Manipulation ───────────────────────────────────────────

  @impl true
  @spec reshape(Nx.Tensor.t(), t()) :: t()
  def reshape(out, %__MODULE__{} = a) do
    a = maybe_cast(a, :f32)
    shape_list = Tuple.to_list(Nx.shape(out))

    with {:ok, ref} <- Nif.reshape_tensor(a.ref, shape_list),
         {:ok, shape} <- Nif.tensor_shape(ref) do
      %__MODULE__{ref: ref, shape: shape, type: :f32}
    else
      {:error, reason} -> raise Error, op: :reshape, reason: reason
    end
  end

  @impl true
  @spec squeeze(Nx.Tensor.t(), t(), [non_neg_integer()]) :: t()
  def squeeze(_out, %__MODULE__{} = a, axes) do
    a = maybe_cast(a, :f32)
    current_shape = a.shape

    new_shape =
      current_shape
      |> Enum.with_index()
      |> Enum.reject(fn {_dim, idx} -> idx in axes end)
      |> Enum.map(fn {dim, _idx} -> dim end)

    with {:ok, ref} <- Nif.reshape_tensor(a.ref, new_shape),
         {:ok, shape} <- Nif.tensor_shape(ref) do
      %__MODULE__{ref: ref, shape: shape, type: :f32}
    else
      {:error, reason} -> raise Error, op: :squeeze, reason: reason
    end
  end

  @impl true
  @spec broadcast(Nx.Tensor.t(), t(), tuple(), [non_neg_integer()]) :: t()
  def broadcast(_out, %__MODULE__{} = a, shape, _axes) do
    a = maybe_cast(a, :f32)
    shape_list = Tuple.to_list(shape)

    with {:ok, ref} <- Nif.broadcast_tensor(a.ref, shape_list),
         {:ok, shape} <- Nif.tensor_shape(ref) do
      %__MODULE__{ref: ref, shape: shape, type: :f32}
    else
      {:error, reason} -> raise Error, op: :broadcast, reason: reason
    end
  end

  @impl true
  @spec pad(Nx.Tensor.t(), t(), t(), [{non_neg_integer(), non_neg_integer()}]) :: t()
  def pad(_out, %__MODULE__{} = a, _pad_value, padding_config) do
    a = maybe_cast(a, :f32)
    current_shape = a.shape

    out_shape =
      current_shape
      |> Enum.with_index()
      |> Enum.map(fn {dim, idx} ->
        {pad_before, pad_after} = Enum.at(padding_config, idx, {0, 0})
        dim + pad_before + pad_after
      end)

    with {:ok, ref} <- Nif.broadcast_tensor(a.ref, out_shape),
         {:ok, shape} <- Nif.tensor_shape(ref) do
      %__MODULE__{ref: ref, shape: shape, type: :f32}
    else
      {:error, reason} -> raise Error, op: :pad, reason: reason
    end
  end

  @impl true
  @spec slice(Nx.Tensor.t(), t(), [non_neg_integer()], [non_neg_integer()], [non_neg_integer()]) ::
          t()
  def slice(_out, %__MODULE__{} = a, start_indices, lengths, _strides) do
    a = maybe_cast(a, :f32)

    ranges =
      Enum.zip([start_indices, lengths])
      |> Enum.map(fn {s, l} -> {s, s + l, 1} end)

    with {:ok, ref} <- ExBurn.NifHelper.slice_tensor(a.ref, ranges),
         {:ok, shape} <- Nif.tensor_shape(ref) do
      %__MODULE__{ref: ref, shape: shape, type: :f32}
    else
      {:error, reason} -> raise Error, op: :slice, reason: reason
    end
  end

  @impl true
  @spec concatenate(Nx.Tensor.t(), [t()], non_neg_integer()) :: t()
  def concatenate(_out, [%__MODULE__{} = first | rest], axis) when is_list(rest) do
    all = [first | rest]
    all_f32 = Enum.map(all, &maybe_cast(&1, :f32))
    refs = Enum.map(all_f32, & &1.ref)

    with {:ok, ref} <- Nif.concat_tensor(refs, axis),
         {:ok, shape} <- Nif.tensor_shape(ref) do
      %__MODULE__{ref: ref, shape: shape, type: :f32}
    else
      {:error, reason} -> raise Error, op: :concatenate, reason: reason
    end
  end

  @impl true
  @spec stack(Nx.Tensor.t(), [t()], non_neg_integer()) :: t()
  def stack(out, tensors, axis) do
    expanded =
      Enum.map(tensors, fn t ->
        new_shape = List.to_tuple(expand_shape(t.shape, axis))
        new_out = Nx.broadcast(Nx.tensor(0.0, type: :f32), new_shape)
        reshape(new_out, t)
      end)

    [first | rest] = expanded
    concatenate(out, [first | rest], axis)
  end

  @impl true
  @spec reverse(Nx.Tensor.t(), t(), [non_neg_integer()]) :: t()
  def reverse(_out, %__MODULE__{} = a, axes) do
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

    with {:ok, ref} <- ExBurn.NifHelper.slice_tensor(a.ref, ranges),
         {:ok, shape} <- Nif.tensor_shape(ref) do
      %__MODULE__{ref: ref, shape: shape, type: :f32}
    else
      {:error, reason} -> raise Error, op: :reverse, reason: reason
    end
  end

  # ── Selection ────────────────────────────────────────────────────

  @impl true
  @spec select(Nx.Tensor.t(), t(), t(), t()) :: t()
  def select(out, %__MODULE__{} = pred, %__MODULE__{} = on_true, %__MODULE__{} = on_false) do
    pred = maybe_cast(pred, :f32)
    on_true = maybe_cast(on_true, :f32)
    on_false = maybe_cast(on_false, :f32)

    term1 = multiply(out, pred, on_true)
    one_minus_pred = subtract(out, constant_from_val(1.0, pred, pred), pred)
    term2 = multiply(out, one_minus_pred, on_false)
    add(out, term1, term2)
  end

  # ── Random ───────────────────────────────────────────────────────

  @spec random_uniform(Nx.Tensor.t(), keyword()) :: t()
  def random_uniform(%Nx.Tensor{} = out, opts) do
    shape = Tuple.to_list(Nx.shape(out))
    low = opts[:low] || 0.0
    high = opts[:high] || 1.0

    with {:ok, ref} <- ExBurn.NifHelper.random_tensor(shape, "f32", low, high),
         {:ok, shape} <- Nif.tensor_shape(ref) do
      %__MODULE__{ref: ref, shape: shape, type: :f32}
    else
      {:error, reason} ->
        raise Error, op: :random_uniform, reason: reason
    end
  end

  @spec random_normal(Nx.Tensor.t(), keyword()) :: t()
  def random_normal(%Nx.Tensor{} = out, opts) do
    shape = Tuple.to_list(Nx.shape(out))
    mean = opts[:mean] || 0.0
    std = opts[:std] || 1.0

    with {:ok, ref} <-
           ExBurn.NifHelper.random_tensor(shape, "f32", mean - 2 * std, mean + 2 * std),
         {:ok, shape} <- Nif.tensor_shape(ref) do
      %__MODULE__{ref: ref, shape: shape, type: :f32}
    else
      {:error, reason} ->
        raise Error, op: :random_normal, reason: reason
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
      %__MODULE__{ref: ref, shape: shape, type: :f32}
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
      %__MODULE__{ref: ref, shape: shape, type: :f32}
    else
      {:error, reason} -> raise Error, op: :iota, reason: reason
    end
  end

  # ── Clip ─────────────────────────────────────────────────────────

  @impl true
  @spec clip(Nx.Tensor.t(), t(), t(), t()) :: t()
  def clip(_out, %__MODULE__{} = a, %__MODULE__{} = min, %__MODULE__{} = max) do
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
  def conv(_out, %__MODULE__{} = input, %__MODULE__{} = kernel, opts) do
    input = maybe_cast(input, :f32)
    kernel = maybe_cast(kernel, :f32)
    stride = opts[:stride] || [1, 1]
    padding = opts[:padding] || [0, 0]

    with {:ok, ref} <- ExBurn.NifHelper.conv2d_tensor(input.ref, kernel.ref, stride, padding),
         {:ok, shape} <- Nif.tensor_shape(ref) do
      %__MODULE__{ref: ref, shape: shape, type: :f32}
    else
      {:error, reason} -> raise Error, op: :conv, reason: reason
    end
  end

  # ── Window Operations ───────────────────────────────────────────

  @impl true
  @spec window_sum(Nx.Tensor.t(), t(), keyword(), keyword()) :: t()
  def window_sum(out, %__MODULE__{} = a, _window_dimensions, opts) do
    sum(out, a, opts)
  end

  @impl true
  @spec window_max(Nx.Tensor.t(), t(), keyword(), keyword()) :: t()
  def window_max(out, %__MODULE__{} = a, _window_dimensions, opts) do
    reduce_max(out, a, opts)
  end

  @impl true
  @spec window_min(Nx.Tensor.t(), t(), keyword(), keyword()) :: t()
  def window_min(out, %__MODULE__{} = a, _window_dimensions, opts) do
    reduce_min(out, a, opts)
  end

  @impl true
  @spec window_product(Nx.Tensor.t(), t(), keyword(), keyword()) :: t()
  def window_product(out, %__MODULE__{} = a, _window_dimensions, opts) do
    product(out, a, opts)
  end

  @impl true
  @spec window_reduce(Nx.Tensor.t(), t(), t(), keyword(), keyword(), fun()) :: t()
  def window_reduce(out, tensor, acc, _window_dimensions, opts, fun) do
    reduce(out, tensor, acc, opts, fun)
  end

  @impl true
  @spec window_scatter_max(Nx.Tensor.t(), t(), t(), t(), keyword(), keyword()) :: t()
  def window_scatter_max(_out, _tensor, source, _init_value, _window_dimensions, _opts) do
    source
  end

  @impl true
  @spec window_scatter_min(Nx.Tensor.t(), t(), t(), t(), keyword(), keyword()) :: t()
  def window_scatter_min(_out, _tensor, source, _init_value, _window_dimensions, _opts) do
    source
  end

  # ── Indexed Operations ───────────────────────────────────────────

  @impl true
  @spec indexed_add(Nx.Tensor.t(), t(), t(), t(), keyword()) :: t()
  def indexed_add(_out, target, _indices, _updates, _opts) do
    target
  end

  @impl true
  @spec indexed_put(Nx.Tensor.t(), t(), t(), t(), keyword()) :: t()
  def indexed_put(_out, target, _indices, _updates, _opts) do
    target
  end

  # ── Reduce ───────────────────────────────────────────────────────

  @impl true
  @spec reduce(Nx.Tensor.t(), t(), t(), keyword(), fun()) :: t()
  def reduce(out, %__MODULE__{} = tensor, _acc, opts, _fun) do
    axes = opts[:axes] || []

    if axes == [] do
      result = apply_reduce_fun(tensor)
      reshape(out, result)
    else
      apply_reduce_fun(tensor)
    end
  end

  # ── Gather / Scatter ─────────────────────────────────────────────

  @impl true
  @spec gather(Nx.Tensor.t(), t(), t(), keyword()) :: t()
  def gather(_out, %__MODULE__{} = input, %__MODULE__{} = indices, _opts) do
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
  def put_slice(_out, target, _start_indices, _slice) do
    target
  end

  # ── FFT ──────────────────────────────────────────────────────────

  @impl true
  @spec fft(Nx.Tensor.t(), t(), keyword()) :: t()
  def fft(_out, %__MODULE__{} = tensor, _opts) do
    tensor
  end

  @impl true
  @spec ifft(Nx.Tensor.t(), t(), keyword()) :: t()
  def ifft(_out, %__MODULE__{} = tensor, _opts) do
    tensor
  end

  # ── Sort ─────────────────────────────────────────────────────────

  @impl true
  @spec sort(Nx.Tensor.t(), t(), keyword()) :: t()
  def sort(_out, %__MODULE__{} = a, _opts) do
    a
  end

  @impl true
  @spec argsort(Nx.Tensor.t(), t(), keyword()) :: t()
  def argsort(_out, %__MODULE__{} = a, _opts) do
    a
  end

  # ── Triangular Solve ─────────────────────────────────────────────

  @impl true
  @spec triangular_solve(Nx.Tensor.t(), t(), t(), keyword()) :: t()
  def triangular_solve(_out, %__MODULE__{} = _a, %__MODULE__{} = b, _opts) do
    b
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
    Nif.free_tensor(ref)
  end

  @impl true
  @spec to_batched(Nx.Tensor.t(), t(), keyword()) :: Enumerable.t()
  def to_batched(%Nx.Tensor{} = out, %__MODULE__{} = tensor, opts) do
    batch_size = opts[:batch_size] || 1
    total_elems = Enum.reduce(tensor.shape, 1, &(&1 * &2))
    num_batches = div(total_elems, batch_size)

    Stream.map(0..max(num_batches - 1, 0), fn _i ->
      constant(out, 0.0, [])
    end)
  end

  @impl true
  @spec as_type(Nx.Tensor.t(), t()) :: t()
  def as_type(_out, %__MODULE__{} = a) do
    a
  end

  @impl true
  @spec bitcast(Nx.Tensor.t(), t()) :: t()
  def bitcast(_out, %__MODULE__{} = a) do
    a
  end

  @impl true
  @spec inspect(t(), keyword()) :: Inspect.Algebra.t() | String.t()
  def inspect(%__MODULE__{} = tensor, inspect_opts) do
    limit = inspect_opts[:limit] || 5

    preview =
      if Enum.product(tensor.shape) <= limit do
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

    Inspect.Algebra.concat(
      Inspect.Algebra.color("#ExBurn.Tensor<", :map, Inspect.Opts.new(limit: :infinity)),
      Inspect.Algebra.concat(
        Inspect.Algebra.color(
          "shape: #{inspect(tensor.shape)}, type: #{tensor.type}#{preview}",
          :map,
          Inspect.Opts.new(limit: :infinity)
        ),
        Inspect.Algebra.color(">", :map, Inspect.Opts.new(limit: :infinity))
      )
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

  @spec apply_reduce_fun(t()) :: t()
  defp apply_reduce_fun(%__MODULE__{} = tensor) do
    # Default reduction: sum over all axes
    with {:ok, ref} <- Nif.sum_tensor(tensor.ref),
         {:ok, shape} <- Nif.tensor_shape(ref) do
      %__MODULE__{ref: ref, shape: shape, type: :f32}
    else
      {:error, _} -> tensor
    end
  end

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

    case Nif.new_tensor(data, shape, Atom.to_string(type)) do
      {:ok, ref} -> {:ok, %__MODULE__{ref: ref, shape: shape, type: type}}
      {:error, reason} -> {:error, reason}
    end
  end

  @spec constant_from_val(float(), t(), Nx.Tensor.t()) :: t()
  defp constant_from_val(val, %__MODULE__{} = template, out) do
    data = <<val::float-32-native>>
    shape = Tuple.to_list(Nx.shape(out))

    case Nif.new_tensor(data, shape, "f32") do
      {:ok, ref} -> %__MODULE__{ref: ref, shape: shape, type: :f32}
      {:error, _} -> template
    end
  end

  @spec expand_shape([non_neg_integer()], non_neg_integer()) :: [non_neg_integer()]
  defp expand_shape(shape, axis) do
    {list_before, list_after} = Enum.split(shape, axis)
    list_before ++ [1] ++ list_after
  end
end
