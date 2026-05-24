defmodule ExBurn.Tensor do
  @moduledoc """
  Tensor conversion utilities between Nx and Burn formats.

  Handles marshaling of tensor data between Elixir's Nx tensor
  representation and the Rust/Burn tensor references used by the NIF.

  ## Type Mapping

  | Nx Type    | Burn Type |
  |------------|-----------|
  | `{:f, 32}` | `:f32`    |
  | `{:f, 64}` | `:f64`    |
  | `{:f, 16}` | `:f32`    |
  | `{:bf, 16}`| `:f32`    |
  | `{:s, 32}` | `:i32`    |
  | `{:s, 64}` | `:i64`    |
  | `{:s, 16}` | `:i32`    |
  | `{:s, 8}`  | `:i32`    |
  | `{:u, 8}`  | `:f32`    |
  """

  @typedoc "Burn element type tag"
  @type burn_type :: :f32 | :f64 | :i32 | :i64

  @typedoc "NIF tensor reference"
  @type t :: %__MODULE__{
          ref: reference(),
          shape: [non_neg_integer()],
          type: burn_type()
        }

  defstruct [:ref, :shape, :type]

  # ── Type Conversions ─────────────────────────────────────────────

  @doc "Converts an Nx type tuple to a Burn type tag."
  @spec nx_type_to_burn(Nx.Type.t()) :: burn_type()
  def nx_type_to_burn(type)
  def nx_type_to_burn({:f, 32}), do: :f32
  def nx_type_to_burn({:f, 64}), do: :f64
  def nx_type_to_burn({:f, 16}), do: :f32
  def nx_type_to_burn({:bf, 16}), do: :f32
  def nx_type_to_burn({:s, 32}), do: :i32
  def nx_type_to_burn({:s, 64}), do: :i64
  def nx_type_to_burn({:s, 16}), do: :i32
  def nx_type_to_burn({:s, 8}), do: :i32
  def nx_type_to_burn({:u, 8}), do: :f32
  def nx_type_to_burn(_), do: :f32

  @doc "Converts a Burn type tag back to an Nx type tuple."
  @spec burn_type_to_nx(burn_type()) :: Nx.Type.t()
  def burn_type_to_nx(type)
  def burn_type_to_nx(:f32), do: {:f, 32}
  def burn_type_to_nx(:f64), do: {:f, 64}
  def burn_type_to_nx(:i32), do: {:s, 32}
  def burn_type_to_nx(:i64), do: {:s, 64}
  def burn_type_to_nx(_), do: {:f, 32}

  # ── Nx ↔ Burn Conversion ─────────────────────────────────────────

  @doc """
  Converts an `Nx.Tensor.t()` into an `ExBurn.Tensor.t()`.

  The tensor data is sent to the Rust NIF layer as a flat binary.
  Returns `{:ok, tensor}` or `{:error, reason}`.
  """
  @spec from_nx(Nx.Tensor.t()) :: {:ok, t()} | {:error, String.t()}
  def from_nx(%Nx.Tensor{} = tensor) do
    data = Nx.to_binary(tensor)
    shape = Nx.shape(tensor) |> Tuple.to_list()
    type = nx_type_to_burn(Nx.type(tensor))

    try do
      ref = ExBurn.Nif.new_tensor(data, shape, Atom.to_string(type))
      {:ok, %__MODULE__{ref: ref, shape: shape, type: type}}
    rescue
      e -> {:error, Exception.message(e)}
    end
  end

  @doc """
  Converts an `ExBurn.Tensor.t()` back into an `Nx.Tensor.t()`.

  Reads the raw data from the Rust NIF layer and reshapes it.
  Returns `{:ok, tensor}` or `{:error, reason}`.
  """
  @spec to_nx(t()) :: {:ok, Nx.Tensor.t()} | {:error, String.t()}
  def to_nx(%__MODULE__{ref: ref, shape: shape, type: type}) do
    nx_type = burn_type_to_nx(type)

    try do
      binary = ExBurn.Nif.tensor_to_binary(ref)

      tensor =
        binary
        |> Nx.from_binary(nx_type)
        |> Nx.reshape(List.to_tuple(shape))

      {:ok, tensor}
    rescue
      e -> {:error, Exception.message(e)}
    end
  end

  @doc """
  Batch converts a list of Nx tensors to Burn tensors.

  More efficient than calling `from_nx/1` individually when you need
  to convert many tensors.
  """
  @spec from_nx_batch([Nx.Tensor.t()]) :: {:ok, [t()]} | {:error, String.t()}
  def from_nx_batch(tensors) do
    Enum.reduce_while(tensors, {:ok, []}, fn tensor, {:ok, acc} ->
      case from_nx(tensor) do
        {:ok, bt} -> {:cont, {:ok, [bt | acc]}}
        {:error, reason} -> {:halt, {:error, reason}}
      end
    end)
    |> case do
      {:ok, list} -> {:ok, Enum.reverse(list)}
      error -> error
    end
  end

  @doc """
  Batch converts a list of Burn tensors to Nx tensors.
  """
  @spec to_nx_batch([t()]) :: {:ok, [Nx.Tensor.t()]} | {:error, String.t()}
  def to_nx_batch(tensors) do
    Enum.reduce_while(tensors, {:ok, []}, fn bt, {:ok, acc} ->
      case to_nx(bt) do
        {:ok, nx} -> {:cont, {:ok, [nx | acc]}}
        {:error, reason} -> {:halt, {:error, reason}}
      end
    end)
    |> case do
      {:ok, list} -> {:ok, Enum.reverse(list)}
      error -> error
    end
  end

  # ── Direct Construction ──────────────────────────────────────────

  @doc "Creates a Burn tensor from raw binary data, shape, and type."
  @spec from_binary(binary(), [non_neg_integer()], burn_type()) ::
          {:ok, t()} | {:error, String.t()}
  def from_binary(data, shape, type) do
    try do
      ref = ExBurn.Nif.new_tensor(data, shape, Atom.to_string(type))
      {:ok, %__MODULE__{ref: ref, shape: shape, type: type}}
    rescue
      e -> {:error, Exception.message(e)}
    end
  end

  # ── Inspection ───────────────────────────────────────────────────

  @doc "Returns the shape of a Burn tensor."
  @spec shape(t()) :: [non_neg_integer()]
  def shape(%__MODULE__{shape: shape}), do: shape

  @doc "Returns the Burn element type of a tensor."
  @spec type(t()) :: burn_type()
  def type(%__MODULE__{type: type}), do: type

  @doc "Returns the NIF reference for a tensor."
  @spec ref(t()) :: reference()
  def ref(%__MODULE__{ref: ref}), do: ref

  @doc "Returns the total number of elements."
  @spec numel(t()) :: non_neg_integer()
  def numel(%__MODULE__{shape: shape}), do: Enum.product(shape)

  @doc "Returns the rank (number of dimensions)."
  @spec rank(t()) :: non_neg_integer()
  def rank(%__MODULE__{shape: shape}), do: length(shape)

  # ── Memory ───────────────────────────────────────────────────────

  @doc "Frees the underlying Rust tensor."
  @spec free(t()) :: :ok
  def free(%__MODULE__{ref: ref}), do: ExBurn.Nif.free_tensor(ref)
end
