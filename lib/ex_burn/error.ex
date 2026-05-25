defmodule ExBurn.Error do
  @moduledoc """
  Structured error type for ExBurn operations.

  ## Fields

    * `:op` — the operation that failed (e.g., `:add`, `:matmul`, `:conv`)
    * `:reason` — a human-readable error message
    * `:details` — optional map with additional context (shapes, types, etc.)

  ## Examples

      raise ExBurn.Error, op: :matmul, reason: "shape mismatch",
        details: %{lhs: [3, 4], rhs: [5, 6]}

  """

  defexception [:op, :reason, :details]

  @type t :: %__MODULE__{
          op: atom(),
          reason: String.t(),
          details: map() | nil
        }

  @impl true
  def message(%__MODULE__{op: op, reason: reason, details: nil}) do
    "ExBurn.#{op}: #{reason}"
  end

  @impl true
  def message(%__MODULE__{op: op, reason: reason, details: details}) do
    "ExBurn.#{op}: #{reason} (#{inspect(details)})"
  end

  @doc """
  Formats an error for logging or display.

  Returns a string with the operation, reason, and any details.
  """
  @spec format_error(t()) :: String.t()
  def format_error(%__MODULE__{} = error), do: message(error)

  @doc """
  Creates an error struct (non-raising).

  Useful for returning errors in pipelines without raising.

  ## Examples

      iex> ExBurn.Error.new(op: :add, reason: %{"bad input"})
      %ExBurn.Error{op: :add, reason: "bad input", details: nil}
  """
  @spec new(keyword()) :: t()
  def new(opts \\ []) do
    struct!(__MODULE__, opts)
  end

  @doc """
  Wraps an error tuple in an ExBurn.Error.

  ## Examples

      iex> ExBurn.Error.from_tuple({:error, "something failed"}, op: :forward)
      %ExBurn.Error{op: :forward, reason: "something failed", details: nil}
  """
  @spec from_tuple({:error, String.t()}, keyword()) :: t()
  def from_tuple({:error, reason}, opts) do
    new(Keyword.put(opts, :reason, to_string(reason)))
  end

  @doc """
  Converts the error to a string representation suitable for logging.
  """
  @spec to_log_string(t()) :: String.t()
  def to_log_string(%__MODULE__{op: op, reason: reason, details: nil}) do
    "[ExBurn:#{op}] #{reason}"
  end

  def to_log_string(%__MODULE__{op: op, reason: reason, details: details}) do
    "[ExBurn:#{op}] #{reason} | details: #{inspect(details, pretty: true, limit: 50)}"
  end
end
