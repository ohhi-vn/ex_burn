defmodule ExBurn.Error do
  @moduledoc """
  Structured error type for ExBurn operations.

  ## Fields

    * `:op` — the operation that failed (e.g., `:add`, `:matmul`, `:conv`)
    * `:reason` — a human-readable error message
    * `:details` — optional map with additional context (shapes, types, etc.)

  ## Examples

      raise ExBurn.Error, op: :matmul, reason: "shape mismatch",
        details:: %{lhs: [3, 4], rhs: [5, 6]}

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
end
