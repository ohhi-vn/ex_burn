defmodule ExBurn.Serving.Server do
  @moduledoc """
  Nx.Serving callback implementation for ExBurn.

  Handles batching and dispatching inference requests to the ExBurn backend.
  """
  @behaviour Nx.Serving

  alias ExBurn.Model

  @impl true
  def init(_template, {model, _partitions}, _opts) do
    {:ok, model}
  end

  @impl true
  def handle_batch(batch, _metadata, model) do
    inputs = Nx.Batch.pad(batch, 0)

    case Model.predict(model, inputs) do
      {:ok, output} ->
        {output, model}

      {:error, reason} ->
        raise ExBurn.Error,
          op: :serving_predict,
          reason: reason,
          details: %{batch_size: Nx.size(inputs)}
    end
  end
end
