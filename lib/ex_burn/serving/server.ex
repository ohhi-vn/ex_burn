defmodule ExBurn.Serving.Server do
  @moduledoc """
  Nx.Serving callback implementation for ExBurn.

  Handles batching and dispatching inference requests to the ExBurn backend.
  """
  @behaviour Nx.Serving

  alias ExBurn.Model

  require Logger

  @impl true
  def init(_template, {model, _partitions}, _opts) do
    {:ok, model}
  end

  @impl true
  def handle_batch(batch, _metadata, model) do
    inputs = Nx.Batch.pad(batch, 0)

    case predict_model(model, inputs) do
      {:ok, output} ->
        {output, model}

      {:error, reason} ->
        Logger.error("Serving prediction failed: #{to_string(reason)}")

        raise ExBurn.Error,
          op: :serving_predict,
          reason: to_string(reason),
          details: %{batch_size: elem(Nx.shape(inputs), 0)}
    end
  end

  defp predict_model(model, inputs) do
    Model.predict(model, inputs)
  end
end
