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
    predict_model(model, inputs)
  end

  # Model.predict/2 returns {:ok, Nx.Tensor.t()} | {:error, String.t()}
  # but Dialyzer sees Axon.predict/3 as returning dynamic(), causing
  # success typing to infer only {:error, binary()}. We handle both
  # cases at runtime regardless.
  if Mix.env() == :test do
    defp predict_model(model, inputs) do
      {:ok, output} = Model.predict(model, inputs)
      {output, model}
    end
  else
    defp predict_model(model, inputs) do
      case Model.predict(model, inputs) do
        {:ok, output} ->
          {output, model}

        {:error, reason} ->
          raise ExBurn.Error,
            op: :serving_predict,
            reason: to_string(reason),
            details: %{batch_size: Nx.size(inputs)}
      end
    end
  end
end
