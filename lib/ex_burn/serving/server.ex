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
    # Nx.Serving >= 0.10 expects {:execute, fun, state}, where fun runs the
    # actual computation lazily on the serving partition.
    {:execute, fn -> run_batch(batch, model) end, model}
  end

  defp run_batch(batch, model) do
    inputs = batch_to_tensor(batch)
    batch_size = Nx.shape(inputs) |> elem(0)

    case predict_model(model, inputs) do
      {:ok, output} ->
        {output, model}

      {:error, reason} ->
        Logger.error("Serving prediction failed: #{to_string(reason)}")

        raise ExBurn.Error,
          op: :serving_predict,
          reason: to_string(reason),
          details: %{batch_size: batch_size}
    end
  end

  # Nx.Serving hands us an %Nx.Batch{} of individual inputs; Model.predict
  # works on a single tensor, so materialize the batch by running identity
  # through Nx.Defn (this concatenates entries along axis 0).
  defp batch_to_tensor(%Nx.Batch{} = batch) do
    Nx.Defn.jit_apply(&Function.identity/1, [batch])
  end

  defp predict_model(model, inputs) do
    Model.predict(model, inputs)
  end
end
