defmodule ExBurn.Serving do
  @moduledoc """
  Nx.Serving integration for ExBurn.

  Provides batched, concurrent inference using `Nx.Serving` so that ExBurn
  can be used in Bumblebee-style production pipelines.

  ## Usage

      # Define a serving for a compiled model
      serving =
        ExBurn.Serving.new(model,
          batch_size: 32,
          batch_timeout: 50,
          partitions: System.schedulers_online()
        )

      # Run batched inference
      Nx.Serving.run(serving, input_tensor)

  ## Options

    * `:batch_size` — Maximum number of inputs to batch together (default: 32)
    * `:batch_timeout` — Max milliseconds to wait for a full batch (default: 50)
    * `:partitions` — Number of serving partitions (default: scheduler count)
    * `:padding` — Whether to pad batches to full size (default: false)
  """

  alias ExBurn.Model

  @type t :: %__MODULE__{
          model: Model.t(),
          batch_size: pos_integer(),
          batch_timeout: pos_integer(),
          partitions: pos_integer(),
          padding: boolean()
        }

  defstruct [:model, :batch_size, :batch_timeout, :partitions, :padding]

  @doc """
  Creates a new ExBurn serving for the given compiled model.

  Returns a struct that can be passed to `Nx.Serving` or used directly
  with `run/2`.
  """
  @spec new(Model.t(), keyword()) :: t()
  def new(%Model{} = model, opts \\ []) do
    %__MODULE__{
      model: model,
      batch_size: Keyword.get(opts, :batch_size, 32),
      batch_timeout: Keyword.get(opts, :batch_timeout, 50),
      partitions: Keyword.get(opts, :partitions, System.schedulers_online()),
      padding: Keyword.get(opts, :padding, false)
    }
  end

  @doc """
  Builds an `Nx.Serving` for the given model and options.

  This is the primary entry point for production use. The returned
  `Nx.Serving` can be used with `Nx.Serving.run/2` or supervised
  in your application tree.

  ## Examples

      serving =
        ExBurn.Serving.build(model,
          batch_size: 16,
          batch_timeout: 100
        )

      # Run inference
      output = Nx.Serving.run(serving, input)
  """
  @spec build(Model.t(), keyword()) :: Nx.Serving.t()
  def build(%Model{} = model, opts \\ []) do
    batch_size = Keyword.get(opts, :batch_size, 32)
    batch_timeout = Keyword.get(opts, :batch_timeout, 50)
    partitions = Keyword.get(opts, :partitions, System.schedulers_online())

    Nx.Serving.new(
      ExBurn.Serving.Server,
      {model, partitions},
      batch_size: batch_size,
      batch_timeout: batch_timeout
    )
  end

  @doc """
  Runs inference on a single input tensor using the serving.

  This is a convenience wrapper around `Nx.Serving.run/2`.
  """
  @spec run(t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def run(%__MODULE__{} = serving, input) do
    serving
    |> build()
    |> Nx.Serving.run(input)
  end
end
