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

  @type model :: ExBurn.Model.t()

  @type t :: %__MODULE__{
          model: model(),
          batch_size: pos_integer(),
          batch_timeout: pos_integer(),
          partitions: pos_integer(),
          padding: boolean(),
          serving: Nx.Serving.t() | nil
        }

  defstruct [:model, :batch_size, :batch_timeout, :partitions, :padding, :serving]

  @doc """
  Creates a new ExBurn serving for the given compiled model.

  Returns a struct that can be passed to `Nx.Serving` or used directly
  with `run/2`.
  """
  @spec new(model(), keyword()) :: t()
  def new(%Model{} = model, opts \\ []) do
    serving =
      if Keyword.get(opts, :lazy?, false) do
        nil
      else
        build(model, opts)
      end

    %__MODULE__{
      model: model,
      serving: serving,
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
  @spec build(model(), keyword()) :: Nx.Serving.t()
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
  def run(%__MODULE__{serving: nil, model: model}, input) do
    input |> then(&Nx.Serving.run(build(model), &1))
  end

  def run(%__MODULE__{serving: serving}, input) when not is_nil(serving) do
    Nx.Serving.run(serving, input)
  end

  @doc """
  Returns the status of the serving as a map.

  ## Returns

      %{batch_size: pos_integer(), batch_timeout: pos_integer(),
        partitions: pos_integer(), padding: boolean()}
  """
  @spec status(t()) :: map()
  def status(%__MODULE__{} = serving) do
    %{
      batch_size: serving.batch_size,
      batch_timeout: serving.batch_timeout,
      partitions: serving.partitions,
      padding: serving.padding
    }
  end

  @doc "Returns a new serving with the specified batch size."
  @spec with_batch_size(t(), pos_integer()) :: t()
  def with_batch_size(%__MODULE__{serving: nil} = serving, batch_size) do
    %{serving | batch_size: batch_size}
  end

  def with_batch_size(%__MODULE__{serving: serving} = wrapper, batch_size)
      when not is_nil(serving) do
    %{wrapper | batch_size: batch_size, serving: build(wrapper.model, serving_opts(wrapper))}
  end

  @doc "Returns a new serving with the specified batch timeout."
  @spec with_timeout(t(), pos_integer()) :: t()
  def with_timeout(%__MODULE__{serving: nil} = serving, timeout) do
    %{serving | batch_timeout: timeout}
  end

  def with_timeout(%__MODULE__{serving: serving} = wrapper, timeout)
      when not is_nil(serving) do
    %{wrapper | batch_timeout: timeout, serving: build(wrapper.model, serving_opts(wrapper))}
  end

  defp serving_opts(%__MODULE__{} = wrapper) do
    [
      batch_size: wrapper.batch_size,
      batch_timeout: wrapper.batch_timeout,
      partitions: wrapper.partitions,
      padding: wrapper.padding
    ]
  end
end
