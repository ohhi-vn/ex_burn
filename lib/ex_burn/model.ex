defmodule ExBurn.Model do
  @moduledoc """
  Model definition and training orchestration for ExBurn.

  This module provides a high-level API for defining, compiling, and
  training neural network models using Axon with the ExBurn backend.

  ## Usage

      # Define a model
      model =
        Axon.input("input", shape: {nil, 784})
        |> Axon.dense(256)
        |> Axon.relu()
        |> Axon.dropout(rate: 0.2)
        |> Axon.dense(128)
        |> Axon.relu()
        |> Axon.dense(10)

      # Compile with ExBurn backend
      compiled = ExBurn.Model.compile(model, loss: :cross_entropy, optimizer: :adam)

      # Train
      ExBurn.Model.fit(compiled, train_data, epochs: 10, batch_size: 32)
  """

  @type t :: %__MODULE__{
          axon_model: Axon.ModelState.t(),
          loss_fn: atom(),
          optimizer: atom(),
          optimizer_state: map(),
          params: map(),
          compiled: boolean()
        }

  defstruct [
    :axon_model,
    :loss_fn,
    :optimizer,
    :optimizer_state,
    :params,
    compiled: false
  ]

  @doc """
  Compiles an Axon model for training with the ExBurn backend.

  ## Options

    * `:loss` — Loss function: `:cross_entropy`, `:mse`, `:binary_cross_entropy` (default: `:cross_entropy`)
    * `:optimizer` — Optimizer: `:adam`, `:sgd`, `:rmsprop` (default: `:adam`)
    * `:learning_rate` — Learning rate (default: 0.001)
    * `:device` — Device: `:cpu` or `:gpu` (default: `:gpu`)

  ## Returns

    An `ExBurn.Model` struct ready for training.
  """
  @spec compile(Axon.ModelState.t(), keyword()) :: t()
  def compile(%Axon.ModelState{} = axon_model, opts \\ []) do
    loss_fn = Keyword.get(opts, :loss, :cross_entropy)
    optimizer = Keyword.get(opts, :optimizer, :adam)
    learning_rate = Keyword.get(opts, :learning_rate, 0.001)
    device = Keyword.get(opts, :device, :gpu)

    # Initialize parameters — Axon.init returns a flat map of Nx tensors
    params = initialize_params(axon_model, device)

    # Initialize optimizer state
    optimizer_state = init_optimizer(optimizer, learning_rate, params)

    %__MODULE__{
      axon_model: axon_model,
      loss_fn: loss_fn,
      optimizer: optimizer,
      optimizer_state: optimizer_state,
      params: params,
      compiled: true
    }
  end

  @doc """
  Runs a forward pass through the model.

  Returns the output tensor as an Nx tensor.
  """
  @spec predict(t(), Nx.Tensor.t()) :: {:ok, Nx.Tensor.t()} | {:error, String.t()}
  def predict(%__MODULE__{compiled: false}, _input) do
    {:error, "Model not compiled. Call ExBurn.Model.compile/2 first."}
  end

  def predict(%__MODULE__{axon_model: model, params: params}, %Nx.Tensor{} = input) do
    case Axon.predict(model, params, input) do
      {:ok, output} -> {:ok, output}
      {:error, reason} -> {:error, reason}
    end
  end

  @doc """
  Computes the loss between predictions and targets.

  Supports `:cross_entropy` (with log-softmax numerical stability) and `:mse`.
  """
  @spec compute_loss(t(), Nx.Tensor.t(), Nx.Tensor.t()) ::
          {:ok, Nx.Tensor.t()} | {:error, String.t()}
  def compute_loss(%__MODULE__{loss_fn: :cross_entropy}, pred, target) do
    # Numerically stable cross-entropy: log_softmax then nll_loss
    # log_softmax(x)_i = x_i - log(sum(exp(x_j)))
    # We subtract max for numerical stability before exp
    pred_max = Nx.reduce_max(pred, axes: [-1], keep_axes: true)
    pred_stable = Nx.subtract(pred, pred_max)
    log_sum_exp = Nx.log(Nx.sum(Nx.exp(pred_stable), axes: [-1], keep_axes: true))
    log_probs = Nx.subtract(pred_stable, log_sum_exp)

    # Negative log-likelihood: -sum(target * log_probs) / batch_size
    batch_size = Nx.shape(pred) |> elem(0)

    # If target is one-hot encoded, element-wise multiply and sum
    # If target is class indices, gather the log_probs
    loss =
      if Nx.rank(target) == Nx.rank(pred) do
        # One-hot targets
        Nx.sum(Nx.multiply(target, log_probs), axes: [-1])
        |> Nx.mean()
        |> Nx.negate()
      else
        # Integer class indices — gather
        batch_indices = Nx.iota({batch_size})
        indices = Nx.stack([batch_indices, target], axis: -1)

        Nx.take(log_probs, indices)
        |> Nx.mean()
        |> Nx.negate()
      end

    {:ok, loss}
  end

  def compute_loss(%__MODULE__{loss_fn: :mse}, pred, target) do
    # MSE = mean((pred - target)^2)
    diff = Nx.subtract(pred, target)
    squared = Nx.multiply(diff, diff)
    loss = Nx.mean(squared)
    {:ok, loss}
  end

  def compute_loss(%__MODULE__{loss_fn: :binary_cross_entropy}, pred, target) do
    # Numerically stable BCE: -[t*log(p) + (1-t)*log(1-p)]
    # Clamp predictions to avoid log(0)
    eps = 1.0e-7
    pred_clamped = Nx.clip(pred, eps, 1.0 - eps)

    loss =
      Nx.add(
        Nx.multiply(target, Nx.log(pred_clamped)),
        Nx.multiply(Nx.subtract(1, target), Nx.log(Nx.subtract(1, pred_clamped)))
      )
      |> Nx.mean()
      |> Nx.negate()

    {:ok, loss}
  end

  def compute_loss(%__MODULE__{loss_fn: loss_fn}, _pred, _target) do
    {:error, "Unsupported loss function: #{inspect(loss_fn)}"}
  end

  @doc """
  Returns the current model parameters.
  """
  @spec parameters(t()) :: map()
  def parameters(%__MODULE__{params: params}), do: params

  @doc """
  Returns the model's loss function.
  """
  @spec loss_function(t()) :: atom()
  def loss_function(%__MODULE__{loss_fn: loss_fn}), do: loss_fn

  @doc """
  Returns the model's optimizer.
  """
  @spec optimizer(t()) :: atom()
  def optimizer(%__MODULE__{optimizer: optimizer}), do: optimizer

  @doc """
  Saves the model parameters to a file using compressed Erlang term format.
  """
  @spec save(t(), Path.t()) :: :ok | {:error, String.t()}
  def save(%__MODULE__{params: params}, path) do
    binary = :erlang.term_to_binary(params, compressed: 9)
    File.write(path, binary)
  end

  @doc """
  Loads model parameters from a file.
  """
  @spec load(t(), Path.t()) :: {:ok, t()} | {:error, String.t()}
  def load(%__MODULE__{} = model, path) do
    case File.read(path) do
      {:ok, binary} ->
        params = :erlang.binary_to_term(binary)
        {:ok, %{model | params: params, compiled: true}}

      {:error, reason} ->
        {:error, "Failed to load model: #{inspect(reason)}"}
    end
  end

  @doc """
  Serializes parameters to a binary for network transfer or storage.
  Uses compressed Erlang term format.
  """
  @spec serialize_params(t()) :: binary()
  def serialize_params(%__MODULE__{params: params}) do
    :erlang.term_to_binary(params, compressed: 9)
  end

  @doc """
  Deserializes parameters from a binary.
  """
  @spec deserialize_params(binary()) :: {:ok, map()} | {:error, String.t()}
  def deserialize_params(binary) when is_binary(binary) do
    try do
      {:ok, :erlang.binary_to_term(binary)}
    rescue
      ArgumentError -> {:error, "Invalid parameter binary"}
    end
  end

  @doc """
  Returns a summary of the model architecture including parameter count.
  """
  @spec summary(t()) :: String.t()
  def summary(%__MODULE__{axon_model: _model, params: params}) do
    # Count total parameters
    {total_params, trainable_params} =
      Enum.reduce(params, {0, 0}, fn {_key, tensor}, {total, trainable} ->
        count = Nx.size(tensor)
        {total + count, trainable + count}
      end)

    # Format parameter count
    param_str =
      if total_params > 1_000_000 do
        "#{:erlang.float_to_binary(total_params / 1_000_000, decimals: 2)}M"
      else
        "#{:erlang.float_to_binary(total_params / 1_000, decimals: 1)}K"
      end

    # Build summary
    header = """
    ╔══════════════════════════════════════════════════════════╗
    ║                   ExBurn Model Summary                  ║
    ╠══════════════════════════════════════════════════════════╣
    """

    param_info = """
    ║  Total params:     #{String.pad_leading("#{total_params}", 36)} ║
    ║  Trainable params: #{String.pad_leading("#{trainable_params}", 36)} ║
    ║  Non-trainable:    #{String.pad_leading("0", 36)} ║
    ║  Formatted:        #{String.pad_leading(param_str, 36)} ║
    """

    footer = """
    ╠══════════════════════════════════════════════════════════╣
    """

    architecture = "(Axon model)"

    close = """
    ╚══════════════════════════════════════════════════════════╝
    """

    header <> param_info <> footer <> "\n" <> architecture <> "\n" <> close
  end

  # ── Private Functions ────────────────────────────────────────────

  defp initialize_params(model, device) do
    # Axon.init/2 returns a flat map of {layer_name => Nx.Tensor}
    # No pattern-matching on internal Axon structs
    params = Axon.build(model, %{})

    if device == :gpu do
      Enum.map(params, fn {key, value} ->
        {key, ExBurn.BurnBridge.to_gpu(ExBurn.BurnBridge.from_nx(value))}
      end)
      |> Map.new()
    else
      params
    end
  end

  defp init_optimizer(:adam, lr, params) do
    %{
      type: :adam,
      learning_rate: lr,
      beta1: 0.9,
      beta2: 0.999,
      epsilon: 1.0e-8,
      m: initialize_zeros_like(params),
      v: initialize_zeros_like(params),
      t: 0
    }
  end

  defp init_optimizer(:sgd, lr, params) do
    %{
      type: :sgd,
      learning_rate: lr,
      momentum: 0.9,
      nesterov: false,
      velocity: initialize_zeros_like(params)
    }
  end

  defp init_optimizer(:rmsprop, lr, params) do
    %{
      type: :rmsprop,
      learning_rate: lr,
      decay: 0.9,
      epsilon: 1.0e-8,
      cache: initialize_zeros_like(params)
    }
  end

  defp initialize_zeros_like(params) when is_map(params) do
    Enum.map(params, fn
      {key, %Nx.Tensor{} = value} ->
        {key, Nx.broadcast(Nx.tensor(0.0, type: Nx.type(value)), Nx.shape(value))}

      {key, _} ->
        {key, Nx.tensor(0.0)}
    end)
    |> Map.new()
  end
end
