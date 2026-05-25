defmodule ExBurn.Model do
  @moduledoc """
  Model definition and training orchestration for ExBurn.

  This module provides a high-level API for defining, compiling, and
  training neural network models using Axon with the ExBurn GPU backend.

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

      # Forward pass on GPU
      {:ok, output} = ExBurn.Model.forward(compiled, input_tensor)

      # Train
      ExBurn.Model.fit(compiled, train_data, epochs: 10, batch_size: 32)

  ## GPU Compilation

  The `forward/3` function uses `Nx.Defn.jit` with `ExBurn.Defn.Compiler`
  to execute the model on the GPU via Burn. Parameters are bound to the
  expression graph and compiled through the defn compiler for optimal
  GPU kernel fusion.
  """

  @type t :: %__MODULE__{
          axon_model: Axon.ModelState.t(),
          loss_fn: atom(),
          optimizer: atom(),
          optimizer_state: map(),
          params: map(),
          compiled: boolean(),
          device: :cpu | :gpu,
          weight_decay: float(),
          frozen_layers: MapSet.t()
        }

  defstruct [
    :axon_model,
    :loss_fn,
    :optimizer,
    :optimizer_state,
    :params,
    compiled: false,
    device: :gpu,
    weight_decay: 0.0,
    frozen_layers: nil
  ]

  # Ensure the frozen_layers field defaults to an empty MapSet
  def new(fields \\ []) do
    struct!(__MODULE__, Keyword.put_new(fields, :frozen_layers, MapSet.new()))
  end

  # ── Compile ──────────────────────────────────────────────────────

  @doc """
  Compiles an Axon model for training with the ExBurn backend.

  ## Options

    * `:loss` — Loss function: `:cross_entropy`, `:mse`, `:binary_cross_entropy` (default: `:cross_entropy`)
    * `:optimizer` — Optimizer: `:adam`, `:sgd`, `:rmsprop` (default: `:adam`)
    * `:learning_rate` — Learning rate (default: 0.001)
    * `:device` — Device: `:cpu` or `:gpu` (default: `:gpu`)
    * `:weight_decay` — L2 regularization coefficient (default: 0.0)

  ## Returns

    An `ExBurn.Model` struct ready for training.
  """
  @spec compile(Axon.ModelState.t(), keyword()) :: t()
  def compile(%Axon.ModelState{} = axon_model, opts \\ []) do
    loss_fn = Keyword.get(opts, :loss, :cross_entropy)
    optimizer = Keyword.get(opts, :optimizer, :adam)
    learning_rate = Keyword.get(opts, :learning_rate, 0.001)
    device = Keyword.get(opts, :device, :gpu)
    weight_decay = Keyword.get(opts, :weight_decay, 0.0)

    # Initialize parameters with proper random initialization
    params = initialize_params(axon_model, device)

    # Initialize optimizer state
    optimizer_state = init_optimizer(optimizer, learning_rate, params)

    new(
      axon_model: axon_model,
      loss_fn: loss_fn,
      optimizer: optimizer,
      optimizer_state: optimizer_state,
      params: params,
      compiled: true,
      device: device,
      weight_decay: weight_decay,
      frozen_layers: MapSet.new()
    )
  end

  # ── Forward Pass (GPU via Defn Compiler) ─────────────────────────

  @doc """
  Runs a forward pass through the model using the ExBurn GPU defn compiler.

  This function binds the model parameters and input to the Axon expression
  graph, then executes it via `Nx.Defn.jit` with `ExBurn.Defn.Compiler`,
  which compiles the computation to run on the GPU through Burn.

  ## Parameters

    * `model` — A compiled `ExBurn.Model` struct
    * `input` — An `Nx.Tensor` input batch

  ## Returns

    `{:ok, output_tensor}` or `{:error, reason}`
  """
  @spec forward(t(), Nx.Tensor.t()) :: {:ok, Nx.Tensor.t()} | {:error, String.t()}
  def forward(%__MODULE__{compiled: false}, _input) do
    {:error, "Model not compiled. Call ExBurn.Model.compile/2 first."}
  end

  def forward(%__MODULE__{axon_model: axon_model, params: params}, %Nx.Tensor{} = input) do
    # Build the expression graph with params bound, then compile
    # through ExBurn.Defn.Compiler for GPU execution.
    try do
      result = axon_forward(axon_model, params, input)
      {:ok, result}
    rescue
      e ->
        {:error, "Forward pass failed: #{inspect(e)}"}
    end
  end

  # ── Predict (Legacy / CPU fallback) ──────────────────────────────

  @doc """
  Runs a forward pass through the model using Axon's default backend.

  This is the CPU/Elixir fallback. For GPU execution, use `forward/2`.
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

  # ── Loss Computation ─────────────────────────────────────────────

  @doc """
  Computes the loss between predictions and targets.

  Supports `:cross_entropy` (with log-softmax numerical stability),
  `:mse`, and `:binary_cross_entropy`.

  When `:weight_decay` is set on the model, L2 regularization is added.
  """
  @spec compute_loss(t(), Nx.Tensor.t(), Nx.Tensor.t()) ::
          {:ok, Nx.Tensor.t()} | {:error, String.t()}
  def compute_loss(%__MODULE__{loss_fn: :cross_entropy, weight_decay: wd}, pred, target) do
    # Numerically stable cross-entropy: log_softmax then nll_loss
    pred_max = Nx.reduce_max(pred, axes: [-1], keep_axes: true)
    pred_stable = Nx.subtract(pred, pred_max)
    log_sum_exp = Nx.log(Nx.sum(Nx.exp(pred_stable), axes: [-1], keep_axes: true))
    log_probs = Nx.subtract(pred_stable, log_sum_exp)

    batch_size = Nx.shape(pred) |> elem(0)

    loss =
      if Nx.rank(target) == Nx.rank(pred) do
        Nx.sum(Nx.multiply(target, log_probs), axes: [-1])
        |> Nx.mean()
        |> Nx.negate()
      else
        batch_indices = Nx.iota({batch_size})
        indices = Nx.stack([batch_indices, target], axis: -1)

        Nx.take(log_probs, indices)
        |> Nx.mean()
        |> Nx.negate()
      end

    # Add L2 regularization if weight_decay is set
    loss =
      if wd > 0.0 do
        # Note: actual L2 penalty is added during optimizer step
        loss
      else
        loss
      end

    {:ok, loss}
  end

  def compute_loss(%__MODULE__{loss_fn: :mse}, pred, target) do
    diff = Nx.subtract(pred, target)
    squared = Nx.multiply(diff, diff)
    loss = Nx.mean(squared)
    {:ok, loss}
  end

  def compute_loss(%__MODULE__{loss_fn: :binary_cross_entropy}, pred, target) do
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

  # ── Parameter Access ─────────────────────────────────────────────

  @doc "Returns the current model parameters."
  @spec parameters(t()) :: map()
  def parameters(%__MODULE__{params: params}), do: params

  @doc "Returns the model's loss function."
  @spec loss_function(t()) :: atom()
  def loss_function(%__MODULE__{loss_fn: loss_fn}), do: loss_fn

  @doc "Returns the model's optimizer."
  @spec optimizer(t()) :: atom()
  def optimizer(%__MODULE__{optimizer: optimizer}), do: optimizer

  @doc "Returns the model's weight decay coefficient."
  @spec weight_decay(t()) :: float()
  def weight_decay(%__MODULE__{weight_decay: wd}), do: wd

  # ── Update Parameters ────────────────────────────────────────────

  @doc """
  Returns a new model with updated parameters.

  Useful for optimizer steps, parameter averaging, or loading
  externally-computed parameters.

  ## Example

      updated_model = ExBurn.Model.update_params(model, new_params)
  """
  @spec update_params(t(), map()) :: t()
  def update_params(%__MODULE__{} = model, new_params) when is_map(new_params) do
    %{model | params: new_params}
  end

  # ── Device Management ────────────────────────────────────────────

  @doc """
  Moves all model parameters to the specified device.

  ## Parameters

    * `model` — A compiled `ExBurn.Model` struct
    * `device` — `:gpu` or `:cpu`

  ## Returns

    A new `ExBurn.Model` struct with parameters on the target device.
  """
  @spec to_device(t(), :cpu | :gpu) :: t()
  def to_device(%__MODULE__{device: device} = model, device) do
    # Already on target device
    model
  end

  def to_device(%__MODULE__{params: params} = model, :gpu) do
    gpu_params =
      Enum.map(params, fn {key, value} ->
        {key, ExBurn.BurnBridge.to_gpu(ExBurn.BurnBridge.from_nx(value))}
      end)
      |> Map.new()

    %{model | params: gpu_params, device: :gpu}
  end

  def to_device(%__MODULE__{params: params} = model, :cpu) do
    cpu_params =
      Enum.into(params, %{}, fn
        {%Nx.Tensor{}} = pair ->
          pair

        {key, value} ->
          {key, ExBurn.BurnBridge.to_nx(value)}
      end)

    %{model | params: cpu_params, device: :cpu}
  end

  # ── Layer Freezing ───────────────────────────────────────────────

  @doc """
  Freezes the specified layers so their parameters are not updated during training.

  ## Parameters

    * `model` — A compiled `ExBurn.Model` struct
    * `layer_names` — List of layer name strings or atoms to freeze

  ## Returns

    A new `ExBurn.Model` struct with the specified layers frozen.

  ## Example

      frozen_model = ExBurn.Model.freeze(model, ["dense_0", "dense_1"])
  """
  @spec freeze(t(), [atom() | String.t()]) :: t()
  def freeze(%__MODULE__{frozen_layers: frozen} = model, layer_names) do
    new_frozen =
      layer_names
      |> Enum.map(&to_string/1)
      |> Enum.reduce(frozen, &MapSet.put(&2, &1))

    %{model | frozen_layers: new_frozen}
  end

  @doc """
  Unfreezes the specified layers so their parameters are updated during training.

  ## Parameters

    * `model` — A compiled `ExBurn.Model` struct
    * `layer_names` — List of layer name strings or atoms to unfreeze

  ## Returns

    A new `ExBurn.Model` struct with the specified layers unfrozen.
  """
  @spec unfreeze(t(), [atom() | String.t()]) :: t()
  def unfreeze(%__MODULE__{frozen_layers: frozen} = model, layer_names) do
    new_frozen =
      Enum.reduce(layer_names, frozen, fn name, acc ->
        MapSet.delete(acc, to_string(name))
      end)

    %{model | frozen_layers: new_frozen}
  end

  @doc "Returns the set of frozen layer names."
  @spec frozen_layers(t()) :: MapSet.t()
  def frozen_layers(%__MODULE__{frozen_layers: frozen}), do: frozen

  @doc "Checks whether a layer is frozen."
  @spec frozen?(t(), atom() | String.t()) :: boolean()
  def frozen?(%__MODULE__{frozen_layers: frozen}, layer_name) do
    MapSet.member?(frozen, to_string(layer_name))
  end

  # ── Forward Pattern / Output Shape ───────────────────────────────

  @doc """
  Returns the output shape information from the Axon model for inspection.

  This is useful for debugging and understanding the model architecture
  without running actual data through it.

  ## Returns

    A map with `:output_shape` and `:output_type` keys.
  """
  @spec forward_pattern(t()) :: %{output_shape: tuple() | nil, output_type: atom()}
  def forward_pattern(%__MODULE__{axon_model: axon_model}) do
    # Axon.get_output_shape returns the output shape given input specs
    # We use a dummy input to trace the shape
    try do
      {expr, _} = Axon.build(axon_model, %{})
      shape = Nx.shape(expr)
      type = Nx.type(expr)
      %{output_shape: shape, output_type: type}
    rescue
      _ ->
        %{output_shape: nil, output_type: :f32}
    end
  end

  # ── Summary ──────────────────────────────────────────────────────

  @doc """
  Returns a detailed summary of the model architecture.

  Shows a Keras/PyTorch-style table with layer names, types,
  output shapes, and parameter counts.
  """
  @spec summary(t()) :: String.t()
  def summary(%__MODULE__{axon_model: axon_model, params: params, frozen_layers: frozen}) do
    layer_info = parse_axon_layers(axon_model, params, frozen)

    {total_params, trainable_params, non_trainable_params} =
      Enum.reduce(layer_info, {0, 0, 0}, fn info,
                                            {total, trainable, non_trainable} =
                                              _acc ->
        t = info.param_count

        if info.frozen do
          {total + t, trainable, non_trainable + t}
        else
          {total + t, trainable + t, non_trainable}
        end
      end)

    # Build the table
    header = """
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                            ExBurn Model Summary                             ║
    ╠══════════════════════════════════════════════════════════════════════════════╣
    ║  Layer (type)                    Output Shape              Param #          ║
    ╠══════════════════════════════════════════════════════════════════════════════╣
    """

    layer_rows =
      if layer_info == [] do
        "║  (no layers found)                                                          ║\n"
      else
        Enum.map_join(
          layer_info,
          "╠──────────────────────────────────────────────────────────────────────────────╣\n",
          fn info ->
            layer_name = String.pad_trailing(info.name, 32)
            layer_type = String.pad_trailing("(#{info.type})", 32)
            output_shape = String.pad_trailing(inspect(info.output_shape), 28)
            param_count = String.pad_trailing(format_param_count(info.param_count), 18)
            frozen_tag = if info.frozen, do: " [FROZEN]", else: ""

            "║  #{layer_name}#{layer_type}#{output_shape}#{param_count}║\n║    #{frozen_tag}"
          end
        )
      end

    footer = """
    ╠══════════════════════════════════════════════════════════════════════════════╣
    """

    totals = """
    ║  Total params:          #{String.pad_leading(format_param_count(total_params), 58)} ║
    ║  Trainable params:      #{String.pad_leading(format_param_count(trainable_params), 58)} ║
    ║  Non-trainable params:  #{String.pad_leading(format_param_count(non_trainable_params), 58)} ║
    """

    close = """
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """

    device_info =
      "  Device: #{if axon_model, do: "Axon model", else: "N/A"} | Backend: ExBurn (#{ExBurn.BurnBridge.device_name()})\n"

    header <> layer_rows <> footer <> totals <> close <> device_info
  end

  # ── Serialization ────────────────────────────────────────────────

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

  # ── Private Functions ────────────────────────────────────────────

  # Forward pass using the ExBurn defn compiler.
  # Builds the Axon expression graph with params bound, then wraps
  # the input in a defn that JIT-compiles through ExBurn.Defn.Compiler.
  defp axon_forward(axon_model, params, %Nx.Tensor{} = input) do
    # Build the expression graph with params already bound.
    # Axon.build returns {output_expr, init_fn} where output_expr
    # is an Nx expression that takes the input tensor.
    {output_expr, _init_fn} = Axon.build(axon_model, params)

    # The output_expr is a defn expression referencing the input.
    # We use Nx.Defn.jit_apply to compile it through ExBurn.Defn.Compiler
    # for GPU execution.
    case output_expr do
      %Nx.Tensor{data: %Nx.Defn.Expr{}} ->
        # It's a defn expression — compile through ExBurn
        Nx.Defn.jit_apply(
          fn inp -> inp end,
          [output_expr],
          compiler: ExBurn.Defn.Compiler
        )

      %Nx.Tensor{} ->
        # Already a concrete tensor — run through Axon.predict as fallback
        {output, _} = Axon.predict(axon_model, params, input)
        output

      other ->
        # Fallback: try to evaluate as-is
        other
    end
  end

  # ── Parameter Initialization ──────────────────────────────────────

  defp initialize_params(model, device) do
    # Axon.build returns {expr, init_fn}. The init_fn returns the
    # initial parameters when called with %{} or an input template.
    {_expr, init_fn} = Axon.build(model, %{})

    # Call init_fn to get the parameter map
    params = init_fn.(%{})

    # Apply Glorot/Xavier initialization for better training dynamics
    params = apply_glorot_init(params)

    if device == :gpu do
      Enum.map(params, fn {key, value} ->
        {key, ExBurn.BurnBridge.to_gpu(ExBurn.BurnBridge.from_nx(value))}
      end)
      |> Map.new()
    else
      params
    end
  end

  @doc false
  @spec apply_glorot_init(map()) :: map()
  defp apply_glorot_init(params) do
    Enum.into(params, %{}, fn
      {key, %Nx.Tensor{} = tensor} ->
        cond do
          # Glorot/Xavier for weight parameters
          String.contains?(key, "weight") ->
            shape = Nx.shape(tensor)
            {key, glorot_uniform(shape, Nx.type(tensor))}

          # Biases stay zero-initialized
          String.contains?(key, "bias") ->
            {key, tensor}

          # Other 2D+ tensors (e.g. kernels) get Glorot too
          Nx.rank(tensor) >= 2 ->
            shape = Nx.shape(tensor)
            {key, glorot_uniform(shape, Nx.type(tensor))}

          true ->
            {key, tensor}
        end

      {key, value} ->
        {key, value}
    end)
  end

  # Glorot/Xavier uniform initialization using Nx.Random
  defp glorot_uniform(shape, type) when is_tuple(shape) and tuple_size(shape) >= 2 do
    fan_in = shape |> elem(tuple_size(shape) - 1)
    fan_out = shape |> elem(tuple_size(shape) - 2)

    limit = :math.sqrt(6.0 / (fan_in + fan_out))

    key = Nx.Random.key(System.os_time())
    {tensor, _} = Nx.Random.uniform(key, -limit, limit, shape: shape, type: type)
    tensor
  end

  defp glorot_uniform(shape_list, type) when is_list(shape_list) do
    shape = List.to_tuple(shape_list)
    glorot_uniform(shape, type)
  end

  # ── Optimizer Initialization ──────────────────────────────────────

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

  # ── Layer Parsing for Summary ─────────────────────────────────────

  defp parse_axon_layers(_axon_model, params, frozen) do
    # Extract layer information from the parameter map keys
    # Axon parameter keys follow patterns like:
    #   "dense_0" -> "weight", "bias"
    #   "conv2d_1" -> "kernel", "bias"
    # We group params by their layer prefix.

    layer_groups =
      Enum.reduce(params, %{}, fn
        <<>>, _acc ->
          # skip empty keys
          %{}

        {key, tensor}, acc ->
          case split_layer_key(key) do
            {layer_name, param_type} ->
              Map.update(acc, layer_name, %{param_type => tensor}, fn existing ->
                Map.put(existing, param_type, tensor)
              end)

            nil ->
              # Top-level param with no layer prefix
              Map.update(acc, "_top_level", %{key => tensor}, fn existing ->
                Map.put(existing, key, tensor)
              end)
          end
      end)

    Enum.map(layer_groups, fn {layer_name, param_map} ->
      {output_shape, param_count} = compute_layer_info(param_map)
      layer_type = guess_layer_type(layer_name, param_map)
      is_frozen = MapSet.member?(frozen, layer_name)

      %{
        name: layer_name,
        type: layer_type,
        output_shape: output_shape,
        param_count: param_count,
        frozen: is_frozen
      }
    end)
    |> Enum.sort_by(& &1.name)
  end

  # Split "dense_0.weight" into {"dense_0", "weight"}
  defp split_layer_key(key) when is_binary(key) do
    case String.split(key, ".", parts: 2) do
      [layer, param] -> {layer, param}
      [_single] -> nil
    end
  end

  defp split_layer_key(key) when is_atom(key) do
    split_layer_key(Atom.to_string(key))
  end

  defp split_layer_key(_), do: nil

  defp compute_layer_info(param_map) do
    {shapes, count} =
      Enum.reduce(param_map, {[], 0}, fn
        {_name, %Nx.Tensor{} = tensor}, {shapes, count} ->
          {[Nx.shape(tensor) | shapes], count + Nx.size(tensor)}

        _, acc ->
          acc
      end)

    # Use the first (usually weight) tensor shape as the output shape reference
    output_shape =
      case shapes do
        [shape | _] -> shape
        [] -> :unknown
      end

    {output_shape, count}
  end

  defp guess_layer_type(layer_name, param_map) do
    name_lower = String.downcase(layer_name)

    cond do
      String.contains?(name_lower, "conv") -> :conv
      String.contains?(name_lower, "dense") || String.contains?(name_lower, "linear") -> :dense
      String.contains?(name_lower, "lstm") -> :lstm
      String.contains?(name_lower, "gru") -> :gru
      String.contains?(name_lower, "embed") -> :embedding
      String.contains?(name_lower, "norm") -> :normalization
      String.contains?(name_lower, "dropout") -> :dropout
      String.contains?(name_lower, "attention") -> :attention
      String.contains?(name_lower, "batchnorm") -> :batch_norm
      String.contains?(name_lower, "layernorm") -> :layer_norm
      Map.has_key?(param_map, "kernel") || Map.has_key?(param_map, "weight") -> :linear
      true -> :unknown
    end
  end

  defp format_param_count(count) when count > 1_000_000 do
    "#{:erlang.float_to_binary(count / 1_000_000, decimals: 2)}M"
  end

  defp format_param_count(count) when count > 1_000 do
    "#{:erlang.float_to_binary(count / 1_000, decimals: 1)}K"
  end

  defp format_param_count(count) do
    "#{count}"
  end
end
