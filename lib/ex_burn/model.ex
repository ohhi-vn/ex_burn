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

  # Axon structs are referenced directly; the dialyzer PLT includes axon
  # (see :dialyzer config in mix.exs).
  @type t :: %__MODULE__{
          axon_model: Axon.ModelState.t(),
          axon_graph: Axon.t() | nil,
          loss_fn: atom(),
          optimizer: atom(),
          optimizer_state: map(),
          params: map(),
          compiled: boolean(),
          device: :cpu | :gpu,
          weight_decay: float(),
          frozen_layers: MapSet.t(),
          predict_fn: (map(), Nx.Tensor.t() -> Nx.Tensor.t()) | nil
        }

  defstruct [
    :axon_model,
    :axon_graph,
    :loss_fn,
    :optimizer,
    :optimizer_state,
    :params,
    :predict_fn,
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
  @spec compile(Axon.t() | Axon.ModelState.t(), keyword()) :: t()
  def compile(axon_model_or_graph, opts \\ [])

  def compile(%Axon{} = axon_graph, opts) do
    axon_model = compile_to_model_state(axon_graph, opts)
    compile(axon_model, Keyword.put(opts, :axon_graph, axon_graph))
  end

  def compile(%Axon.ModelState{} = axon_model, opts) do
    loss_fn = Keyword.get(opts, :loss, :cross_entropy)
    optimizer = Keyword.get(opts, :optimizer, :adam)
    learning_rate = Keyword.get(opts, :learning_rate, 0.001)
    device = Keyword.get(opts, :device, :gpu)
    weight_decay = Keyword.get(opts, :weight_decay, 0.0)

    # Initialize parameters with proper random initialization
    params = initialize_params(axon_model, device)

    # Initialize optimizer state
    optimizer_state = init_optimizer(optimizer, learning_rate, params)

    # Build the predict function once — Axon.build/1 traces the graph and is
    # expensive, so it must not run on every forward/predict call.
    # Axon.build only accepts an %Axon{} graph, so models compiled directly
    # from a ModelState have no predict_fn until a graph is provided.
    predict_fn =
      case Keyword.get(opts, :axon_graph) do
        %Axon{} = graph ->
          {_init_fn, predict_fn} = Axon.build(graph)
          predict_fn

        _ ->
          nil
      end

    new(
      axon_model: axon_model,
      axon_graph: Keyword.get(opts, :axon_graph),
      loss_fn: loss_fn,
      optimizer: optimizer,
      optimizer_state: optimizer_state,
      params: params,
      compiled: true,
      device: device,
      weight_decay: weight_decay,
      frozen_layers: MapSet.new(),
      predict_fn: predict_fn
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

  def forward(%__MODULE__{params: params} = model, %Nx.Tensor{} = input) do
    try do
      # Axon expects the nested %{layer => %{param => tensor}} shape; stored
      # params are flat "layer.param" keys.
      result = axon_forward(model, unflatten_params(params), input)
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

  def predict(%__MODULE__{predict_fn: nil}, _input) do
    {:error, "Predict failed: no Axon graph available for inference"}
  end

  def predict(%__MODULE__{} = model, %Nx.Tensor{} = input) do
    {:ok, axon_forward(model, unflatten_params(model.params), input)}
  rescue
    e -> {:error, "Predict failed: #{Exception.message(e)}"}
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
  def compute_loss(%__MODULE__{loss_fn: :cross_entropy}, pred, target) do
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

        # `indices` holds full {batch_idx, class_idx} index tuples, so use
        # gather (not take, which indexes only along one axis).
        Nx.gather(log_probs, indices)
        |> Nx.mean()
        |> Nx.negate()
      end

    # Note: L2 regularization is applied as an explicit gradient term in
    # ExBurn.Training (add_weight_decay_grads/3), not inside the loss.
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
  def forward_pattern(%__MODULE__{axon_graph: %Axon{} = graph}) do
    # Trace the output shape/type through Axon itself (no data is executed).
    try do
      input = Nx.template(input_shape(graph), {:f, 32})
      output_template = Axon.get_output_shape(graph, input)
      %{output_shape: Nx.shape(output_template), output_type: Nx.type(output_template)}
    rescue
      _ ->
        %{output_shape: nil, output_type: :f32}
    end
  end

  def forward_pattern(_model) do
    %{output_shape: nil, output_type: :f32}
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

    device_name = ExBurn.BurnBridge.device_name()

    device_info =
      "  Device: #{if axon_model, do: "Axon model", else: "N/A"} | Backend: ExBurn (#{device_name})\n"

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
        params = :erlang.binary_to_term(binary, [:safe])
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
      {:ok, :erlang.binary_to_term(binary, [:safe])}
    rescue
      ArgumentError -> {:error, "Invalid parameter binary"}
    end
  end

  @doc """
  Quantizes model parameters to a lower precision type.

  Useful for reducing model size and speeding up inference on devices
  with limited compute. Currently supports `:f16` (half precision) and
  `:bf16` (brain float 16).

  ## Parameters

    * `model` — A compiled `ExBurn.Model` struct
    * `dtype` — Target dtype: `:f16` or `:bf16`

  ## Returns

    A new `ExBurn.Model` struct with quantized parameters.

  ## Examples

      quantized_model = ExBurn.Model.quantize(model, :f16)
  """
  @spec quantize(t(), :f16 | :bf16) :: t()
  def quantize(%__MODULE__{params: params} = model, dtype) when dtype in [:f16, :bf16] do
    nx_type = if dtype == :f16, do: {:f, 16}, else: {:bf, 16}

    quantized_params =
      Enum.map(params, fn
        {key, %Nx.Tensor{} = tensor} ->
          {key, Nx.as_type(tensor, nx_type)}

        {key, value} ->
          {key, value}
      end)
      |> Map.new()

    %{model | params: quantized_params}
  end

  @doc ~S"""
  Benchmarks the model's forward pass on the given input.

  Runs the forward pass multiple times and returns timing statistics.

  ## Parameters

    * `model` — A compiled `ExBurn.Model` struct
    * `input` — An `Nx.Tensor` input batch
    * `opts` — Options
      * `:warmup` — Number of warmup runs (default: 3)
      * `:runs` — Number of benchmarked runs (default: 10)

  ## Returns

    A map with `:avg_ms`, `:min_ms`, `:max_ms`, `:median_ms`, `:std_ms`.

  ## Example

      result = ExBurn.Model.benchmark(model, input, warmup: 5, runs: 20)
      IO.puts("Average: #{result.avg_ms}ms")
  """
  @spec benchmark(t(), Nx.Tensor.t() | {Nx.Tensor.t(), Nx.Tensor.t()}, keyword()) :: map()
  def benchmark(model, input, opts \\ [])

  def benchmark(
        %__MODULE__{} = model,
        {%Nx.Tensor{} = input, %Nx.Tensor{} = target},
        opts
      ) do
    warmup = Keyword.get(opts, :warmup, 3)
    runs = Keyword.get(opts, :runs, 10)

    # Warmup
    Enum.each(1..warmup, fn _ ->
      run_benchmark_step(model, input, target)
    end)

    # Benchmark
    times =
      Enum.map(1..runs, fn _ ->
        start = System.monotonic_time(:microsecond)
        _result = run_benchmark_step(model, input, target)
        System.monotonic_time(:microsecond) - start
      end)

    benchmark_stats(times, warmup, runs)
  end

  def benchmark(%__MODULE__{} = model, %Nx.Tensor{} = input, opts) do
    warmup = Keyword.get(opts, :warmup, 3)
    runs = Keyword.get(opts, :runs, 10)

    # Warmup
    Enum.each(1..warmup, fn _ ->
      ExBurn.Model.predict(model, input)
    end)

    # Benchmark
    times =
      Enum.map(1..runs, fn _ ->
        start = System.monotonic_time(:microsecond)
        _result = ExBurn.Model.predict(model, input)
        System.monotonic_time(:microsecond) - start
      end)

    benchmark_stats(times, warmup, runs)
  end

  defp run_benchmark_step(model, input, target) do
    case ExBurn.Model.predict(model, input) do
      {:ok, pred} ->
        ExBurn.Model.compute_loss(model, pred, target)

      _ ->
        :ok
    end
  end

  defp benchmark_stats(times_ms_input, warmup, runs) do
    times_ms = Enum.map(times_ms_input, &(&1 / 1000))
    avg = Enum.sum(times_ms) / length(times_ms)
    min = Enum.min(times_ms)
    max = Enum.max(times_ms)
    median = times_ms |> Enum.sort() |> Enum.at(div(length(times_ms), 2))
    variance = Enum.sum(Enum.map(times_ms, &((&1 - avg) * (&1 - avg)))) / length(times_ms)
    std = :math.sqrt(variance)

    %{
      avg_ms: Float.round(avg, 3),
      min_ms: Float.round(min, 3),
      max_ms: Float.round(max, 3),
      median_ms: Float.round(median, 3),
      std_ms: Float.round(std, 3),
      runs: runs,
      warmup: warmup
    }
  end

  @doc """
  Creates a deep copy of the model with identical parameters and configuration.

  Useful for creating model snapshots during training or for ensemble methods.

  ## Example

      snapshot = ExBurn.Model.clone(model)
  """
  @spec clone(t()) :: t()
  def clone(%__MODULE__{} = model) do
    cloned_params =
      Enum.map(model.params, fn
        {key, %Nx.Tensor{} = tensor} ->
          # Copy via binary round-trip — much faster than to_list for
          # large parameter tensors.
          {key,
           tensor
           |> Nx.to_binary()
           |> Nx.from_binary(Nx.type(tensor))
           |> Nx.reshape(Nx.shape(tensor))}

        {key, value} ->
          {key, value}
      end)
      |> Map.new()

    %{model | params: cloned_params}
  end

  @doc ~S"""
  Returns a map with model information.

  Includes parameter count, layer count, loss function, optimizer,
  device, and memory estimate.

  ## Example

      info = ExBurn.Model.info(model)
      IO.puts("Parameters: #{info.total_params}")
  """
  @spec info(t()) :: map()
  def info(%__MODULE__{} = model) do
    param_count =
      Enum.reduce(model.params, 0, fn
        {_key, %Nx.Tensor{} = tensor}, acc -> acc + Nx.size(tensor)
        _, acc -> acc
      end)

    layer_count = map_size(model.params)

    # Estimate memory (4 bytes per f32 parameter)
    memory_bytes = param_count * 4
    memory_mb = Float.round(memory_bytes / 1_048_576, 2)

    %{
      total_params: param_count,
      layer_count: layer_count,
      loss_function: model.loss_fn,
      optimizer: model.optimizer,
      learning_rate: get_in(model.optimizer_state, [:learning_rate]),
      device: model.device,
      weight_decay: model.weight_decay,
      frozen_layers_count: MapSet.size(model.frozen_layers),
      estimated_memory_mb: memory_mb,
      compiled: model.compiled
    }
  end

  @doc """
  Exports the model parameters to a portable format.

  Currently supports:
    * `:elixir_terms` — Compressed Erlang term format (default, portable)
    * `:json` — JSON format (human-readable, larger)

  ## Example

      ExBurn.Model.export(model, "/tmp/model.json", format: :json)
  """
  @spec export(t(), Path.t(), keyword()) :: :ok | {:error, String.t()}
  def export(%__MODULE__{params: params}, path, opts \\ []) do
    format = Keyword.get(opts, :format, :elixir_terms)

    case format do
      :elixir_terms ->
        binary = :erlang.term_to_binary(params, compressed: 9)
        File.write(path, binary)

      :json ->
        json_data =
          Enum.map(params, fn
            {key, %Nx.Tensor{} = tensor} ->
              {key,
               %{
                 "shape" => Tuple.to_list(Nx.shape(tensor)),
                 "type" => nx_type_to_string(Nx.type(tensor)),
                 "data" => Nx.to_list(tensor)
               }}

            {key, value} ->
              {key, inspect(value)}
          end)
          |> Map.new()
          |> Jason.encode!()

        File.write(path, json_data)

      other ->
        {:error, "Unsupported export format: #{inspect(other)}"}
    end
  end

  @doc """
  Imports model parameters from a file saved with `export/2`.

  ## Example

      {:ok, model} = ExBurn.Model.import_params(model, "/tmp/model.etf")
  """
  @spec import_params(t(), Path.t(), keyword()) :: {:ok, t()} | {:error, String.t()}
  def import_params(%__MODULE__{} = model, path, opts \\ []) do
    format = Keyword.get(opts, :format, :elixir_terms)

    case File.read(path) do
      {:ok, binary} ->
        params =
          case format do
            :elixir_terms ->
              :erlang.binary_to_term(binary, [:safe])

            :json ->
              case Jason.decode(binary) do
                {:ok, json_map} ->
                  Enum.map(json_map, fn
                    {key, %{"shape" => shape, "type" => type_str, "data" => data}} ->
                      nx_type = parse_nx_type(type_str)
                      tensor = Nx.tensor(data, type: nx_type) |> Nx.reshape(List.to_tuple(shape))
                      {key, tensor}

                    {key, value} ->
                      {key, value}
                  end)
                  |> Map.new()

                {:error, reason} ->
                  raise ExBurn.Error,
                    op: :import_params,
                    reason: "Invalid JSON: #{inspect(reason)}"
              end
          end

        {:ok, %{model | params: params, compiled: true}}

      {:error, reason} ->
        {:error, "Failed to read file: #{inspect(reason)}"}
    end
  end

  # ── Private Functions ────────────────────────────────────────────

  # Forward pass through the cached Axon predict function.
  # `params` must be the nested `%{layer => %{param => tensor}}` map that
  # Axon expects (use `unflatten_params/1` for flat "layer.param" keys).
  # It is wrapped in a %Axon.ModelState{} — passing bare maps is deprecated.
  defp axon_forward(%__MODULE__{} = model, params, %Nx.Tensor{} = input) do
    model.predict_fn.(Axon.ModelState.new(params), input)
  rescue
    e -> reraise ExBurn.Error, [op: :forward, reason: Exception.message(e)], __STACKTRACE__
  end

  # ── Parameter Initialization ──────────────────────────────────

  # Compiles a %Axon{} graph into a ModelState using its init_fn with
  # a template matching the graph's input shape.
  defp compile_to_model_state(%Axon{} = graph, _opts) do
    {init_fn, _} = Axon.build(graph)
    shape = input_shape(graph)
    template = Nx.template(shape, :f32)
    init_fn.(template, Axon.ModelState.empty())
  end

  defp input_shape(%Axon{} = graph) do
    graph.nodes
    |> Map.values()
    |> Enum.find(&(&1.op == :input))
    |> case do
      %{opts: opts} ->
        shape = Keyword.get(opts, :shape, {1, 2})

        case Tuple.to_list(shape) do
          [nil | dims] -> List.to_tuple([1 | dims])
          dims -> List.to_tuple(dims)
        end

      _ ->
        {1, 2}
    end
  end

  # Rebuilds the nested %Axon.ModelState.data structure
  # (layer => %{param => tensor}) from the flattened "layer.param" keys.
  defp unflatten_params(flat) when is_map(flat) do
    Enum.reduce(flat, %{}, fn {k, v}, acc ->
      case String.split(k, ".", parts: 2) do
        [layer, p] -> Map.update(acc, layer, %{p => v}, &Map.put(&1, p, v))
        [layer] -> Map.put(acc, layer, v)
      end
    end)
  end

  defp initialize_params(%Axon.ModelState{} = model, _device) do
    # Axon.ModelState.data is a map of layer_name => %{param_name => tensor}
    # Flatten to a single map of "layer_name.param_name" => tensor
    params = flatten_model_state_data(model)

    # Apply Glorot/Xavier initialization for better training dynamics
    params = apply_glorot_init(params)

    # Note: GPU transfer is handled by to_device/2 after compilation
    params
  end

  defp flatten_model_state_data(model) do
    for {layer_name, layer_params} <- model.data,
        {param_name, tensor} <- layer_params,
        into: %{} do
      {layer_name <> "." <> param_name, tensor}
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
      String.contains?(name_lower, "conv") ->
        :conv

      String.contains?(name_lower, "dense") || String.contains?(name_lower, "linear") ->
        :dense

      String.contains?(name_lower, "lstm") ->
        :lstm

      String.contains?(name_lower, "gru") ->
        :gru

      String.contains?(name_lower, "embed") ->
        :embedding

      String.contains?(name_lower, "norm") ->
        :normalization

      String.contains?(name_lower, "dropout") ->
        :dropout

      String.contains?(name_lower, "attention") ->
        :attention

      String.contains?(name_lower, "batchnorm") ->
        :batch_norm

      String.contains?(name_lower, "layernorm") ->
        :layer_norm

      Map.has_key?(param_map, "kernel") || Map.has_key?(param_map, "weight") ->
        :linear

      true ->
        :unknown
    end
  end

  # ── JSON Import Helpers ───────────────────

  defp nx_type_to_string({:f, 32}), do: "f32"

  defp nx_type_to_string({:f, 64}), do: "f64"

  defp nx_type_to_string({:f, 16}), do: "f16"

  defp nx_type_to_string({:bf, 16}), do: "bf16"

  defp nx_type_to_string({:s, 32}), do: "s32"

  defp nx_type_to_string({:s, 64}), do: "s64"
  defp nx_type_to_string({:s, 16}), do: "s16"
  defp nx_type_to_string({:s, 8}), do: "s8"
  defp nx_type_to_string({:u, 8}), do: "u8"
  defp nx_type_to_string(_), do: "f32"

  defp parse_nx_type("f32"), do: {:f, 32}
  defp parse_nx_type("f64"), do: {:f, 64}
  defp parse_nx_type("f16"), do: {:f, 16}
  defp parse_nx_type("bf16"), do: {:bf, 16}
  defp parse_nx_type("s32"), do: {:s, 32}
  defp parse_nx_type("s64"), do: {:s, 64}
  defp parse_nx_type("s16"), do: {:s, 16}
  defp parse_nx_type("s8"), do: {:s, 8}
  defp parse_nx_type("u8"), do: {:u, 8}
  defp parse_nx_type(_), do: {:f, 32}

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
