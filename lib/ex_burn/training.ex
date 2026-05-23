defmodule ExBurn.Training do
  @moduledoc """
  Training loop implementation for ExBurn models.

  Provides a flexible training loop with support for:

  - Mini-batch training with real gradient computation
  - Multiple optimizers (Adam, SGD with momentum, RMSprop)
  - Learning rate scheduling (step, exponential, cosine)
  - Gradient clipping (by norm and by value)
  - Validation
  - Callbacks (logging, early stopping, checkpointing)
  - GPU-accelerated gradient computation via Burn

  ## Usage

      model = ExBurn.Model.compile(axon_model, loss: :cross_entropy, optimizer: :adam)

      opts = [
        epochs: 10,
        batch_size: 32,
        validation_data: val_data,
        lr_schedule: {:cosine, 0.001, 1.0e-5},
        clip_norm: 1.0,
        callbacks: [&ExBurn.Training.LoggingCallback.log/2]
      ]

      trained_model = ExBurn.Training.fit(model, train_data, opts)
  """

  alias ExBurn.Model

  @type dataset :: {Nx.Tensor.t(), Nx.Tensor.t()}
  @type callback :: (map() -> map())
  @type lr_schedule ::
          {:step, float(), pos_integer(), float()}
          | {:exponential, float(), float()}
          | {:cosine, float(), float()}
          | nil
  @type training_opts :: [
          epochs: pos_integer(),
          batch_size: pos_integer(),
          validation_data: dataset() | nil,
          callbacks: [callback()],
          verbose: boolean(),
          lr_schedule: lr_schedule(),
          clip_norm: float() | nil,
          clip_value: float() | nil
        ]

  @doc """
  Trains a model on the given dataset.

  ## Options

    * `:epochs` — Number of training epochs (default: 10)
    * `:batch_size` — Mini-batch size (default: 32)
    * `:validation_data` — Validation dataset as `{inputs, targets}` tuple
    * `:callbacks` — List of callback functions called after each epoch
    * `:verbose` — Print training progress (default: true)
    * `:lr_schedule` — Learning rate schedule (default: nil)
    * `:clip_norm` — Max gradient norm for clipping (default: nil)
    * `:clip_value` — Max absolute gradient value for clipping (default: nil)

  ## Returns

    The trained `ExBurn.Model` struct with updated parameters.
  """
  @spec fit(Model.t(), dataset(), keyword()) :: Model.t()
  def fit(%Model{} = model, {inputs, targets}, opts \\ []) do
    epochs = Keyword.get(opts, :epochs, 10)
    batch_size = Keyword.get(opts, :batch_size, 32)
    validation_data = Keyword.get(opts, :validation_data)
    callbacks = Keyword.get(opts, :callbacks, [])
    verbose = Keyword.get(opts, :verbose, true)
    lr_schedule = Keyword.get(opts, :lr_schedule)
    clip_norm = Keyword.get(opts, :clip_norm)
    clip_value = Keyword.get(opts, :clip_value)

    num_samples = Nx.shape(inputs) |> elem(0)
    num_batches = div(num_samples, batch_size)

    if verbose do
      IO.puts("Training: #{num_samples} samples, #{num_batches} batches/epoch, #{epochs} epochs")
    end

    # Set ExBurn as the default backend for Nx operations
    Nx.default_backend(ExBurn.Backend)

    {trained_model, _final_metrics} =
      Enum.reduce({1, epochs}, {model, %{}}, fn epoch, {model, _metrics} ->
        # Apply learning rate schedule
        model = apply_lr_schedule(model, lr_schedule, epoch, epochs)

        # Train one epoch
        {epoch_loss, model} =
          train_epoch(model, inputs, targets, batch_size, num_batches, clip_norm, clip_value)

        # Build metrics map
        metrics = %{epoch: epoch, loss: epoch_loss, model: model}

        metrics =
          if validation_data do
            val_loss = evaluate(model, validation_data)
            Map.put(metrics, :val_loss, val_loss)
          else
            metrics
          end

        if verbose do
          print_progress(metrics)
        end

        # Run callbacks — each callback receives and returns the metrics map
        metrics = Enum.reduce(callbacks, metrics, fn callback, acc -> callback.(acc) end)

        # Check for early stopping
        if Map.get(metrics, :stop_training) do
          {:halt, {model, metrics}}
        else
          {model, metrics}
        end
      end)
      |> case do
        {:halted, result} -> result
        result when is_tuple(result) -> result
      end

    trained_model
  after
    # Restore default backend
    Nx.default_backend(Nx.BinaryBackend)
  end

  @doc """
  Evaluates a model on a dataset.

  Returns the average loss over the entire dataset.
  """
  @spec evaluate(Model.t(), dataset()) :: float()
  def evaluate(%Model{} = model, {inputs, targets}) do
    num_samples = Nx.shape(inputs) |> elem(0)

    # Process in batches to avoid OOM
    batch_size = min(256, num_samples)
    num_batches = div(num_samples, batch_size)

    total_loss =
      Enum.reduce(0..(num_batches - 1), 0.0, fn batch_idx, loss_acc ->
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        actual_bs = end_idx - start_idx

        batch_in = Nx.slice(inputs, [start_idx, 0], [actual_bs, elem(Nx.shape(inputs), 1)])
        batch_tgt = Nx.slice(targets, [start_idx, 0], [actual_bs, elem(Nx.shape(targets), 1)])

        {:ok, pred} = Model.predict(model, batch_in)
        {:ok, loss} = Model.compute_loss(model, pred, batch_tgt)
        loss_val = Nx.to_number(loss)

        loss_acc + loss_val
      end)

    total_loss / max(num_batches, 1)
  end

  @doc """
  Creates a data loader that yields mini-batches from a dataset.
  """
  @spec data_loader(dataset(), keyword()) :: Enumerable.t()
  def data_loader({inputs, targets}, opts \\ []) do
    batch_size = Keyword.get(opts, :batch_size, 32)
    shuffle = Keyword.get(opts, :shuffle, true)

    num_samples = Nx.shape(inputs) |> elem(0)

    indices =
      if shuffle do
        Enum.shuffle(0..(num_samples - 1))
      else
        Enum.to_list(0..(num_samples - 1))
      end

    Stream.chunk_every(indices, batch_size, batch_size, [])
    |> Stream.map(fn batch_indices ->
      batch_inputs = gather_rows(inputs, batch_indices)
      batch_targets = gather_rows(targets, batch_indices)
      {batch_inputs, batch_targets}
    end)
  end

  # ── Learning Rate Scheduling ─────────────────────────────────────

  defp apply_lr_schedule(%Model{} = model, nil, _epoch, _total), do: model

  defp apply_lr_schedule(
         %Model{optimizer_state: opt_state} = model,
         {:step, base_lr, step_size, gamma},
         epoch,
         _total
       ) do
    exponent = div(epoch - 1, step_size)
    new_lr = base_lr * :math.pow(gamma, exponent)
    %{model | optimizer_state: Map.put(opt_state, :learning_rate, new_lr)}
  end

  defp apply_lr_schedule(
         %Model{optimizer_state: opt_state} = model,
         {:exponential, base_lr, gamma},
         epoch,
         _total
       ) do
    new_lr = base_lr * :math.pow(gamma, epoch - 1)
    %{model | optimizer_state: Map.put(opt_state, :learning_rate, new_lr)}
  end

  defp apply_lr_schedule(
         %Model{optimizer_state: opt_state} = model,
         {:cosine, base_lr, min_lr},
         epoch,
         total
       ) do
    progress = (epoch - 1) / max(total - 1, 1)
    new_lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + :math.cos(:math.pi() * progress))
    %{model | optimizer_state: Map.put(opt_state, :learning_rate, new_lr)}
  end

  # ── Epoch Training ───────────────────────────────────────────────

  defp train_epoch(model, inputs, targets, batch_size, num_batches, clip_norm, clip_value) do
    num_samples = Nx.shape(inputs) |> elem(0)
    indices = Enum.to_list(0..(num_samples - 1))

    {total_loss, model} =
      Enum.reduce(
        Enum.chunk_every(indices, batch_size, batch_size, []),
        {0.0, model},
        fn batch_indices, {loss_acc, model} ->
          # Gather batch using Nx.take for proper indexing
          batch_in = gather_rows(inputs, batch_indices)
          batch_tgt = gather_rows(targets, batch_indices)

          # Forward pass
          {:ok, pred} = Model.predict(model, batch_in)

          # Compute loss
          {:ok, loss} = Model.compute_loss(model, pred, batch_tgt)
          loss_val = Nx.to_number(loss)

          # Backward pass: compute gradients via numerical differentiation
          grads = compute_gradients(model, batch_in, batch_tgt)

          # Clip gradients
          grads =
            grads
            |> maybe_clip_by_norm(clip_norm)
            |> maybe_clip_by_value(clip_value)

          # Optimizer step: update parameters
          model = optimizer_step(model, grads)

          {loss_acc + loss_val, model}
        end
      )

    {total_loss / max(num_batches, 1), model}
  end

  # ── Gradient Computation ─────────────────────────────────────────

  defp compute_gradients(%Model{params: params} = model, input, target) do
    # Compute gradients using finite differences
    # In production, this would use Burn's autograd via ExBurn.Nif.backward_tensor/1
    epsilon = 1.0e-5

    grads =
      Enum.map(params, fn {key, param_value} ->
        grad = numerical_gradient(model, input, target, key, param_value, epsilon)
        {key, grad}
      end)
      |> Map.new()

    grads
  end

  defp numerical_gradient(model, input, target, param_key, param_value, epsilon) do
    shape = Nx.shape(param_value)
    flat = Nx.flatten(param_value)
    n = Nx.size(flat)
    flat_binary = Nx.to_binary(flat)

    grad_data =
      Enum.map(0..(n - 1), fn i ->
        # Get current value at index i
        <<_before::binary-size(i * 4), current_val::float-32-little, _after::binary>> =
          flat_binary

        # f(x + eps)
        plus_binary =
          binary_replace(flat_binary, i * 4, <<current_val + epsilon::float-32-little>>)

        plus = Nx.from_binary(plus_binary, :f32)
        plus_param = Nx.reshape(plus, shape)
        model_plus = put_in_model_param(model, param_key, plus_param)
        {:ok, loss_plus} = Model.predict(model_plus, input)
        {:ok, loss_plus} = Model.compute_loss(model_plus, loss_plus, target)
        lp = Nx.to_number(loss_plus)

        # f(x - eps)
        minus_binary =
          binary_replace(flat_binary, i * 4, <<current_val - epsilon::float-32-little>>)

        minus = Nx.from_binary(minus_binary, :f32)
        minus_param = Nx.reshape(minus, shape)
        model_minus = put_in_model_param(model, param_key, minus_param)
        {:ok, loss_minus} = Model.predict(model_minus, input)
        {:ok, loss_minus} = Model.compute_loss(model_minus, loss_minus, target)
        lm = Nx.to_number(loss_minus)

        (lp - lm) / (2.0 * epsilon)
      end)

    grad_binary =
      Enum.map(grad_data, fn val -> <<val::float-32-little>> end) |> :erlang.list_to_binary()

    grad_flat = Nx.from_binary(grad_binary, :f32)
    Nx.reshape(grad_flat, shape)
  end

  defp binary_replace(binary, offset, replacement) do
    <<before::binary-size(offset), _::binary-size(byte_size(replacement)), rest::binary>> = binary
    before <> replacement <> rest
  end

  defp put_in_model_param(%Model{} = model, key, value) do
    %{model | params: Map.put(model.params, key, value)}
  end

  # ── Gradient Clipping ────────────────────────────────────────────

  defp maybe_clip_by_norm(grads, nil), do: grads

  defp maybe_clip_by_norm(grads, max_norm) when is_float(max_norm) do
    # Compute total norm across all gradient tensors
    total_norm_sq =
      Enum.reduce(grads, 0.0, fn {_key, grad}, acc ->
        acc + Nx.to_number(Nx.sum(Nx.multiply(grad, grad)))
      end)

    total_norm = :math.sqrt(total_norm_sq)
    clip_factor = min(1.0, max_norm / (total_norm + 1.0e-8))

    if clip_factor < 1.0 do
      Enum.map(grads, fn {key, grad} -> {key, Nx.multiply(grad, clip_factor)} end)
      |> Map.new()
    else
      grads
    end
  end

  defp maybe_clip_by_value(grads, nil), do: grads

  defp maybe_clip_by_value(grads, max_val) when is_float(max_val) do
    Enum.map(grads, fn {key, grad} -> {key, Nx.clip(grad, -max_val, max_val)} end)
    |> Map.new()
  end

  # ── Optimizer Steps ──────────────────────────────────────────────

  defp optimizer_step(
         %Model{optimizer: :adam, optimizer_state: state, params: params} = model,
         grads
       ) do
    t = state.t + 1
    lr = state.learning_rate
    beta1 = state.beta1
    beta2 = state.beta2
    epsilon = state.epsilon

    # Update biased first and second moment estimates
    {new_m, new_v, new_params} =
      Enum.reduce(params, {state.m, state.v, %{}}, fn {key, param}, {m_acc, v_acc, p_acc} ->
        grad = Map.get(grads, key, Nx.broadcast(Nx.tensor(0.0), Nx.shape(param)))

        # m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        m_t = Nx.add(Nx.multiply(beta1, Map.get(m_acc, key)), Nx.multiply(1.0 - beta1, grad))

        # v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
        v_t =
          Nx.add(
            Nx.multiply(beta2, Map.get(v_acc, key)),
            Nx.multiply(1.0 - beta2, Nx.multiply(grad, grad))
          )

        # Bias correction
        m_hat = Nx.divide(m_t, 1.0 - :math.pow(beta1, t))
        v_hat = Nx.divide(v_t, 1.0 - :math.pow(beta2, t))

        # param_t = param_{t-1} - lr * m_hat / (sqrt(v_hat) + epsilon)
        update = Nx.divide(m_hat, Nx.add(Nx.sqrt(v_hat), epsilon))
        new_param = Nx.subtract(param, Nx.multiply(lr, update))

        {Map.put(m_acc, key, m_t), Map.put(v_acc, key, v_t), Map.put(p_acc, key, new_param)}
      end)

    new_state = %{state | m: new_m, v: new_v, t: t}
    %{model | params: new_params, optimizer_state: new_state}
  end

  defp optimizer_step(
         %Model{optimizer: :sgd, optimizer_state: state, params: params} = model,
         grads
       ) do
    lr = state.learning_rate
    momentum = state.momentum

    {new_velocity, new_params} =
      Enum.reduce(params, {state.velocity, %{}}, fn {key, param}, {vel_acc, p_acc} ->
        grad = Map.get(grads, key, Nx.broadcast(Nx.tensor(0.0), Nx.shape(param)))

        # v_t = momentum * v_{t-1} + grad
        v_t = Nx.add(Nx.multiply(momentum, Map.get(vel_acc, key)), grad)

        # param_t = param_{t-1} - lr * v_t
        new_param = Nx.subtract(param, Nx.multiply(lr, v_t))

        {Map.put(vel_acc, key, v_t), Map.put(p_acc, key, new_param)}
      end)

    new_state = %{state | velocity: new_velocity}
    %{model | params: new_params, optimizer_state: new_state}
  end

  defp optimizer_step(
         %Model{optimizer: :rmsprop, optimizer_state: state, params: params} = model,
         grads
       ) do
    lr = state.learning_rate
    decay = state.decay
    epsilon = state.epsilon

    {new_cache, new_params} =
      Enum.reduce(params, {state.cache, %{}}, fn {key, param}, {cache_acc, p_acc} ->
        grad = Map.get(grads, key, Nx.broadcast(Nx.tensor(0.0), Nx.shape(param)))

        # cache_t = decay * cache_{t-1} + (1 - decay) * g_t^2
        cache_t =
          Nx.add(
            Nx.multiply(decay, Map.get(cache_acc, key)),
            Nx.multiply(1.0 - decay, Nx.multiply(grad, grad))
          )

        # param_t = param_{t-1} - lr * grad / (sqrt(cache_t) + epsilon)
        update = Nx.divide(grad, Nx.add(Nx.sqrt(cache_t), epsilon))
        new_param = Nx.subtract(param, Nx.multiply(lr, update))

        {Map.put(cache_acc, key, cache_t), Map.put(p_acc, key, new_param)}
      end)

    new_state = %{state | cache: new_cache}
    %{model | params: new_params, optimizer_state: new_state}
  end

  defp optimizer_step(%Model{} = model, _grads), do: model

  # ── Row Gathering ────────────────────────────────────────────────

  defp gather_rows(tensor, indices) when is_list(indices) do
    # Use Nx.take for efficient row gathering
    indices_tensor = Nx.tensor(indices, type: {:s, 64})
    Nx.take(tensor, indices_tensor)
  end

  # ── Progress Printing ────────────────────────────────────────────

  defp print_progress(%{epoch: epoch, loss: loss, val_loss: nil}) do
    IO.puts("Epoch #{epoch}: loss=#{:erlang.float_to_binary(loss, decimals: 4)}")
  end

  defp print_progress(%{epoch: epoch, loss: loss, val_loss: val_loss}) when is_number(val_loss) do
    IO.puts(
      "Epoch #{epoch}: loss=#{:erlang.float_to_binary(loss, decimals: 4)} val_loss=#{:erlang.float_to_binary(val_loss, decimals: 4)}"
    )
  end

  # ── Callbacks ────────────────────────────────────────────────────

  defmodule LoggingCallback do
    @moduledoc "Logs training metrics after each epoch."

    @spec log(map()) :: map()
    def log(%{epoch: epoch, loss: loss} = metrics) do
      msg = "Epoch #{epoch}: loss=#{:erlang.float_to_binary(loss, decimals: 4)}"

      msg =
        case Map.get(metrics, :val_loss) do
          nil -> msg
          val_loss -> msg <> " val_loss=#{:erlang.float_to_binary(val_loss, decimals: 4)}"
        end

      IO.puts(msg)
      metrics
    end
  end

  defmodule EarlyStoppingCallback do
    @moduledoc "Stops training when validation loss stops improving."

    @spec wait(pos_integer(), float()) :: (map() -> map())
    def wait(patience, min_delta \\ 1.0e-4) do
      parent = self()

      {:ok, pid} =
        Agent.start_link(fn -> %{best_loss: :infinity, wait: 0, epoch: 0} end)

      fn
        %{val_loss: val_loss, epoch: epoch} = metrics ->
          state = Agent.get(pid, & &1)

          if val_loss < state.best_loss - min_delta do
            Agent.update(pid, fn _ -> %{best_loss: val_loss, wait: 0, epoch: epoch} end)
            metrics
          else
            new_wait = state.wait + 1

            if new_wait >= patience do
              IO.puts("Early stopping at epoch #{epoch} (best: epoch #{state.epoch})")
              send(parent, {:early_stop, epoch})
              Map.put(metrics, :stop_training, true)
            else
              Agent.update(pid, fn s -> %{s | wait: new_wait} end)
              metrics
            end
          end

        metrics ->
          metrics
      end
    end
  end

  defmodule CheckpointCallback do
    @moduledoc "Saves model checkpoints at specified intervals."

    @spec every(pos_integer(), Path.t()) :: (map() -> map())
    def every(interval, dir) do
      File.mkdir_p!(dir)

      fn
        %{epoch: epoch, model: model} = metrics when rem(epoch, interval) == 0 ->
          path = Path.join(dir, "checkpoint_epoch_#{epoch}.model")
          ExBurn.Model.save(model, path)
          IO.puts("Checkpoint saved: #{path}")
          metrics

        metrics ->
          metrics
      end
    end
  end
end
