defmodule ExBurn.Training do
  @moduledoc """
  Training loop implementation for ExBurn models.

  Provides a flexible training loop with support for:

  - Mini-batch training with gradient computation
  - Multiple optimizers (Adam, SGD with momentum / Nesterov, RMSprop)
  - Learning rate scheduling (step, exponential, cosine)
  - Gradient clipping (by norm and by value)
  - Weight decay (L2 regularization)
  - Gradient accumulation for effective larger batch sizes
  - Batch shuffling each epoch
  - Validation with partial batch handling
  - Training metrics tracking (loss, accuracy)
  - Progress reporting with ETA and throughput
  - Callbacks (logging, early stopping, checkpointing)
  - Public `train_step/3` and `compute_gradients/3` for custom loops

  ## Usage

      model = ExBurn.Model.compile(axon_model, loss: :cross_entropy, optimizer: :adam)

      opts = [
        epochs: 10,
        batch_size: 32,
        shuffle: true,
        validation_data: val_data,
        lr_schedule: {:cosine, 0.001, 1.0e-5},
        clip_norm: 1.0,
        weight_decay: 1.0e-4,
        accumulate_gradients: 4,
        accuracy: true,
        nesterov: true,
        callbacks: [&ExBurn.Training.LoggingCallback.log/1]
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
          shuffle: boolean(),
          validation_data: dataset() | nil,
          callbacks: [callback()],
          verbose: boolean(),
          lr_schedule: lr_schedule(),
          clip_norm: float() | nil,
          clip_value: float() | nil,
          weight_decay: float() | nil,
          accumulate_gradients: pos_integer(),
          accuracy: boolean(),
          nesterov: boolean()
        ]

  @doc """
  Trains a model on the given dataset.

  ## Options

    * `:epochs` — Number of training epochs (default: 10)
    * `:batch_size` — Mini-batch size (default: 32)
    * `:shuffle` — Shuffle training data each epoch (default: true)
    * `:validation_data` — Validation dataset as `{inputs, targets}` tuple
    * `:callbacks` — List of callback functions called after each epoch
    * `:verbose` — Print training progress (default: true)
    * `:lr_schedule` — Learning rate schedule (default: nil)
    * `:clip_norm` — Max gradient norm for clipping (default: nil)
    * `:clip_value` — Max absolute gradient value for clipping (default: nil)
    * `:weight_decay` — L2 regularization coefficient (default: nil)
    * `:accumulate_gradients` — Number of mini-batches to accumulate before
      an optimizer step, effectively multiplying batch size (default: 1)
    * `:accuracy` — Compute and report classification accuracy (default: false)
    * `:nesterov` — Use Nesterov momentum for SGD optimizer (default: false)

  ## Returns

    The trained `ExBurn.Model` struct with updated parameters.
  """
  @spec fit(Model.t(), dataset(), keyword()) :: Model.t()
  def fit(%Model{} = model, {inputs, targets}, opts \\ []) do
    epochs = Keyword.get(opts, :epochs, 10)
    batch_size = Keyword.get(opts, :batch_size, 32)
    shuffle = Keyword.get(opts, :shuffle, true)
    validation_data = Keyword.get(opts, :validation_data)
    callbacks = Keyword.get(opts, :callbacks, [])
    verbose = Keyword.get(opts, :verbose, true)
    lr_schedule = Keyword.get(opts, :lr_schedule)
    clip_norm = Keyword.get(opts, :clip_norm)
    clip_value = Keyword.get(opts, :clip_value)
    weight_decay = Keyword.get(opts, :weight_decay)
    accumulate = Keyword.get(opts, :accumulate_gradients, 1)
    track_accuracy = Keyword.get(opts, :accuracy, false)
    nesterov = Keyword.get(opts, :nesterov, false)

    num_samples = Nx.shape(inputs) |> elem(0)
    num_batches = div(num_samples, batch_size)

    # Enable Nesterov in SGD optimizer state if requested
    model =
      if nesterov && model.optimizer == :sgd do
        %{model | optimizer_state: Map.put(model.optimizer_state, :nesterov, true)}
      else
        model
      end

    if verbose do
      effective_bs = batch_size * accumulate
      IO.puts("Training: #{num_samples} samples, #{num_batches} batches/epoch, #{epochs} epochs")

      IO.puts(
        "  batch_size=#{batch_size}, effective_batch_size=#{effective_bs}, optimizer=#{model.optimizer}"
      )

      opts_summary =
        [
          if(weight_decay, do: "weight_decay=#{weight_decay}", else: nil),
          if(track_accuracy, do: "accuracy=true", else: nil),
          if(nesterov, do: "nesterov=true", else: nil),
          if(accumulate > 1, do: "accumulate=#{accumulate}", else: nil)
        ]
        |> Enum.reject(&is_nil/1)
        |> Enum.join(", ")

      if opts_summary != "", do: IO.puts("  #{opts_summary}")
    end

    # Set ExBurn as the default backend for Nx operations
    Nx.default_backend(ExBurn.Backend)

    start_time = System.monotonic_time(:millisecond)

    {trained_model, _final_metrics} =
      Enum.reduce_while({1, epochs}, {model, %{}}, fn epoch, {model, _metrics} ->
        epoch_start = System.monotonic_time(:millisecond)

        # Apply learning rate schedule
        model = apply_lr_schedule(model, lr_schedule, epoch, epochs)

        # Train one epoch
        {epoch_loss, epoch_correct, epoch_total, model} =
          train_epoch(
            model,
            inputs,
            targets,
            batch_size,
            num_batches,
            clip_norm,
            clip_value,
            weight_decay,
            accumulate,
            track_accuracy,
            shuffle
          )

        epoch_elapsed = System.monotonic_time(:millisecond) - epoch_start
        total_elapsed = System.monotonic_time(:millisecond) - start_time

        # Build metrics map
        avg_loss = epoch_loss / max(num_batches, 1)

        metrics =
          %{
            epoch: epoch,
            loss: avg_loss,
            model: model,
            epoch_time_ms: epoch_elapsed,
            total_time_ms: total_elapsed
          }

        # Compute accuracy if tracking
        metrics =
          if track_accuracy && epoch_total > 0 do
            acc = epoch_correct / epoch_total
            Map.put(metrics, :accuracy, acc)
          else
            metrics
          end

        # Compute ETA
        metrics =
          if epoch < epochs do
            avg_epoch_time = total_elapsed / epoch
            remaining_epochs = epochs - epoch
            eta_ms = avg_epoch_time * remaining_epochs
            Map.put(metrics, :eta_ms, eta_ms)
          else
            metrics
          end

        metrics =
          if validation_data do
            val_result = evaluate(model, validation_data, track_accuracy)

            case val_result do
              {val_loss, nil} ->
                Map.put(metrics, :val_loss, val_loss)

              {val_loss, val_acc} ->
                Map.put(metrics, :val_loss, val_loss) |> Map.put(:val_accuracy, val_acc)
            end
          else
            metrics
          end

        if verbose do
          print_progress(metrics, num_samples, epoch_elapsed)
        end

        # Run callbacks — each callback receives and returns the metrics map
        metrics = Enum.reduce(callbacks, metrics, fn callback, acc -> callback.(acc) end)

        # Check for early stopping
        if Map.get(metrics, :stop_training) do
          {:halt, {model, metrics}}
        else
          {:cont, {model, metrics}}
        end
      end)

    trained_model
  after
    # Restore default backend
    Nx.default_backend(Nx.BinaryBackend)
  end

  @doc """
  Performs a single training step: forward + backward + optimizer update.

  This is a public API for custom training loops. It processes a single
  mini-batch and returns the updated model along with the loss value.

  ## Parameters

    * `model` — The current model state
    * `batch` — A `{inputs, targets}` tuple for one mini-batch
    * `opts` — Options (same as `fit/3` for clip_norm, clip_value, weight_decay)

  ## Returns

    `{loss, updated_model}` where loss is a float.
  """
  @spec train_step(Model.t(), dataset(), keyword()) :: {float(), Model.t()}
  def train_step(%Model{} = model, {batch_in, batch_tgt}, opts \\ []) do
    clip_norm = Keyword.get(opts, :clip_norm)
    clip_value = Keyword.get(opts, :clip_value)
    weight_decay = Keyword.get(opts, :weight_decay)

    # Forward pass
    {:ok, pred} = Model.predict(model, batch_in)
    {:ok, loss} = Model.compute_loss(model, pred, batch_tgt)
    loss_val = Nx.to_number(loss)

    # Backward pass: compute gradients
    grads = compute_gradients(model, {batch_in, batch_tgt}, opts)

    # Add weight decay gradients (L2 regularization)
    grads =
      if weight_decay && weight_decay > 0 do
        add_weight_decay_grads(grads, model.params, weight_decay)
      else
        grads
      end

    # Clip gradients
    grads =
      grads
      |> maybe_clip_by_norm(clip_norm)
      |> maybe_clip_by_value(clip_value)

    # Optimizer step
    model = optimizer_step(model, grads)

    {loss_val, model}
  end

  @doc """
  Computes gradients for a given mini-batch.

  Supports multiple gradient computation methods via the `:grad_method` option:

    * `:numerical` — Central finite differences (default, slow but general)
    * `:numerical_batch` — Numerical gradients computed on the full batch at once
      (more efficient, fewer forward passes)

  ## Parameters

    * `model` — The current model state
    * `batch` — A `{inputs, targets}` tuple
    * `opts` — Options list

  ## Options

    * `:grad_method` — Gradient computation method (default: `:numerical`)
    * `:epsilon` — Finite difference step size (default: 1.0e-5)

  ## Returns

    A map of `{param_key => gradient_tensor}`.
  """
  @spec compute_gradients(Model.t(), dataset(), keyword()) :: map()
  def compute_gradients(%Model{} = model, {batch_in, batch_tgt}, opts \\ []) do
    method = Keyword.get(opts, :grad_method, :numerical)
    epsilon = Keyword.get(opts, :epsilon, 1.0e-5)

    case method do
      :numerical ->
        compute_gradients_numerical(model, batch_in, batch_tgt, epsilon)

      :numerical_batch ->
        compute_gradients_numerical_batch(model, batch_in, batch_tgt, epsilon)

      _ ->
        compute_gradients_numerical(model, batch_in, batch_tgt, epsilon)
    end
  end

  @doc """
  Evaluates a model on a dataset.

  Returns the average loss over the entire dataset.
  When `track_accuracy` is true, returns `{loss, accuracy}` where accuracy
  is a float or `nil` if the loss function is not cross_entropy.
  """
  @spec evaluate(Model.t(), dataset(), boolean()) :: float() | {float(), float() | nil}
  def evaluate(%Model{} = model, {inputs, targets}, track_accuracy \\ false) do
    num_samples = Nx.shape(inputs) |> elem(0)

    # Process in batches to avoid OOM
    batch_size = min(256, num_samples)
    num_full_batches = div(num_samples, batch_size)
    has_remainder = rem(num_samples, batch_size) > 0

    # Process full batches
    {total_loss, total_correct, total_count} =
      Enum.reduce(0..(num_full_batches - 1), {0.0, 0, 0}, fn batch_idx,
                                                             {loss_acc, correct_acc, count_acc} ->
        start_idx = batch_idx * batch_size
        actual_bs = batch_size

        batch_in = slice_batch(inputs, start_idx, actual_bs)
        batch_tgt = slice_batch(targets, start_idx, actual_bs)

        {:ok, pred} = Model.predict(model, batch_in)
        {:ok, loss} = Model.compute_loss(model, pred, batch_tgt)
        loss_val = Nx.to_number(loss)

        {correct, count} =
          if track_accuracy && model.loss_fn == :cross_entropy do
            compute_batch_accuracy(pred, batch_tgt)
          else
            {0, 0}
          end

        {loss_acc + loss_val, correct_acc + correct, count_acc + count}
      end)

    # Handle last partial batch
    {total_loss, total_correct, total_count} =
      if has_remainder do
        start_idx = num_full_batches * batch_size
        actual_bs = num_samples - start_idx

        batch_in = slice_batch(inputs, start_idx, actual_bs)
        batch_tgt = slice_batch(targets, start_idx, actual_bs)

        {:ok, pred} = Model.predict(model, batch_in)
        {:ok, loss} = Model.compute_loss(model, pred, batch_tgt)
        loss_val = Nx.to_number(loss)

        {correct, count} =
          if track_accuracy && model.loss_fn == :cross_entropy do
            compute_batch_accuracy(pred, batch_tgt)
          else
            {0, 0}
          end

        {total_loss + loss_val, total_correct + correct, total_count + count}
      else
        {total_loss, total_correct, total_count}
      end

    total_batches = num_full_batches + if(has_remainder, do: 1, else: 0)
    avg_loss = total_loss / max(total_batches, 1)

    if track_accuracy do
      accuracy = if total_count > 0, do: total_correct / total_count, else: nil
      {avg_loss, accuracy}
    else
      avg_loss
    end
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

  defp train_epoch(
         model,
         inputs,
         targets,
         batch_size,
         _num_batches,
         clip_norm,
         clip_value,
         weight_decay,
         accumulate,
         track_accuracy,
         shuffle
       ) do
    num_samples = Nx.shape(inputs) |> elem(0)

    indices =
      if shuffle do
        Enum.shuffle(0..(num_samples - 1))
      else
        Enum.to_list(0..(num_samples - 1))
      end

    batches = Enum.chunk_every(indices, batch_size, batch_size, [])

    # Process batches with gradient accumulation
    {total_loss, total_correct, total_count, model} =
      accumulate_batches(
        batches,
        inputs,
        targets,
        model,
        clip_norm,
        clip_value,
        weight_decay,
        accumulate,
        track_accuracy,
        {0.0, 0, 0, model}
      )

    {total_loss, total_correct, total_count, model}
  end

  # Process batches with gradient accumulation
  defp accumulate_batches(
         [],
         _inputs,
         _targets,
         model,
         _clip_norm,
         _clip_value,
         _weight_decay,
         _accumulate,
         _track_acc,
         {loss, correct, count, model}
       ) do
    {loss, correct, count, model}
  end

  defp accumulate_batches(
         batches,
         inputs,
         targets,
         model,
         clip_norm,
         clip_value,
         weight_decay,
         accumulate,
         track_accuracy,
         acc
       ) do
    # Take up to `accumulate` batches
    {to_process, remaining} = Enum.split(batches, accumulate)

    {batch_loss, batch_correct, batch_count, updated_model} =
      process_accumulated_batches(
        to_process,
        inputs,
        targets,
        model,
        clip_norm,
        clip_value,
        weight_decay,
        track_accuracy
      )

    {loss, correct, count, _old_model} = acc

    accumulate_batches(
      remaining,
      inputs,
      targets,
      updated_model,
      clip_norm,
      clip_value,
      weight_decay,
      accumulate,
      track_accuracy,
      {loss + batch_loss, correct + batch_correct, count + batch_count, updated_model}
    )
  end

  # Process a group of accumulated batches: average their gradients, then step
  defp process_accumulated_batches(
         batch_indices_list,
         inputs,
         targets,
         model,
         clip_norm,
         clip_value,
         weight_decay,
         track_accuracy
       ) do
    num_batches = length(batch_indices_list)

    {total_loss, total_correct, total_count, accumulated_grads, model} =
      Enum.reduce(
        batch_indices_list,
        {0.0, 0, 0, %{}, model},
        fn batch_indices, {loss_acc, correct_acc, count_acc, grad_acc, model} ->
          batch_in = gather_rows(inputs, batch_indices)
          batch_tgt = gather_rows(targets, batch_indices)

          # Forward pass
          {:ok, pred} = Model.predict(model, batch_in)
          {:ok, loss} = Model.compute_loss(model, pred, batch_tgt)
          loss_val = Nx.to_number(loss)

          # Compute gradients
          grads = compute_gradients(model, {batch_in, batch_tgt}, grad_method: :numerical)

          # Accumulate gradients
          grad_acc =
            Enum.reduce(grads, grad_acc, fn {key, grad}, acc ->
              case Map.fetch(acc, key) do
                {:ok, existing} -> Map.put(acc, key, Nx.add(existing, grad))
                :error -> Map.put(acc, key, grad)
              end
            end)

          # Track accuracy
          {correct, count} =
            if track_accuracy && model.loss_fn == :cross_entropy do
              compute_batch_accuracy(pred, batch_tgt)
            else
              {0, 0}
            end

          {loss_acc + loss_val, correct_acc + correct, count_acc + count, grad_acc, model}
        end
      )

    # Average accumulated gradients
    avg_grads =
      Enum.map(accumulated_grads, fn {key, grad} ->
        {key, Nx.divide(grad, num_batches * 1.0)}
      end)
      |> Map.new()

    # Add weight decay gradients
    grads =
      if weight_decay && weight_decay > 0 do
        add_weight_decay_grads(avg_grads, model.params, weight_decay)
      else
        avg_grads
      end

    # Clip gradients
    grads =
      grads
      |> maybe_clip_by_norm(clip_norm)
      |> maybe_clip_by_value(clip_value)

    # Optimizer step
    model = optimizer_step(model, grads)

    {total_loss, total_correct, total_count, model}
  end

  # ── Gradient Computation ─────────────────────────────────────────

  defp compute_gradients_numerical(%Model{params: params} = model, input, target, epsilon) do
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

  # Batch numerical gradient: computes gradient for all params using full-batch
  # forward passes. More efficient as it amortizes the forward pass cost.
  defp compute_gradients_numerical_batch(%Model{params: params} = model, input, target, epsilon) do
    # Compute base loss once
    {:ok, base_pred} = Model.predict(model, input)
    {:ok, base_loss} = Model.compute_loss(model, base_pred, target)
    base_loss_val = Nx.to_number(base_loss)

    grads =
      Enum.map(params, fn {key, param_value} ->
        shape = Nx.shape(param_value)
        flat = Nx.flatten(param_value)
        n = Nx.size(flat)
        flat_binary = Nx.to_binary(flat)

        grad_data =
          Enum.map(0..(n - 1), fn i ->
            <<_before::binary-size(i * 4), current_val::float-32-little, _after::binary>> =
              flat_binary

            # Only need f(x + eps) since we already have f(x)
            plus_binary =
              binary_replace(flat_binary, i * 4, <<current_val + epsilon::float-32-little>>)

            plus = Nx.from_binary(plus_binary, :f32)
            plus_param = Nx.reshape(plus, shape)
            model_plus = put_in_model_param(model, key, plus_param)
            {:ok, loss_plus} = Model.predict(model_plus, input)
            {:ok, loss_plus} = Model.compute_loss(model_plus, loss_plus, target)
            lp = Nx.to_number(loss_plus)

            # One-sided finite difference: (f(x+eps) - f(x)) / eps
            (lp - base_loss_val) / epsilon
          end)

        grad_binary =
          Enum.map(grad_data, fn val -> <<val::float-32-little>> end)
          |> :erlang.list_to_binary()

        grad_flat = Nx.from_binary(grad_binary, :f32)
        {key, Nx.reshape(grad_flat, shape)}
      end)
      |> Map.new()

    grads
  end

  defp binary_replace(binary, offset, replacement) do
    <<before::binary-size(offset), _::binary-size(byte_size(replacement)), rest::binary>> = binary
    before <> replacement <> rest
  end

  defp put_in_model_param(%Model{} = model, key, value) do
    %{model | params: Map.put(model.params, key, value)}
  end

  # ── Weight Decay (L2 Regularization) ─────────────────────────────

  defp add_weight_decay_grads(grads, params, weight_decay) do
    Enum.map(grads, fn {key, grad} ->
      param = Map.get(params, key)

      if param do
        # grad_wd = grad + weight_decay * param
        {key, Nx.add(grad, Nx.multiply(weight_decay, param))}
      else
        {key, grad}
      end
    end)
    |> Map.new()
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
    nesterov = Map.get(state, :nesterov, false)

    {new_velocity, new_params} =
      Enum.reduce(params, {state.velocity, %{}}, fn {key, param}, {vel_acc, p_acc} ->
        grad = Map.get(grads, key, Nx.broadcast(Nx.tensor(0.0), Nx.shape(param)))

        # v_t = momentum * v_{t-1} + grad
        v_t = Nx.add(Nx.multiply(momentum, Map.get(vel_acc, key)), grad)

        # Parameter update
        {update, stored_v} =
          if nesterov do
            # Nesterov: param -= lr * (momentum * v_t + grad)
            # v_t already = momentum * v_{t-1} + grad
            # So: update = momentum * v_t + grad
            nesterov_update = Nx.add(Nx.multiply(momentum, v_t), grad)
            {nesterov_update, v_t}
          else
            # Standard momentum: param -= lr * v_t
            {v_t, v_t}
          end

        new_param = Nx.subtract(param, Nx.multiply(lr, update))

        {Map.put(vel_acc, key, stored_v), Map.put(p_acc, key, new_param)}
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
    indices_tensor = Nx.tensor(indices, type: {:s, 64})
    Nx.take(tensor, indices_tensor)
  end

  # ── Batch Slicing (handles 1D and 2D+ targets) ──────────────────

  defp slice_batch(tensor, start_idx, batch_size) do
    rank = Nx.rank(tensor)

    if rank == 1 do
      Nx.slice(tensor, [start_idx], [batch_size])
    else
      Nx.slice(tensor, [start_idx, 0], [batch_size, elem(Nx.shape(tensor), 1)])
    end
  end

  # ── Accuracy Computation ─────────────────────────────────────────

  defp compute_batch_accuracy(pred, target) do
    # pred is logits, target is class indices or one-hot
    pred_classes = Nx.argmax(pred, axis: -1)

    correct =
      if Nx.rank(target) == Nx.rank(pred) do
        # One-hot encoded targets — convert to class indices
        target_classes = Nx.argmax(target, axis: -1)
        Nx.equal(pred_classes, target_classes)
      else
        # Integer class indices
        Nx.equal(pred_classes, target)
      end

    num_correct = Nx.sum(correct) |> Nx.to_number()
    num_total = Nx.shape(pred) |> elem(0)
    {num_correct, num_total}
  end

  # ── Progress Printing ────────────────────────────────────────────

  defp print_progress(%{epoch: epoch, loss: loss, val_loss: nil} = metrics, num_samples, epoch_ms) do
    samples_per_sec = num_samples / max(epoch_ms, 1) * 1000
    base = "Epoch #{epoch}: loss=#{:erlang.float_to_binary(loss, decimals: 4)}"

    base =
      if Map.has_key?(metrics, :accuracy) do
        acc = Map.get(metrics, :accuracy)
        base <> " acc=#{:erlang.float_to_binary(acc * 100, decimals: 1)}%"
      else
        base
      end

    base = base <> " (#{round(samples_per_sec)} samples/s, #{epoch_ms}ms)"

    base =
      if Map.has_key?(metrics, :eta_ms) do
        eta_sec = round(Map.get(metrics, :eta_ms) / 1000)
        base <> " ETA=#{format_duration(eta_sec)}"
      else
        base
      end

    IO.puts(base)
  end

  defp print_progress(
         %{epoch: epoch, loss: loss, val_loss: val_loss} = metrics,
         num_samples,
         epoch_ms
       )
       when is_number(val_loss) do
    samples_per_sec = num_samples / max(epoch_ms, 1) * 1000

    base =
      "Epoch #{epoch}: loss=#{:erlang.float_to_binary(loss, decimals: 4)} val_loss=#{:erlang.float_to_binary(val_loss, decimals: 4)}"

    base =
      if Map.has_key?(metrics, :accuracy) do
        acc = Map.get(metrics, :accuracy)
        val_acc = Map.get(metrics, :val_accuracy)

        base <>
          " acc=#{:erlang.float_to_binary(acc * 100, decimals: 1)}%" <>
          if(val_acc,
            do: " val_acc=#{:erlang.float_to_binary(val_acc * 100, decimals: 1)}%",
            else: ""
          )
      else
        base
      end

    base = base <> " (#{round(samples_per_sec)} samples/s, #{epoch_ms}ms)"

    base =
      if Map.has_key?(metrics, :eta_ms) do
        eta_sec = round(Map.get(metrics, :eta_ms) / 1000)
        base <> " ETA=#{format_duration(eta_sec)}"
      else
        base
      end

    IO.puts(base)
  end

  defp format_duration(seconds) when seconds < 60, do: "#{seconds}s"

  defp format_duration(seconds) when seconds < 3600,
    do: "#{div(seconds, 60)}m#{rem(seconds, 60)}s"

  defp format_duration(seconds), do: "#{div(seconds, 3600)}h#{div(rem(seconds, 3600), 60)}m"

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

      msg =
        case Map.get(metrics, :accuracy) do
          nil -> msg
          acc -> msg <> " acc=#{:erlang.float_to_binary(acc * 100, decimals: 1)}%"
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
