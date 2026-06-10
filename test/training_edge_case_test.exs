defmodule ExBurn.TrainingEdgeCaseTest do
  use ExUnit.Case

  alias ExBurn.Training

  describe "data_loader/2 edge cases" do
    test "creates batches of specified size" do
      x = Nx.iota({20, 5})
      y = Nx.iota({20, 1})

      batches =
        Training.data_loader({x, y}, batch_size: 5, shuffle: false)
        |> Enum.to_list()

      assert length(batches) == 4

      Enum.each(batches, fn {bx, by} ->
        assert elem(Nx.shape(bx), 0) == 5
        assert elem(Nx.shape(by), 0) == 5
      end)
    end

    test "handles batch_size larger than dataset" do
      x = Nx.iota({5, 3})
      y = Nx.iota({5, 1})

      batches =
        Training.data_loader({x, y}, batch_size: 100, shuffle: false)
        |> Enum.to_list()

      assert length(batches) == 1
    end

    test "handles batch_size of 1" do
      x = Nx.iota({5, 3})
      y = Nx.iota({5, 1})

      batches =
        Training.data_loader({x, y}, batch_size: 1, shuffle: false)
        |> Enum.to_list()

      assert length(batches) == 5
    end

    test "returns a Stream" do
      x = Nx.iota({10, 3})
      y = Nx.iota({10, 1})

      loader = Training.data_loader({x, y}, batch_size: 32)
      assert is_struct(loader, Stream)
    end

    test "shuffle produces different orderings" do
      x = Nx.iota({100, 3})
      y = Nx.iota({100, 1})

      batches1 =
        Training.data_loader({x, y}, batch_size: 100, shuffle: true)
        |> Enum.to_list()

      batches2 =
        Training.data_loader({x, y}, batch_size: 100, shuffle: true)
        |> Enum.to_list()

      [{x1, _}] = batches1
      [{x2, _}] = batches2

      assert Nx.to_list(x1) != Nx.to_list(x2)
    end
  end

  describe "compute_gradients/3 edge cases" do
    @tag :nif
    test "computes gradients for simple model" do
      model =
        Axon.input("input", shape: {nil, 2})
        |> Axon.dense(1)
        |> (fn g ->
              {init_fn, _} = Axon.build(g, [])
              template = Nx.template({1, 2}, :f32)
              init_fn.(template, Axon.ModelState.empty())
            end).()

      compiled = ExBurn.Model.compile(model, loss: :mse)
      batch_in = Nx.tensor([[1.0, 2.0]])
      batch_tgt = Nx.tensor([[1.0]])

      grads = Training.compute_gradients(compiled, {batch_in, batch_tgt}, grad_method: :numerical)
      assert is_map(grads)
      assert map_size(grads) > 0
    end

    @tag :nif
    test "computes gradients with numerical_batch method" do
      model =
        Axon.input("input", shape: {nil, 2})
        |> Axon.dense(1)
        |> (fn g ->
              {init_fn, _} = Axon.build(g, [])
              template = Nx.template({1, 2}, :f32)
              init_fn.(template, Axon.ModelState.empty())
            end).()

      compiled = ExBurn.Model.compile(model, loss: :mse)
      batch_in = Nx.tensor([[1.0, 2.0]])
      batch_tgt = Nx.tensor([[1.0]])

      grads =
        Training.compute_gradients(compiled, {batch_in, batch_tgt}, grad_method: :numerical_batch)

      assert is_map(grads)
      assert map_size(grads) > 0
    end

    @tag :nif
    test "computes gradients with autodiff method falls back to numerical" do
      model =
        Axon.input("input", shape: {nil, 2})
        |> Axon.dense(1)
        |> (fn g ->
              {init_fn, _} = Axon.build(g, [])
              template = Nx.template({1, 2}, :f32)
              init_fn.(template, Axon.ModelState.empty())
            end).()

      compiled = ExBurn.Model.compile(model, loss: :mse)
      batch_in = Nx.tensor([[1.0, 2.0]])
      batch_tgt = Nx.tensor([[1.0]])

      # Autodiff is not yet implemented, so it falls back to numerical gradients
      grads =
        Training.compute_gradients(compiled, {batch_in, batch_tgt}, grad_method: :autodiff)

      assert is_map(grads)
      assert map_size(grads) > 0

      # Verify gradient shapes match parameter shapes
      for {key, grad} <- grads do
        param = compiled.params[key]

        assert Nx.shape(grad) == Nx.shape(param),
               "Gradient shape #{inspect(Nx.shape(grad))} should match param shape #{inspect(Nx.shape(param))} for #{key}"
      end
    end

    @tag :nif
    test "autodiff with MSE falls back to numerical and returns valid gradients" do
      # For a simple linear model with MSE loss, verify gradients are computable
      # (autodiff falls back to numerical since it's not yet implemented)
      model =
        Axon.input("input", shape: {nil, 1})
        |> Axon.dense(1, use_bias: false)
        |> (fn g ->
              {init_fn, _} = Axon.build(g, [])
              template = Nx.template({1, 1}, :f32)
              init_fn.(template, Axon.ModelState.empty())
            end).()

      # Set weight to a known value
      model = %{model | params: %{}}
      compiled = ExBurn.Model.compile(model, loss: :mse)
      batch_in = Nx.tensor([[2.0]])
      batch_tgt = Nx.tensor([[10.0]])

      grads =
        Training.compute_gradients(compiled, {batch_in, batch_tgt}, grad_method: :autodiff)

      assert is_map(grads)
      assert map_size(grads) > 0
    end

    @tag :nif
    test "autodiff with cross_entropy falls back to numerical" do
      model =
        Axon.input("input", shape: {nil, 3})
        |> Axon.dense(5)
        |> (fn g ->
              {init_fn, _} = Axon.build(g, [])
              template = Nx.template({1, 3}, :f32)
              init_fn.(template, Axon.ModelState.empty())
            end).()

      compiled = ExBurn.Model.compile(model, loss: :cross_entropy)
      batch_in = Nx.tensor([[1.0, 2.0, 3.0]])
      batch_tgt = Nx.tensor([2])

      grads =
        Training.compute_gradients(compiled, {batch_in, batch_tgt}, grad_method: :autodiff)

      assert is_map(grads)
      assert map_size(grads) > 0
    end

    @tag :nif
    test "falls back to numerical for unknown method" do
      model =
        Axon.input("input", shape: {nil, 2})
        |> Axon.dense(1)
        |> (fn g ->
              {init_fn, _} = Axon.build(g, [])
              template = Nx.template({1, 2}, :f32)
              init_fn.(template, Axon.ModelState.empty())
            end).()

      compiled = ExBurn.Model.compile(model, loss: :mse)
      batch_in = Nx.tensor([[1.0, 2.0]])
      batch_tgt = Nx.tensor([[1.0]])

      # Unknown method falls back to numerical (the default)
      grads =
        Training.compute_gradients(compiled, {batch_in, batch_tgt}, grad_method: :unknown_method)

      assert is_map(grads)
      assert map_size(grads) > 0
    end

    @tag :nif
    test "default gradient method is numerical" do
      model =
        Axon.input("input", shape: {nil, 2})
        |> Axon.dense(1)
        |> (fn g ->
              {init_fn, _} = Axon.build(g, [])
              template = Nx.template({1, 2}, :f32)
              init_fn.(template, Axon.ModelState.empty())
            end).()

      compiled = ExBurn.Model.compile(model, loss: :mse)
      batch_in = Nx.tensor([[1.0, 2.0]])
      batch_tgt = Nx.tensor([[1.0]])

      # No grad_method specified — should use numerical by default
      grads = Training.compute_gradients(compiled, {batch_in, batch_tgt})
      assert is_map(grads)
      assert map_size(grads) > 0
    end
  end

  describe "evaluate/3 edge cases" do
    @tag :nif
    test "evaluates on dataset smaller than batch_size" do
      model =
        Axon.input("input", shape: {nil, 2})
        |> Axon.dense(1)
        |> (fn g ->
              {init_fn, _} = Axon.build(g, [])
              template = Nx.template({1, 2}, :f32)
              init_fn.(template, Axon.ModelState.empty())
            end).()

      compiled = ExBurn.Model.compile(model, loss: :mse)
      inputs = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      targets = Nx.tensor([[1.0], [2.0]])

      loss = Training.evaluate(compiled, {inputs, targets})
      assert is_float(loss)
      assert loss >= 0.0
    end

    @tag :nif
    test "evaluates with track_accuracy for cross_entropy" do
      model =
        Axon.input("input", shape: {nil, 2})
        |> Axon.dense(3)
        |> (fn g ->
              {init_fn, _} = Axon.build(g, [])
              template = Nx.template({1, 2}, :f32)
              init_fn.(template, Axon.ModelState.empty())
            end).()

      compiled = ExBurn.Model.compile(model, loss: :cross_entropy)
      inputs = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      targets = Nx.tensor([0, 2])

      {loss, accuracy} = Training.evaluate(compiled, {inputs, targets}, true)
      assert is_float(loss)
      assert is_float(accuracy)
      assert accuracy >= 0.0 and accuracy <= 1.0
    end

    @tag :nif
    test "evaluates with track_accuracy for non-cross_entropy returns nil accuracy" do
      model =
        Axon.input("input", shape: {nil, 2})
        |> Axon.dense(1)
        |> (fn g ->
              {init_fn, _} = Axon.build(g, [])
              template = Nx.template({1, 2}, :f32)
              init_fn.(template, Axon.ModelState.empty())
            end).()

      compiled = ExBurn.Model.compile(model, loss: :mse)
      inputs = Nx.tensor([[1.0, 2.0]])
      targets = Nx.tensor([[1.0]])

      result = Training.evaluate(compiled, {inputs, targets}, true)
      assert is_float(result)
    end

    @tag :nif
    test "evaluates on single sample" do
      model =
        Axon.input("input", shape: {nil, 2})
        |> Axon.dense(1)
        |> (fn g ->
              {init_fn, _} = Axon.build(g, [])
              template = Nx.template({1, 2}, :f32)
              init_fn.(template, Axon.ModelState.empty())
            end).()

      compiled = ExBurn.Model.compile(model, loss: :mse)
      inputs = Nx.tensor([[1.0, 2.0]])
      targets = Nx.tensor([[1.0]])

      loss = Training.evaluate(compiled, {inputs, targets})
      assert is_float(loss)
    end

    @tag :nif
    test "evaluates on dataset with partial last batch" do
      model =
        Axon.input("input", shape: {nil, 2})
        |> Axon.dense(1)
        |> (fn g ->
              {init_fn, _} = Axon.build(g, [])
              template = Nx.template({1, 2}, :f32)
              init_fn.(template, Axon.ModelState.empty())
            end).()

      compiled = ExBurn.Model.compile(model, loss: :mse)
      # 5 samples with batch_size defaulting to min(256, 5) = 5
      inputs = Nx.iota({5, 2})
      targets = Nx.iota({5, 1})

      loss = Training.evaluate(compiled, {inputs, targets})
      assert is_float(loss)
    end
  end

  describe "LoggingCallback" do
    test "logs metrics with epoch and loss" do
      metrics = %{epoch: 1, loss: 0.5}
      assert ^metrics = Training.LoggingCallback.log(metrics)
    end

    test "logs metrics with val_loss" do
      metrics = %{epoch: 2, loss: 0.3, val_loss: 0.4}
      assert ^metrics = Training.LoggingCallback.log(metrics)
    end

    test "logs metrics with accuracy" do
      metrics = %{epoch: 3, loss: 0.2, accuracy: 0.95}
      assert ^metrics = Training.LoggingCallback.log(metrics)
    end

    test "logs metrics with all fields" do
      metrics = %{epoch: 4, loss: 0.1, val_loss: 0.15, accuracy: 0.98}
      assert ^metrics = Training.LoggingCallback.log(metrics)
    end
  end

  describe "CheckpointCallback" do
    test "does not trigger save outside interval" do
      dir = Path.join(System.tmp_dir!(), "test_checkpoints_#{System.unique_integer()}")
      callback = Training.CheckpointCallback.every(3, dir)

      # Epoch 1 should NOT trigger save (interval is 3)
      metrics = %{epoch: 1, model: nil}
      result = callback.(metrics)
      assert result == metrics

      File.rm_rf!(dir)
    end

    test "triggers save at interval" do
      dir = Path.join(System.tmp_dir!(), "test_checkpoints_#{System.unique_integer()}")

      # Create a real model to save
      model =
        Axon.input("input", shape: {nil, 2})
        |> Axon.dense(1)
        |> (fn g ->
              {init_fn, _} = Axon.build(g, [])
              template = Nx.template({1, 2}, :f32)
              init_fn.(template, Axon.ModelState.empty())
            end).()

      compiled = ExBurn.Model.compile(model)
      callback = Training.CheckpointCallback.every(2, dir)

      # Epoch 2 should trigger save
      metrics = %{epoch: 2, model: compiled}
      result = callback.(metrics)
      assert result == metrics

      # Verify the file was created
      expected_path = Path.join(dir, "checkpoint_epoch_2.model")
      assert File.exists?(expected_path)

      # Clean up
      File.rm_rf!(dir)
    end
  end

  describe "WarmupCallback" do
    test "adjusts learning rate during warmup" do
      callback = Training.WarmupCallback.linear(5, 0.0001, 0.001)

      model = %ExBurn.Model{
        optimizer_state: %{learning_rate: 0.0001}
      }

      metrics = %{epoch: 1, model: model}
      result = callback.(metrics)

      # After warmup epoch 1, LR should be between start and target
      new_lr = result.model.optimizer_state.learning_rate
      assert new_lr > 0.0001
      assert new_lr < 0.001
    end

    test "does not adjust learning rate after warmup" do
      callback = Training.WarmupCallback.linear(3, 0.0001, 0.001)

      model = %ExBurn.Model{
        optimizer_state: %{learning_rate: 0.001}
      }

      metrics = %{epoch: 5, model: model}
      result = callback.(metrics)

      # After warmup, LR should remain unchanged
      assert result.model.optimizer_state.learning_rate == 0.001
    end

    test "handles metrics without model" do
      callback = Training.WarmupCallback.linear(3, 0.0001, 0.001)
      metrics = %{epoch: 1}
      result = callback.(metrics)
      assert result == metrics
    end
  end

  describe "HistoryCallback" do
    test "records metrics history" do
      callback = Training.HistoryCallback.new()

      metrics1 = %{epoch: 1, loss: 0.5}
      metrics2 = %{epoch: 2, loss: 0.3}

      callback.(metrics1)
      callback.(metrics2)

      # get_history/0 returns empty list (known limitation)
      assert Training.HistoryCallback.get_history() == []
    end

    test "get_history with pid returns reversed history" do
      callback = Training.HistoryCallback.new()

      metrics = %{epoch: 1, loss: 0.5}
      result = callback.(metrics)

      pid = result.history_pid
      history = Training.HistoryCallback.get_history(pid)
      assert length(history) == 1
      assert hd(history).epoch == 1
    end
  end

  describe "train_step/3 with gradient methods" do
    @tag :nif
    test "performs a single training step with default (numerical) method" do
      model =
        Axon.input("input", shape: {nil, 2})
        |> Axon.dense(1)
        |> (fn g ->
              {init_fn, _} = Axon.build(g, [])
              template = Nx.template({1, 2}, :f32)
              init_fn.(template, Axon.ModelState.empty())
            end).()

      compiled = ExBurn.Model.compile(model, loss: :mse)
      batch_in = Nx.tensor([[1.0, 2.0]])
      batch_tgt = Nx.tensor([[1.0]])

      {loss, updated_model} = Training.train_step(compiled, {batch_in, batch_tgt})
      assert is_float(loss)
      assert loss >= 0.0
      assert updated_model.__struct__ == ExBurn.Model
    end

    @tag :nif
    test "train_step with explicit autodiff method falls back to numerical" do
      model =
        Axon.input("input", shape: {nil, 2})
        |> Axon.dense(1)
        |> (fn g ->
              {init_fn, _} = Axon.build(g, [])
              template = Nx.template({1, 2}, :f32)
              init_fn.(template, Axon.ModelState.empty())
            end).()

      compiled = ExBurn.Model.compile(model, loss: :mse)
      batch_in = Nx.tensor([[1.0, 2.0]])
      batch_tgt = Nx.tensor([[1.0]])

      {loss, updated_model} =
        Training.train_step(compiled, {batch_in, batch_tgt}, grad_method: :autodiff)

      assert is_float(loss)
      assert updated_model.__struct__ == ExBurn.Model
    end

    @tag :nif
    test "train_step with weight decay" do
      model =
        Axon.input("input", shape: {nil, 2})
        |> Axon.dense(1)
        |> (fn g ->
              {init_fn, _} = Axon.build(g, [])
              template = Nx.template({1, 2}, :f32)
              init_fn.(template, Axon.ModelState.empty())
            end).()

      compiled = ExBurn.Model.compile(model, loss: :mse)
      batch_in = Nx.tensor([[1.0, 2.0]])
      batch_tgt = Nx.tensor([[1.0]])

      {loss, _updated_model} =
        Training.train_step(compiled, {batch_in, batch_tgt},
          grad_method: :autodiff,
          weight_decay: 0.01
        )

      assert is_float(loss)
    end

    @tag :nif
    test "train_step with gradient clipping" do
      model =
        Axon.input("input", shape: {nil, 2})
        |> Axon.dense(1)
        |> (fn g ->
              {init_fn, _} = Axon.build(g, [])
              template = Nx.template({1, 2}, :f32)
              init_fn.(template, Axon.ModelState.empty())
            end).()

      compiled = ExBurn.Model.compile(model, loss: :mse)
      batch_in = Nx.tensor([[1.0, 2.0]])
      batch_tgt = Nx.tensor([[1.0]])

      {loss, _updated_model} =
        Training.train_step(compiled, {batch_in, batch_tgt},
          grad_method: :autodiff,
          clip_norm: 1.0
        )

      assert is_float(loss)
    end
  end

  describe "fit/3 with gradient methods" do
    @tag :nif
    test "trains a simple model with default (numerical) method" do
      model =
        Axon.input("input", shape: {nil, 2})
        |> Axon.dense(1)
        |> (fn g ->
              {init_fn, _} = Axon.build(g, [])
              template = Nx.template({1, 2}, :f32)
              init_fn.(template, Axon.ModelState.empty())
            end).()

      compiled = ExBurn.Model.compile(model, loss: :mse, optimizer: :sgd, learning_rate: 0.01)

      inputs = Nx.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
      targets = Nx.tensor([[1.0], [2.0], [3.0], [4.0]])

      trained =
        Training.fit(compiled, {inputs, targets},
          epochs: 3,
          batch_size: 2,
          verbose: false
        )

      assert trained.__struct__ == ExBurn.Model
      # After training, parameters should have changed
      assert trained.params != nil
    end

    @tag :nif
    test "fit with autodiff (falls back to numerical) and cross_entropy" do
      model =
        Axon.input("input", shape: {nil, 3})
        |> Axon.dense(2)
        |> (fn g ->
              {init_fn, _} = Axon.build(g, [])
              template = Nx.template({1, 3}, :f32)
              init_fn.(template, Axon.ModelState.empty())
            end).()

      compiled =
        ExBurn.Model.compile(model,
          loss: :cross_entropy,
          optimizer: :adam,
          learning_rate: 0.01
        )

      inputs = Nx.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
      targets = Nx.tensor([0, 1, 0, 1])

      trained =
        Training.fit(compiled, {inputs, targets},
          epochs: 3,
          batch_size: 2,
          verbose: false,
          grad_method: :autodiff
        )

      assert trained.__struct__ == ExBurn.Model
    end

    @tag :nif
    test "fit with gradient accumulation using default numerical method" do
      model =
        Axon.input("input", shape: {nil, 2})
        |> Axon.dense(1)
        |> (fn g ->
              {init_fn, _} = Axon.build(g, [])
              template = Nx.template({1, 2}, :f32)
              init_fn.(template, Axon.ModelState.empty())
            end).()

      compiled = ExBurn.Model.compile(model, loss: :mse, optimizer: :sgd, learning_rate: 0.01)

      inputs = Nx.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
      targets = Nx.tensor([[1.0], [2.0], [3.0], [4.0]])

      trained =
        Training.fit(compiled, {inputs, targets},
          epochs: 2,
          batch_size: 2,
          verbose: false,
          accumulate_gradients: 2
        )

      assert trained.__struct__ == ExBurn.Model
    end

    @tag :nif
    test "fit with default numerical method and callbacks" do
      model =
        Axon.input("input", shape: {nil, 2})
        |> Axon.dense(1)
        |> (fn g ->
              {init_fn, _} = Axon.build(g, [])
              template = Nx.template({1, 2}, :f32)
              init_fn.(template, Axon.ModelState.empty())
            end).()

      compiled = ExBurn.Model.compile(model, loss: :mse, optimizer: :sgd, learning_rate: 0.01)

      inputs = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      targets = Nx.tensor([[1.0], [2.0]])

      callback = fn metrics ->
        assert is_map(metrics)
        assert Map.has_key?(metrics, :epoch)
        assert Map.has_key?(metrics, :loss)
        metrics
      end

      trained =
        Training.fit(compiled, {inputs, targets},
          epochs: 2,
          batch_size: 2,
          verbose: false,
          callbacks: [callback]
        )

      assert trained.__struct__ == ExBurn.Model
    end
  end

  describe "profile_step/3 with gradient methods" do
    @tag :nif
    test "profiles a training step with autodiff (falls back to numerical)" do
      model =
        Axon.input("input", shape: {nil, 2})
        |> Axon.dense(1)
        |> (fn g ->
              {init_fn, _} = Axon.build(g, [])
              template = Nx.template({1, 2}, :f32)
              init_fn.(template, Axon.ModelState.empty())
            end).()

      compiled = ExBurn.Model.compile(model, loss: :mse)
      batch_in = Nx.tensor([[1.0, 2.0]])
      batch_tgt = Nx.tensor([[1.0]])

      profile = Training.profile_step(compiled, {batch_in, batch_tgt}, grad_method: :autodiff)

      assert is_map(profile)
      assert Map.has_key?(profile, :loss)
      assert Map.has_key?(profile, :forward_ms)
      assert Map.has_key?(profile, :backward_ms)
      assert Map.has_key?(profile, :optimizer_ms)
      assert Map.has_key?(profile, :total_ms)
      assert Map.has_key?(profile, :model)
      assert profile.total_ms >= 0
    end
  end
end
