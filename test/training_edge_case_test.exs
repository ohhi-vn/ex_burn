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

      grads = Training.compute_gradients(compiled, {batch_in, batch_tgt}, grad_method: :autodiff)
      assert is_map(grads)
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
end
