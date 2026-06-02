defmodule ExBurn.TrainingLRScheduleTest do
  use ExUnit.Case

  alias ExBurn.Training

  # Helper to build a simple Axon.ModelState
  defp simple_model do
    Axon.input("input", shape: {nil, 2})
    |> Axon.dense(1)
    |> (fn g ->
          {init_fn, _} = Axon.build(g, [])
          template = Nx.template({1, 2}, :f32)
          init_fn.(template, Axon.ModelState.empty())
        end).()
  end

  describe "fit/3 with step LR schedule" do
    @tag :nif
    test "decays learning rate at step boundaries" do
      model = simple_model()
      compiled = ExBurn.Model.compile(model, loss: :mse, optimizer: :sgd, learning_rate: 0.1)

      inputs = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      targets = Nx.tensor([[1.0], [2.0]])

      callback = fn metrics ->
        send(self(), {:lr, metrics.epoch, metrics.model.optimizer_state.learning_rate})
        metrics
      end

      Training.fit(compiled, {inputs, targets},
        epochs: 6,
        batch_size: 2,
        verbose: false,
        lr_schedule: {:step, 0.1, 3, 0.5},
        callbacks: [callback]
      )

      # Epoch 1-3: LR = 0.1
      assert_receive {:lr, 1, lr1}
      assert_in_delta lr1, 0.1, 1.0e-6
      assert_receive {:lr, 3, lr3}
      assert_in_delta lr3, 0.1, 1.0e-6

      # Epoch 4-6: LR = 0.1 * 0.5 = 0.05
      assert_receive {:lr, 4, lr4}
      assert_in_delta lr4, 0.05, 1.0e-6
      assert_receive {:lr, 6, lr6}
      assert_in_delta lr6, 0.05, 1.0e-6
    end
  end

  describe "fit/3 with exponential LR schedule" do
    @tag :nif
    test "decays learning rate exponentially" do
      model = simple_model()
      compiled = ExBurn.Model.compile(model, loss: :mse, optimizer: :sgd, learning_rate: 0.1)

      inputs = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      targets = Nx.tensor([[1.0], [2.0]])

      callback = fn metrics ->
        send(self(), {:lr, metrics.epoch, metrics.model.optimizer_state.learning_rate})
        metrics
      end

      Training.fit(compiled, {inputs, targets},
        epochs: 3,
        batch_size: 2,
        verbose: false,
        lr_schedule: {:exponential, 0.1, 0.5},
        callbacks: [callback]
      )

      # Epoch 1: LR = 0.1 * 0.5^0 = 0.1
      assert_receive {:lr, 1, lr1}
      assert_in_delta lr1, 0.1, 1.0e-6

      # Epoch 2: LR = 0.1 * 0.5^1 = 0.05
      assert_receive {:lr, 2, lr2}
      assert_in_delta lr2, 0.05, 1.0e-6

      # Epoch 3: LR = 0.1 * 0.5^2 = 0.025
      assert_receive {:lr, 3, lr3}
      assert_in_delta lr3, 0.025, 1.0e-6
    end
  end

  describe "fit/3 with cosine LR schedule" do
    @tag :nif
    test "decays learning rate following cosine curve" do
      model = simple_model()
      compiled = ExBurn.Model.compile(model, loss: :mse, optimizer: :sgd, learning_rate: 0.1)

      inputs = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      targets = Nx.tensor([[1.0], [2.0]])

      callback = fn metrics ->
        send(self(), {:lr, metrics.epoch, metrics.model.optimizer_state.learning_rate})
        metrics
      end

      Training.fit(compiled, {inputs, targets},
        epochs: 4,
        batch_size: 2,
        verbose: false,
        lr_schedule: {:cosine, 0.1, 0.001},
        callbacks: [callback]
      )

      # Epoch 1: LR ≈ 0.1 (start of cosine)
      assert_receive {:lr, 1, lr1}
      assert_in_delta lr1, 0.1, 0.01

      # Epoch 4: LR ≈ 0.001 (end of cosine, near min_lr)
      assert_receive {:lr, 4, lr4}
      assert_in_delta lr4, 0.001, 0.01

      # LR should decrease monotonically
      assert_receive {:lr, 2, lr2}
      assert_receive {:lr, 3, lr3}
      assert lr1 > lr2
      assert lr2 > lr3
      assert lr3 > lr4
    end
  end

  describe "fit/3 with nesterov momentum" do
    @tag :nif
    test "trains with nesterov momentum enabled" do
      model = simple_model()
      compiled = ExBurn.Model.compile(model, loss: :mse, optimizer: :sgd, learning_rate: 0.01)

      inputs = Nx.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
      targets = Nx.tensor([[1.0], [2.0], [3.0], [4.0]])

      trained =
        Training.fit(compiled, {inputs, targets},
          epochs: 5,
          batch_size: 2,
          verbose: false,
          nesterov: true
        )

      assert trained.__struct__ == ExBurn.Model
      assert trained.optimizer_state.nesterov == true
    end
  end

  describe "fit/3 with clip_value" do
    @tag :nif
    test "trains with gradient value clipping" do
      model = simple_model()
      compiled = ExBurn.Model.compile(model, loss: :mse, optimizer: :sgd, learning_rate: 0.01)

      inputs = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      targets = Nx.tensor([[1.0], [2.0]])

      trained =
        Training.fit(compiled, {inputs, targets},
          epochs: 3,
          batch_size: 2,
          verbose: false,
          clip_value: 1.0
        )

      assert trained.__struct__ == ExBurn.Model
    end
  end

  describe "fit/3 with early stopping" do
    @tag :nif
    test "stops training when validation loss plateaus" do
      model = simple_model()
      compiled = ExBurn.Model.compile(model, loss: :mse, optimizer: :sgd, learning_rate: 0.001)

      # Use identical inputs/targets so loss doesn't improve
      inputs = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      targets = Nx.tensor([[1.0], [2.0]])

      callback = fn metrics ->
        send(self(), {:epoch, metrics.epoch})
        metrics
      end

      trained =
        Training.fit(compiled, {inputs, targets},
          epochs: 20,
          batch_size: 2,
          verbose: false,
          validation_data: {inputs, targets},
          callbacks: [
            ExBurn.Training.EarlyStoppingCallback.wait(2, 1.0e-4),
            callback
          ]
        )

      # Should have stopped before epoch 20
      assert trained.__struct__ == ExBurn.Model

      # Collect all epoch messages
      epochs_received =
        Enum.reduce(1..20, [], fn _, acc ->
          receive do
            {:epoch, e} -> [e | acc]
          after
            0 -> acc
          end
        end)

      # Should have stopped early (before epoch 20)
      assert length(epochs_received) < 20
    end
  end

  describe "fit/3 with ReduceLROnPlateau" do
    @tag :nif
    test "reduces learning rate when loss plateaus" do
      model = simple_model()
      compiled = ExBurn.Model.compile(model, loss: :mse, optimizer: :sgd, learning_rate: 0.1)

      inputs = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      targets = Nx.tensor([[1.0], [2.0]])

      trained =
        Training.fit(compiled, {inputs, targets},
          epochs: 10,
          batch_size: 2,
          verbose: false,
          validation_data: {inputs, targets},
          callbacks: [
            ExBurn.Training.ReduceLROnPlateauCallback.new(patience: 2, factor: 0.5, min_lr: 0.001)
          ]
        )

      assert trained.__struct__ == ExBurn.Model
      # LR should have been reduced from initial 0.1
      assert trained.optimizer_state.learning_rate < 0.1
    end
  end

  describe "fit/3 returns trained model with updated params" do
    @tag :nif
    test "parameters change after training" do
      model = simple_model()
      compiled = ExBurn.Model.compile(model, loss: :mse, optimizer: :sgd, learning_rate: 0.01)

      inputs = Nx.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
      targets = Nx.tensor([[1.0], [2.0], [3.0], [4.0]])

      original_params = compiled.params

      trained =
        Training.fit(compiled, {inputs, targets},
          epochs: 5,
          batch_size: 2,
          verbose: false
        )

      # Parameters should have changed
      assert trained.params != nil

      # At least some parameter values should differ
      params_changed =
        Enum.any?(trained.params, fn {key, tensor} ->
          original = Map.get(original_params, key)
          original && not match?(^original, tensor)
        end)

      assert params_changed, "Parameters should have changed after training"
    end
  end

  describe "fit/3 with gradient accumulation" do
    @tag :nif
    test "accumulates gradients across multiple batches" do
      model = simple_model()
      compiled = ExBurn.Model.compile(model, loss: :mse, optimizer: :sgd, learning_rate: 0.01)

      # 4 samples, batch_size=2, accumulate=2 → effective batch = 4
      inputs = Nx.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
      targets = Nx.tensor([[1.0], [2.0], [3.0], [4.0]])

      trained =
        Training.fit(compiled, {inputs, targets},
          epochs: 3,
          batch_size: 2,
          verbose: false,
          accumulate_gradients: 2
        )

      assert trained.__struct__ == ExBurn.Model
    end
  end

  describe "fit/3 with accuracy tracking" do
    @tag :nif
    test "tracks accuracy during training" do
      model =
        Axon.input("input", shape: {nil, 3})
        |> Axon.dense(2)
        |> (fn g ->
              {init_fn, _} = Axon.build(g, [])
              template = Nx.template({1, 3}, :f32)
              init_fn.(template, Axon.ModelState.empty())
            end).()

      compiled =
        ExBurn.Model.compile(model, loss: :cross_entropy, optimizer: :adam, learning_rate: 0.01)

      inputs = Nx.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
      targets = Nx.tensor([0, 1, 0, 1])

      callback = fn metrics ->
        if Map.has_key?(metrics, :accuracy) do
          send(self(), {:accuracy, metrics.accuracy})
        end

        metrics
      end

      Training.fit(compiled, {inputs, targets},
        epochs: 3,
        batch_size: 2,
        verbose: false,
        accuracy: true,
        callbacks: [callback]
      )

      # Should have received accuracy metrics
      assert_receive {:accuracy, acc}
      assert is_float(acc)
      assert acc >= 0.0 and acc <= 1.0
    end
  end
end
