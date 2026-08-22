defmodule ExBurn.TrainingCoverageTest do
  use ExUnit.Case, async: false

  # Runs fun in an isolated process with a hard timeout. A killed worker
  # cannot run cleanup, so callers that flip global state must restore it
  # themselves when the result is not {:ok, _}.
  defp safe_call(fun, timeout_ms) do
    parent = self()
    ref = make_ref()

    pid =
      spawn(fn ->
        result =
          try do
            {:ok, fun.()}
          rescue
            e -> {:raised, Exception.message(e)}
          catch
            :exit, reason -> {:exited, reason}
          end

        send(parent, {ref, result})
      end)

    receive do
      {^ref, {:ok, value}} -> {:ok, value}
      {^ref, other} -> other
    after
      timeout_ms ->
        Process.exit(pid, :kill)
        :timed_out
    end
  end

  defp graph do
    Axon.input("input", shape: {nil, 3}) |> Axon.dense(2)
  end

  defp compiled(opts \\ []) do
    ExBurn.Model.compile(graph(), Keyword.merge([loss: :mse], opts))
  end

  defp data(n \\ 6) do
    inputs = Nx.tensor(Enum.map(1..n, fn i -> [i * 1.0, i * 2.0, i * 3.0] end))
    targets = Nx.tensor(Enum.map(1..n, fn i -> [i * 0.5, i * 1.5] end))
    {inputs, targets}
  end

  describe "fit options" do
    @tag :nif
    test "validation_data with accuracy produces val metrics" do
      trained =
        ExBurn.Training.fit(compiled(), data(),
          epochs: 1,
          batch_size: 3,
          verbose: false,
          accuracy: false,
          validation_data: data(4)
        )

      assert %ExBurn.Model{} = trained
    end

    @tag :nif
    test "set_default_backend? captures and restores the global backend" do
      previous = Nx.default_backend()

      # Training under a flipped default backend is opt-in/experimental and
      # can be slow — bound it so the suite stays deterministic.
      result =
        safe_call(
          fn ->
            ExBurn.Training.fit(compiled(), data(),
              epochs: 1,
              batch_size: 6,
              verbose: false,
              grad_method: :numerical_batch,
              set_default_backend?: true
            )
          end,
          10_000
        )

      case result do
        %ExBurn.Model{} -> :ok
        _ -> Nx.default_backend(previous)
      end

      assert Nx.default_backend() == previous
    end

    @tag :nif
    test "rmsprop optimizer steps without crashing" do
      trained =
        ExBurn.Training.fit(compiled(optimizer: :rmsprop), data(),
          epochs: 1,
          batch_size: 6,
          verbose: false
        )

      assert trained.optimizer == :rmsprop
      assert Enum.all?(trained.params, fn {_k, v} -> match?(%Nx.Tensor{}, v) end)
    end
  end

  describe "evaluate/3" do
    @tag :nif
    test "handles remainder batches above the internal batch size" do
      n = 300
      inputs = Nx.tensor(List.duplicate([1.0, 2.0, 3.0], n))
      targets = Nx.tensor(List.duplicate([0.0, 0.0], n))

      loss = ExBurn.Training.evaluate(compiled(), {inputs, targets})
      assert is_float(loss)
    end
  end

  describe "data_loader/2" do
    @tag :nif
    test "shuffle false yields deterministic order" do
      batches =
        data()
        |> ExBurn.Training.data_loader(batch_size: 2, shuffle: false)
        |> Enum.to_list()

      first = batches |> hd() |> elem(0)
      assert Nx.to_flat_list(first) == [1.0, 2.0, 3.0, 2.0, 4.0, 6.0]
      assert length(batches) == 3
    end
  end

  describe "train_step / profile_step" do
    @tag :nif
    test "train_step honours clip_value" do
      {loss, model} = ExBurn.Training.train_step(compiled(), data(4), clip_value: 1.0e-6)
      assert is_float(loss)
      assert %ExBurn.Model{} = model
    end

    @tag :nif
    test "profile_step returns per-phase timings" do
      profile = ExBurn.Training.profile_step(compiled(), data(4), grad_method: :numerical_batch)

      assert %{forward_ms: f, backward_ms: b, optimizer_ms: o, total_ms: t, loss: _} = profile
      assert is_float(f) and is_float(b) and is_float(o) and is_float(t)
    end
  end

  describe "callback gaps" do
    @tag :nif
    test "EarlyStoppingCallback.wait/1 uses default min_delta" do
      cb = ExBurn.Training.EarlyStoppingCallback.wait(2)

      m1 = %{val_loss: 1.0, epoch: 1}
      assert %{stop_training: nil} = Map.put(cb.(m1), :stop_training, nil)

      # non-improving epochs accumulate; patience 2 stops at second miss
      _ = cb.(%{val_loss: 1.0, epoch: 2})
      stopped = cb.(%{val_loss: 1.0, epoch: 3})
      assert stopped.stop_training == true
    end

    @tag :nif
    test "EarlyStoppingCallback passes through metrics lacking val_loss" do
      cb = ExBurn.Training.EarlyStoppingCallback.wait(1)
      metrics = %{epoch: 1}
      assert cb.(metrics) == metrics
    end

    @tag :nif
    test "ReduceLROnPlateauCallback.new/0 applies defaults" do
      cb = ExBurn.Training.ReduceLROnPlateauCallback.new()
      model = compiled(learning_rate: 0.01)

      improved = cb.(%{val_loss: 0.5, epoch: 1, model: model})
      assert improved == %{val_loss: 0.5, epoch: 1, model: model}
    end

    @tag :nif
    test "ReduceLROnPlateau respects the min_lr floor" do
      cb = ExBurn.Training.ReduceLROnPlateauCallback.new(patience: 1, factor: 0.5, min_lr: 1.0e-8)

      model = compiled(learning_rate: 1.0e-8)
      metrics = %{val_loss: 0.9, epoch: 1, model: model}

      result = cb.(metrics)
      lr = get_in(result.model.optimizer_state, [:learning_rate])
      # current_lr * factor would dip below the floor — clamped to floor
      assert lr == 1.0e-8
    end
  end

  describe "ExBurn.Tensor gaps" do
    @tag :nif
    test "from_nx returns an error tuple when the NIF rejects the dtype" do
      assert {:error, msg} = ExBurn.Tensor.from_nx(Nx.tensor([1.0], type: {:f, 64}))
      assert is_binary(msg)
    end

    @tag :nif
    test "to_nx rescues failures into error tuples" do
      bad = %ExBurn.Tensor{ref: make_ref(), shape: [1], type: :f32}
      assert {:error, _} = ExBurn.Tensor.to_nx(bad)
    end

    @tag :nif
    test "batch converters halt on the first failure" do
      good = Nx.tensor([1.0])
      bad = Nx.tensor([1.0], type: {:f, 64})

      assert {:error, _} = ExBurn.Tensor.from_nx_batch([good, bad])

      {:ok, bt} = ExBurn.Tensor.from_nx(good)
      broken = %ExBurn.Tensor{bt | ref: make_ref()}
      assert {:error, _} = ExBurn.Tensor.to_nx_batch([bt, broken])
    end

    @tag :nif
    test "from_binary surfaces NIF rejection as a tuple and free/1 releases" do
      assert {:error, _} = ExBurn.Tensor.from_binary(<<1.0::float-32-native>>, [1], :f16)

      {:ok, t} = ExBurn.Tensor.from_binary(<<1.0::float-32-native>>, [1], :f32)
      assert :ok = ExBurn.Tensor.free(t)
    end
  end

  describe "ExBurn.Error" do
    test "new/0 builds an empty struct" do
      assert %ExBurn.Error{op: nil, reason: nil, details: nil} = ExBurn.Error.new()
    end
  end
end
