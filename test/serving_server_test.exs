defmodule ExBurn.ServingServerTest do
  use ExUnit.Case, async: false

  alias ExBurn.Serving.Server

  defp tiny_graph do
    Axon.input("input", shape: {nil, 4}) |> Axon.dense(2)
  end

  defp compiled_model do
    ExBurn.Model.compile(tiny_graph(), loss: :mse)
  end

  defp sample(idx), do: Nx.tensor([idx * 1.0, idx + 1.0, idx + 2.0, idx + 3.0])

  describe "init/3" do
    test "returns the model as state" do
      model = compiled_model()
      assert {:ok, ^model} = Server.init(nil, {model, 1}, [])
    end
  end

  describe "handle_batch/3" do
    @tag :nif
    test "returns an execute thunk running prediction over the batch" do
      model = compiled_model()
      batch = Nx.Batch.stack([sample(0), sample(1)])

      {:execute, fun, ^model} = Server.handle_batch(batch, %{ref: make_ref()}, model)

      {output, ^model} = fun.()
      assert Nx.shape(output) == {2, 2}
      assert Nx.type(output) == {:f, 32}
    end

    @tag :nif
    test "raises a structured error when the model has no predict function" do
      # Compiling directly from a ModelState leaves predict_fn nil
      {init_fn, _predict} = Axon.build(tiny_graph())
      model_state = init_fn.(Nx.template({1, 4}, :f32), Axon.ModelState.empty())
      model = ExBurn.Model.compile(model_state, loss: :mse)

      batch = Nx.Batch.stack([sample(0)])
      {:execute, fun, ^model} = Server.handle_batch(batch, %{}, model)

      assert_raise ExBurn.Error, ~r/serving_predict/, fn -> fun.() end
    end
  end

  describe "Nx.Serving integration" do
    @tag :nif
    test "serving runs end-to-end through Nx.Serving.run" do
      model = compiled_model()

      serving =
        Nx.Serving.new(Server, {model, 1}, batch_size: 2, batch_timeout: 5)

      result = Nx.Serving.run(serving, Nx.Batch.stack([sample(3)]))

      output =
        case result do
          %Nx.Serving{} = s -> Map.get(s, :output)
          tensor -> tensor
        end

      assert Nx.shape(output) == {1, 2}
      assert Enum.all?(Nx.to_flat_list(output), &is_float/1)
    end
  end

  describe "ExBurn.Serving wrapper" do
    @tag :nif
    test "new with lazy?: true defers building the serving" do
      lazy = ExBurn.Serving.new(compiled_model(), lazy?: true)
      assert is_nil(lazy.serving)
      assert lazy.batch_size == 32
    end

    @tag :nif
    test "run/2 builds the serving on demand for lazy servings" do
      lazy = ExBurn.Serving.new(compiled_model(), lazy?: true)
      result = ExBurn.Serving.run(lazy, Nx.Batch.stack([sample(1)]))

      output =
        case result do
          %Nx.Serving{} = s -> Map.get(s, :output)
          tensor -> tensor
        end

      assert Nx.shape(output) == {1, 2}
    end

    @tag :nif
    test "run/2 delegates to the pre-built serving" do
      wrapper = ExBurn.Serving.new(compiled_model(), batch_size: 2)
      refute is_nil(wrapper.serving)

      result = ExBurn.Serving.run(wrapper, Nx.Batch.stack([sample(2)]))

      output =
        case result do
          %Nx.Serving{} = s -> Map.get(s, :output)
          tensor -> tensor
        end

      assert Nx.shape(output) == {1, 2}
    end

    test "with_batch_size rebuilds or updates depending on laziness" do
      built = ExBurn.Serving.new(compiled_model(), batch_size: 4)
      updated = ExBurn.Serving.with_batch_size(built, 8)
      assert updated.batch_size == 8
      refute is_nil(updated.serving)

      lazy = ExBurn.Serving.new(compiled_model(), lazy?: true)
      still_lazy = ExBurn.Serving.with_batch_size(lazy, 16)
      assert still_lazy.batch_size == 16
      assert is_nil(still_lazy.serving)
    end

    test "with_timeout updates both variants" do
      built = ExBurn.Serving.new(compiled_model())
      updated = ExBurn.Serving.with_timeout(built, 250)
      assert updated.batch_timeout == 250
      refute is_nil(updated.serving)

      lazy = ExBurn.Serving.new(compiled_model(), lazy?: true)
      assert ExBurn.Serving.with_timeout(lazy, 100).batch_timeout == 100
    end

    test "status exposes configuration" do
      wrapper = ExBurn.Serving.new(compiled_model(), padding: true, partitions: 2)

      assert ExBurn.Serving.status(wrapper) == %{
               batch_size: 32,
               batch_timeout: 50,
               partitions: 2,
               padding: true
             }
    end
  end
end
