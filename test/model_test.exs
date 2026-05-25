defmodule ExBurn.ModelTest do
  use ExUnit.Case

  # Helper to build an Axon.ModelState from an Axon model graph.
  # Uses Axon.build/2 which returns {init_fn, predict_fn}, then
  # calls init_fn with a template and empty ModelState.
  defp axon_model(axon_graph) do
    {init_fn, _predict_fn} = Axon.build(axon_graph, [])
    template = Nx.template({1, 1}, :f32)
    init_fn.(template, Axon.ModelState.empty())
  end

  describe "compile/2" do
    test "compiles a simple model" do
      model =
        Axon.input("input", shape: {nil, 10})
        |> Axon.dense(5)
        |> axon_model()

      compiled = ExBurn.Model.compile(model, loss: :mse, optimizer: :adam)
      assert compiled.compiled == true
      assert is_map(compiled.params)
      assert map_size(compiled.params) > 0
    end

    test "initializes optimizer state" do
      model =
        Axon.input("input", shape: {nil, 10})
        |> Axon.dense(5)
        |> axon_model()

      compiled = ExBurn.Model.compile(model, optimizer: :adam)
      assert compiled.optimizer_state.type == :adam
      assert compiled.optimizer_state.learning_rate == 0.001
      assert compiled.optimizer_state.beta1 == 0.9
      assert compiled.optimizer_state.beta2 == 0.999
    end

    test "supports SGD optimizer" do
      model =
        Axon.input("input", shape: {nil, 10})
        |> Axon.dense(5)
        |> axon_model()

      compiled = ExBurn.Model.compile(model, optimizer: :sgd)
      assert compiled.optimizer_state.type == :sgd
      assert compiled.optimizer_state.momentum == 0.9
    end

    test "supports RMSprop optimizer" do
      model =
        Axon.input("input", shape: {nil, 10})
        |> Axon.dense(5)
        |> axon_model()

      compiled = ExBurn.Model.compile(model, optimizer: :rmsprop)
      assert compiled.optimizer_state.type == :rmsprop
      assert compiled.optimizer_state.decay == 0.9
    end
  end

  describe "parameters/1" do
    test "returns the parameter map" do
      model =
        Axon.input("input", shape: {nil, 10})
        |> Axon.dense(5)
        |> axon_model()

      compiled = ExBurn.Model.compile(model)
      params = ExBurn.Model.parameters(compiled)
      assert is_map(params)
    end
  end

  describe "update_params/2" do
    test "updates model parameters" do
      model =
        Axon.input("input", shape: {nil, 10})
        |> Axon.dense(5)
        |> axon_model()

      compiled = ExBurn.Model.compile(model)
      new_params = %{test: Nx.tensor([1.0])}
      updated = ExBurn.Model.update_params(compiled, new_params)
      assert updated.params == new_params
    end
  end

  describe "freeze/2 and unfreeze/2" do
    test "freezes and unfreezes layers" do
      model =
        Axon.input("input", shape: {nil, 10})
        |> Axon.dense(5, name: "layer1")
        |> Axon.dense(3, name: "layer2")
        |> axon_model()

      compiled = ExBurn.Model.compile(model)

      # Initially no frozen layers
      assert ExBurn.Model.frozen_layers(compiled) == MapSet.new()

      # Freeze a layer
      frozen = ExBurn.Model.freeze(compiled, ["layer1"])
      assert ExBurn.Model.frozen?(frozen, "layer1")
      refute ExBurn.Model.frozen?(frozen, "layer2")

      # Unfreeze it
      unfrozen = ExBurn.Model.unfreeze(frozen, ["layer1"])
      refute ExBurn.Model.frozen?(unfrozen, "layer1")
    end
  end

  describe "to_device/2" do
    test "returns same model when already on target device" do
      model =
        Axon.input("input", shape: {nil, 10})
        |> Axon.dense(5)
        |> axon_model()

      compiled = ExBurn.Model.compile(model, device: :gpu)
      assert ExBurn.Model.to_device(compiled, :gpu) == compiled
    end
  end

  describe "compute_loss/3" do
    test "computes MSE loss" do
      model =
        Axon.input("input", shape: {nil, 10})
        |> Axon.dense(5)
        |> axon_model()

      compiled = ExBurn.Model.compile(model, loss: :mse)
      pred = Nx.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
      target = Nx.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])

      {:ok, loss} = ExBurn.Model.compute_loss(compiled, pred, target)
      assert Nx.to_number(loss) < 1.0e-6
    end

    test "computes cross-entropy loss" do
      model =
        Axon.input("input", shape: {nil, 10})
        |> Axon.dense(5)
        |> axon_model()

      compiled = ExBurn.Model.compile(model, loss: :cross_entropy)
      pred = Nx.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
      target = Nx.tensor([[0.0, 0.0, 0.0, 0.0, 1.0]])

      {:ok, loss} = ExBurn.Model.compute_loss(compiled, pred, target)
      assert Nx.to_number(loss) > 0
    end

    test "returns error for unsupported loss" do
      model =
        Axon.input("input", shape: {nil, 10})
        |> Axon.dense(5)
        |> axon_model()

      compiled = ExBurn.Model.compile(model, loss: :custom_loss)
      pred = Nx.tensor([[1.0]])
      target = Nx.tensor([[1.0]])

      assert {:error, _} = ExBurn.Model.compute_loss(compiled, pred, target)
    end
  end

  describe "save/2 and load/2" do
    test "saves and loads model parameters" do
      model =
        Axon.input("input", shape: {nil, 5})
        |> Axon.dense(3)
        |> axon_model()

      compiled = ExBurn.Model.compile(model)
      path = Path.join(System.tmp_dir!(), "test_model_#{System.unique_integer()}.model")

      try do
        assert ExBurn.Model.save(compiled, path) == :ok
        assert {:ok, loaded} = ExBurn.Model.load(compiled, path)
        assert loaded.compiled == true
      after
        File.rm(path)
      end
    end

    test "load returns error for missing file" do
      model =
        Axon.input("input", shape: {nil, 5})
        |> Axon.dense(3)
        |> axon_model()

      compiled = ExBurn.Model.compile(model)
      assert {:error, _} = ExBurn.Model.load(compiled, "/nonexistent/path.model")
    end
  end

  describe "serialize_params/1 and deserialize_params/1" do
    test "round-trips parameters" do
      model =
        Axon.input("input", shape: {nil, 5})
        |> Axon.dense(3)
        |> axon_model()

      compiled = ExBurn.Model.compile(model)
      binary = ExBurn.Model.serialize_params(compiled)
      assert is_binary(binary)

      assert {:ok, params} = ExBurn.Model.deserialize_params(binary)
      assert is_map(params)
    end

    test "deserialize returns error for invalid binary" do
      assert {:error, _} = ExBurn.Model.deserialize_params(<<1, 2, 3>>)
    end
  end

  describe "summary/1" do
    test "returns a string summary" do
      model =
        Axon.input("input", shape: {nil, 10})
        |> Axon.dense(5, name: "hidden")
        |> Axon.dense(3, name: "output")
        |> axon_model()

      compiled = ExBurn.Model.compile(model)
      summary = ExBurn.Model.summary(compiled)
      assert is_binary(summary)
      assert summary =~ "ExBurn Model Summary"
    end
  end

  describe "forward_pattern/1" do
    test "returns output shape info" do
      model =
        Axon.input("input", shape: {nil, 10})
        |> Axon.dense(5)
        |> axon_model()

      compiled = ExBurn.Model.compile(model)
      pattern = ExBurn.Model.forward_pattern(compiled)
      assert is_map(pattern)
      assert Map.has_key?(pattern, :output_shape)
      assert Map.has_key?(pattern, :output_type)
    end
  end

  describe "quantize/2" do
    test "quantizes parameters to f16" do
      model =
        Axon.input("input", shape: {nil, 5})
        |> Axon.dense(3)
        |> axon_model()

      compiled = ExBurn.Model.compile(model)
      quantized = ExBurn.Model.quantize(compiled, :f16)
      assert quantized.params != nil
    end

    test "quantizes parameters to bf16" do
      model =
        Axon.input("input", shape: {nil, 5})
        |> Axon.dense(3)
        |> axon_model()

      compiled = ExBurn.Model.compile(model)
      quantized = ExBurn.Model.quantize(compiled, :bf16)
      assert quantized.params != nil
    end
  end

  describe "clone/1" do
    test "creates a deep copy" do
      model =
        Axon.input("input", shape: {nil, 5})
        |> Axon.dense(3)
        |> axon_model()

      compiled = ExBurn.Model.compile(model)
      cloned = ExBurn.Model.clone(compiled)

      assert cloned.params != nil
      assert cloned.loss_fn == compiled.loss_fn
      assert cloned.optimizer == compiled.optimizer
      assert cloned.device == compiled.device
    end
  end

  describe "info/1" do
    test "returns model information map" do
      model =
        Axon.input("input", shape: {nil, 10})
        |> Axon.dense(5)
        |> axon_model()

      compiled = ExBurn.Model.compile(model)
      info = ExBurn.Model.info(compiled)

      assert is_map(info)
      assert is_integer(info.total_params)
      assert info.total_params > 0
      assert info.loss_function == :cross_entropy
      assert info.optimizer == :adam
      assert is_integer(info.layer_count)
      assert info.compiled == true
    end
  end
end
