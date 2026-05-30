defmodule ExBurn.ModelEdgeCaseTest do
  use ExUnit.Case

  # Helper to build an Axon.ModelState from an Axon model graph.
  defp axon_model(axon_graph) do
    {init_fn, _predict_fn} = Axon.build(axon_graph, [])
    template = Nx.template({1, 1}, :f32)
    init_fn.(template, Axon.ModelState.empty())
  end

  describe "compile/2 edge cases" do
    test "compiles with custom learning rate" do
      model =
        Axon.input("input", shape: {nil, 10})
        |> Axon.dense(5)
        |> axon_model()

      compiled = ExBurn.Model.compile(model, learning_rate: 0.01)
      assert compiled.optimizer_state.learning_rate == 0.01
    end

    test "compiles with CPU device" do
      model =
        Axon.input("input", shape: {nil, 10})
        |> Axon.dense(5)
        |> axon_model()

      compiled = ExBurn.Model.compile(model, device: :cpu)
      assert compiled.device == :cpu
    end

    test "compiles with weight decay" do
      model =
        Axon.input("input", shape: {nil, 10})
        |> Axon.dense(5)
        |> axon_model()

      compiled = ExBurn.Model.compile(model, weight_decay: 0.01)
      assert compiled.weight_decay == 0.01
    end

    test "compiles with binary_cross_entropy loss" do
      model =
        Axon.input("input", shape: {nil, 10})
        |> Axon.dense(1)
        |> axon_model()

      compiled = ExBurn.Model.compile(model, loss: :binary_cross_entropy)
      assert compiled.loss_fn == :binary_cross_entropy
    end

    test "compiles multi-layer model" do
      model =
        Axon.input("input", shape: {nil, 10})
        |> Axon.dense(64, name: "hidden1")
        |> Axon.dense(32, name: "hidden2")
        |> Axon.dense(10, name: "output")
        |> axon_model()

      compiled = ExBurn.Model.compile(model)
      assert compiled.compiled == true
      assert map_size(compiled.params) > 0
    end
  end

  describe "compute_loss/3 edge cases" do
    test "MSE loss with identical pred and target" do
      model =
        Axon.input("input", shape: {nil, 10})
        |> Axon.dense(5)
        |> axon_model()

      compiled = ExBurn.Model.compile(model, loss: :mse)
      pred = Nx.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
      target = Nx.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])

      {:ok, loss} = ExBurn.Model.compute_loss(compiled, pred, target)
      assert_in_delta Nx.to_number(loss), 0.0, 1.0e-6
    end

    test "MSE loss with different pred and target" do
      model =
        Axon.input("input", shape: {nil, 10})
        |> Axon.dense(2)
        |> axon_model()

      compiled = ExBurn.Model.compile(model, loss: :mse)
      pred = Nx.tensor([[0.0, 0.0]])
      target = Nx.tensor([[1.0, 1.0]])

      {:ok, loss} = ExBurn.Model.compute_loss(compiled, pred, target)
      # MSE = mean((0-1)^2 + (0-1)^2) = mean(1 + 1) = 1.0
      assert_in_delta Nx.to_number(loss), 1.0, 1.0e-4
    end

    test "cross-entropy loss with one-hot targets" do
      model =
        Axon.input("input", shape: {nil, 10})
        |> Axon.dense(3)
        |> axon_model()

      compiled = ExBurn.Model.compile(model, loss: :cross_entropy)
      pred = Nx.tensor([[1.0, 2.0, 3.0]])
      target = Nx.tensor([[0.0, 0.0, 1.0]])

      {:ok, loss} = ExBurn.Model.compute_loss(compiled, pred, target)
      assert Nx.to_number(loss) > 0
    end

    test "binary cross entropy with perfect prediction" do
      model =
        Axon.input("input", shape: {nil, 10})
        |> Axon.dense(1)
        |> axon_model()

      compiled = ExBurn.Model.compile(model, loss: :binary_cross_entropy)
      # pred close to 1, target = 1 → low loss
      pred = Nx.tensor([[0.99]])
      target = Nx.tensor([[1.0]])

      {:ok, loss} = ExBurn.Model.compute_loss(compiled, pred, target)
      assert Nx.to_number(loss) < 0.1
    end

    test "binary cross entropy with wrong prediction" do
      model =
        Axon.input("input", shape: {nil, 10})
        |> Axon.dense(1)
        |> axon_model()

      compiled = ExBurn.Model.compile(model, loss: :binary_cross_entropy)
      # pred close to 0, target = 1 → high loss
      pred = Nx.tensor([[0.01]])
      target = Nx.tensor([[1.0]])

      {:ok, loss} = ExBurn.Model.compute_loss(compiled, pred, target)
      assert Nx.to_number(loss) > 1.0
    end
  end

  describe "forward/2 edge cases" do
    test "returns error for uncompiled model" do
      model = ExBurn.Model.new()
      assert {:error, "Model not compiled" <> _} = ExBurn.Model.forward(model, Nx.tensor([1.0]))
    end

    test "predict returns error for uncompiled model" do
      model = ExBurn.Model.new()
      assert {:error, "Model not compiled" <> _} = ExBurn.Model.predict(model, Nx.tensor([1.0]))
    end
  end

  describe "freeze/2 and unfreeze/2 edge cases" do
    test "freezing non-existent layer doesn't crash" do
      model =
        Axon.input("input", shape: {nil, 10})
        |> Axon.dense(5, name: "layer1")
        |> axon_model()

      compiled = ExBurn.Model.compile(model)
      frozen = ExBurn.Model.freeze(compiled, ["nonexistent_layer"])
      assert ExBurn.Model.frozen_layers(frozen) == MapSet.new(["nonexistent_layer"])
    end

    test "freezing same layer twice is idempotent" do
      model =
        Axon.input("input", shape: {nil, 10})
        |> Axon.dense(5, name: "layer1")
        |> axon_model()

      compiled = ExBurn.Model.compile(model)
      frozen1 = ExBurn.Model.freeze(compiled, ["layer1"])
      frozen2 = ExBurn.Model.freeze(frozen1, ["layer1"])
      assert ExBurn.Model.frozen_layers(frozen1) == ExBurn.Model.frozen_layers(frozen2)
    end

    test "unfreezing non-frozen layer is a no-op" do
      model =
        Axon.input("input", shape: {nil, 10})
        |> Axon.dense(5, name: "layer1")
        |> axon_model()

      compiled = ExBurn.Model.compile(model)
      unfrozen = ExBurn.Model.unfreeze(compiled, ["layer1"])
      assert ExBurn.Model.frozen_layers(unfrozen) == MapSet.new()
    end

    test "freeze accepts atom layer names" do
      model =
        Axon.input("input", shape: {nil, 10})
        |> Axon.dense(5, name: "layer1")
        |> axon_model()

      compiled = ExBurn.Model.compile(model)
      frozen = ExBurn.Model.freeze(compiled, [:layer1])
      assert ExBurn.Model.frozen?(frozen, :layer1)
    end

    test "frozen? with atom layer name" do
      model =
        Axon.input("input", shape: {nil, 10})
        |> Axon.dense(5, name: "layer1")
        |> axon_model()

      compiled = ExBurn.Model.compile(model)
      frozen = ExBurn.Model.freeze(compiled, ["layer1"])
      assert ExBurn.Model.frozen?(frozen, :layer1)
    end
  end

  describe "to_device/2 edge cases" do
    test "to_cpu from cpu is a no-op" do
      model =
        Axon.input("input", shape: {nil, 10})
        |> Axon.dense(5)
        |> axon_model()

      compiled = ExBurn.Model.compile(model, device: :cpu)
      assert ExBurn.Model.to_device(compiled, :cpu) == compiled
    end

    test "to_gpu from gpu is a no-op" do
      model =
        Axon.input("input", shape: {nil, 10})
        |> Axon.dense(5)
        |> axon_model()

      compiled = ExBurn.Model.compile(model, device: :gpu)
      assert ExBurn.Model.to_device(compiled, :gpu) == compiled
    end
  end

  describe "accessor functions" do
    test "parameters/1 returns the params map" do
      model =
        Axon.input("input", shape: {nil, 10})
        |> Axon.dense(5)
        |> axon_model()

      compiled = ExBurn.Model.compile(model)
      params = ExBurn.Model.parameters(compiled)
      assert is_map(params)
      assert map_size(params) > 0
    end

    test "loss_function/1 returns the loss function atom" do
      model =
        Axon.input("input", shape: {nil, 10})
        |> Axon.dense(5)
        |> axon_model()

      compiled = ExBurn.Model.compile(model, loss: :mse)
      assert ExBurn.Model.loss_function(compiled) == :mse
    end

    test "optimizer/1 returns the optimizer atom" do
      model =
        Axon.input("input", shape: {nil, 10})
        |> Axon.dense(5)
        |> axon_model()

      compiled = ExBurn.Model.compile(model, optimizer: :sgd)
      assert ExBurn.Model.optimizer(compiled) == :sgd
    end

    test "weight_decay/1 returns the weight decay value" do
      model =
        Axon.input("input", shape: {nil, 10})
        |> Axon.dense(5)
        |> axon_model()

      compiled = ExBurn.Model.compile(model, weight_decay: 0.001)
      assert ExBurn.Model.weight_decay(compiled) == 0.001
    end
  end

  describe "quantize/2 edge cases" do
    test "quantize to f16" do
      model =
        Axon.input("input", shape: {nil, 5})
        |> Axon.dense(3)
        |> axon_model()

      compiled = ExBurn.Model.compile(model)
      quantized = ExBurn.Model.quantize(compiled, :f16)
      assert quantized.params != nil
      assert map_size(quantized.params) == map_size(compiled.params)
    end

    test "quantize to bf16" do
      model =
        Axon.input("input", shape: {nil, 5})
        |> Axon.dense(3)
        |> axon_model()

      compiled = ExBurn.Model.compile(model)
      quantized = ExBurn.Model.quantize(compiled, :bf16)
      assert quantized.params != nil
    end
  end

  describe "clone/1 edge cases" do
    test "clone preserves all fields" do
      model =
        Axon.input("input", shape: {nil, 5})
        |> Axon.dense(3, name: "layer1")
        |> axon_model()

      compiled = ExBurn.Model.compile(model, loss: :mse, optimizer: :sgd)
      frozen = ExBurn.Model.freeze(compiled, ["layer1"])
      cloned = ExBurn.Model.clone(frozen)

      assert cloned.loss_fn == :mse
      assert cloned.optimizer == :sgd
      assert cloned.device == compiled.device
      assert cloned.compiled == true
      assert ExBurn.Model.frozen_layers(cloned) == ExBurn.Model.frozen_layers(frozen)
    end
  end

  describe "info/1 edge cases" do
    test "info returns correct structure" do
      model =
        Axon.input("input", shape: {nil, 10})
        |> Axon.dense(5)
        |> axon_model()

      compiled = ExBurn.Model.compile(model)
      info = ExBurn.Model.info(compiled)

      assert is_map(info)
      assert is_integer(info.total_params)
      assert info.total_params > 0
      assert is_integer(info.layer_count)
      assert info.layer_count > 0
      assert info.compiled == true
      assert is_float(info.estimated_memory_mb)
      assert info.estimated_memory_mb >= 0
    end

    test "info with frozen layers" do
      model =
        Axon.input("input", shape: {nil, 10})
        |> Axon.dense(5, name: "layer1")
        |> axon_model()

      compiled = ExBurn.Model.compile(model)
      frozen = ExBurn.Model.freeze(compiled, ["layer1"])
      info = ExBurn.Model.info(frozen)

      assert info.frozen_layers_count == 1
    end
  end

  describe "export/2 and import_params/2 edge cases" do
    test "export and import with elixir_terms format" do
      model =
        Axon.input("input", shape: {nil, 5})
        |> Axon.dense(3)
        |> axon_model()

      compiled = ExBurn.Model.compile(model)
      path = Path.join(System.tmp_dir!(), "test_export_#{System.unique_integer()}.etf")

      try do
        assert ExBurn.Model.export(compiled, path, format: :elixir_terms) == :ok
        assert {:ok, imported} = ExBurn.Model.import_params(compiled, path, format: :elixir_terms)
        assert imported.compiled == true
      after
        File.rm(path)
      end
    end

    test "export with elixir_terms format" do
      model =
        Axon.input("input", shape: {nil, 5})
        |> Axon.dense(3)
        |> axon_model()

      compiled = ExBurn.Model.compile(model)
      path = Path.join(System.tmp_dir!(), "test_export_#{System.unique_integer()}.etf")

      try do
        assert ExBurn.Model.export(compiled, path, format: :elixir_terms) == :ok
        assert File.exists?(path)
      after
        File.rm(path)
      end
    end

    test "export returns error for unsupported format" do
      model =
        Axon.input("input", shape: {nil, 5})
        |> Axon.dense(3)
        |> axon_model()

      compiled = ExBurn.Model.compile(model)
      assert {:error, _} = ExBurn.Model.export(compiled, "/tmp/test.bin", format: :protobuf)
    end

    test "import returns error for missing file" do
      model =
        Axon.input("input", shape: {nil, 5})
        |> Axon.dense(3)
        |> axon_model()

      compiled = ExBurn.Model.compile(model)
      assert {:error, _} = ExBurn.Model.import_params(compiled, "/nonexistent/path.etf")
    end
  end

  describe "forward_pattern/1 edge cases" do
    test "returns map with expected keys" do
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
end
