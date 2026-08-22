defmodule ExBurn.ModelCoverageTest do
  use ExUnit.Case, async: false

  defp graph do
    Axon.input("input", shape: {nil, 4}) |> Axon.dense(2)
  end

  defp compiled(opts \\ []) do
    ExBurn.Model.compile(graph(), Keyword.merge([loss: :mse], opts))
  end

  defp sample, do: Nx.tensor([1.0, 2.0, 3.0, 4.0])

  describe "forward/2" do
    @tag :nif
    test "runs a happy-path forward pass with a real graph" do
      model = compiled()

      assert {:ok, out} = ExBurn.Model.forward(model, Nx.reshape(sample(), {1, 4}))
      assert Nx.shape(out) == {1, 2}
    end

    @tag :nif
    test "returns an error for uncompiled models" do
      uncompiled = ExBurn.Model.new()
      assert {:error, msg} = ExBurn.Model.forward(uncompiled, sample())
      assert msg =~ "not compiled"
    end
  end

  describe "predict/2" do
    @tag :nif
    test "errors clearly when no Axon graph was provided" do
      # Compiling from a ModelState leaves predict_fn nil
      {init_fn, _} = Axon.build(graph())
      model_state = init_fn.(Nx.template({1, 4}, :f32), Axon.ModelState.empty())
      model = ExBurn.Model.compile(model_state)

      assert {:error, msg} = ExBurn.Model.predict(model, Nx.reshape(sample(), {1, 4}))
      assert msg =~ "no Axon graph"
    end

    @tag :nif
    test "predicts through the cached predict function" do
      model = compiled()
      assert {:ok, out} = ExBurn.Model.predict(model, Nx.reshape(sample(), {1, 4}))
      assert Nx.shape(out) == {1, 2}
    end
  end

  describe "to_device/2" do
    @tag :nif
    test "moves params to gpu structs and back to nx tensors" do
      cpu_model = compiled(device: :cpu)
      assert cpu_model.device == :cpu

      gpu_model = ExBurn.Model.to_device(cpu_model, :gpu)
      assert gpu_model.device == :gpu
      # GPU params are burn tensor structs, not Nx tensors
      assert Enum.all?(gpu_model.params, fn {_k, v} -> match?(%ExBurn.Tensor{}, v) end)

      back = ExBurn.Model.to_device(gpu_model, :cpu)
      assert back.device == :cpu
      assert Enum.all?(back.params, fn {_k, v} -> match?(%Nx.Tensor{}, v) end)
    end
  end

  describe "forward_pattern/1" do
    @tag :nif
    test "falls back to unknown output for models without a graph" do
      {init_fn, _} = Axon.build(graph())
      model_state = init_fn.(Nx.template({1, 4}, :f32), Axon.ModelState.empty())
      model = ExBurn.Model.compile(model_state)

      assert %{output_shape: nil, output_type: :f32} = ExBurn.Model.forward_pattern(model)
    end
  end

  describe "summary/1" do
    @tag :nif
    test "marks frozen layers in the table" do
      model =
        graph()
        |> Axon.dense(2, name: "layer1")
        |> Axon.dense(3, name: "layer2")
        |> ExBurn.Model.compile()

      summary = ExBurn.Model.summary(ExBurn.Model.freeze(model, ["layer1"]))
      assert summary =~ "[FROZEN]"
      assert summary =~ "Trainable params:"
      assert summary =~ "Non-trainable params:"
    end

    @tag :nif
    test "guesses layer families from parameter names" do
      model = compiled()
      t = Nx.tensor([[1.0]])

      crafted =
        ExBurn.Model.update_params(model, %{
          "conv.weight" => t,
          "lstm.weight" => t,
          "gru.weight" => t,
          "embed.weight" => t,
          "norm.weight" => t,
          "dropout.weight" => t,
          "attention.weight" => t,
          "mysterious" => t,
          :some_atom_key => t,
          42 => t
        })

      summary = ExBurn.Model.summary(crafted)
      assert summary =~ "(conv)"
      assert summary =~ "(lstm)"
      assert summary =~ "(gru)"
      assert summary =~ "(embedding)"
      assert summary =~ "(normalization)"
      assert summary =~ "(dropout)"
      assert summary =~ "(attention)"
      assert summary =~ "(unknown)"
      # top-level bucket catches keys without a "layer.param" split
      assert summary =~ "_top_level"
    end
  end

  describe "export/JSON round trip" do
    @tag :nif
    test "exports json and imports it back preserving shapes" do
      model = compiled()
      path = Path.join(System.tmp_dir!(), "exburn_model_#{System.unique_integer()}.json")

      try do
        assert :ok = ExBurn.Model.export(model, path, format: :json)
        assert {:ok, imported} = ExBurn.Model.import_params(model, path, format: :json)
        assert imported.compiled

        Enum.each(model.params, fn {key, tensor} ->
          assert Nx.shape(imported.params[key]) == Nx.shape(tensor)
        end)
      after
        File.rm(path)
      end
    end

    @tag :nif
    test "import_params raises structured error on invalid json" do
      path = Path.join(System.tmp_dir!(), "exburn_bad_#{System.unique_integer()}.json")
      File.write!(path, "{definitely not json")

      try do
        assert_raise ExBurn.Error, ~r/Invalid JSON/, fn ->
          ExBurn.Model.import_params(compiled(), path, format: :json)
        end
      after
        File.rm(path)
      end
    end
  end

  describe "benchmark/3" do
    @tag :nif
    test "times forward-only runs" do
      stats = ExBurn.Model.benchmark(compiled(), Nx.reshape(sample(), {1, 4}), warmup: 0, runs: 2)
      assert %{avg_ms: avg, runs: 2, warmup: 0} = stats
      assert is_float(avg)
    end

    @tag :nif
    test "times forward+loss runs when given targets" do
      input = Nx.reshape(sample(), {1, 4})
      target = Nx.tensor([[0.0, 0.0]])

      stats = ExBurn.Model.benchmark(compiled(), {input, target}, warmup: 0, runs: 2)
      assert %{avg_ms: _, min_ms: _, max_ms: _, median_ms: _, std_ms: _} = stats
    end
  end

  describe "clone/1" do
    @tag :nif
    test "produces independent parameter storage" do
      original = compiled()
      clone = ExBurn.Model.clone(original)

      key = hd(Map.keys(original.params))

      mutated =
        Map.put(clone.params, key, Nx.broadcast(Nx.tensor(99.0), Nx.shape(original.params[key])))

      refute ExBurn.Model.update_params(clone, mutated).params == original.params
    end
  end

  describe "info/1" do
    @tag :nif
    test "includes optimizer learning rate" do
      info = ExBurn.Model.info(compiled(learning_rate: 0.05))
      assert info.learning_rate == 0.05
      assert is_float(info.estimated_memory_mb)
    end
  end

  describe "quantize/2" do
    @tag :nif
    test "downcasts parameter tensors" do
      quantized = ExBurn.Model.quantize(compiled(), :f16)

      assert Enum.all?(quantized.params, fn {_k, v} ->
               match?(%Nx.Tensor{type: {:f, 16}}, v) or not match?(%Nx.Tensor{}, v)
             end)
    end
  end
end
