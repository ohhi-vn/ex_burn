defmodule ExBurn.ModelBenchmarkTest do
  use ExUnit.Case

  defp axon_model(input_dim, hidden, output_dim) do
    Axon.input("input", shape: {nil, input_dim})
    |> Axon.dense(hidden, activation: :relu, name: "hidden")
    |> Axon.dense(output_dim, name: "output")
  end

  describe "benchmark/3" do
    @tag :nif
    test "returns a map with timing statistics" do
      model = axon_model(10, 5, 3)
      compiled = ExBurn.Model.compile(model)
      {input, _key} = Nx.Random.uniform(Nx.Random.key(1), -1.0, 1.0, shape: {1, 10})

      result = ExBurn.Model.benchmark(compiled, input, warmup: 2, runs: 5)

      assert is_map(result)
      assert Map.has_key?(result, :avg_ms)
      assert Map.has_key?(result, :min_ms)
      assert Map.has_key?(result, :max_ms)
      assert Map.has_key?(result, :median_ms)
      assert Map.has_key?(result, :std_ms)
      assert Map.has_key?(result, :runs)
      assert Map.has_key?(result, :warmup)
      assert result.runs == 5
      assert result.warmup == 2
      assert result.avg_ms > 0
      assert result.min_ms <= result.avg_ms
      assert result.max_ms >= result.avg_ms
    end

    @tag :nif
    test "uses default warmup and runs" do
      model = axon_model(5, 3, 2)
      compiled = ExBurn.Model.compile(model)
      {input, _key} = Nx.Random.uniform(Nx.Random.key(1), -1.0, 1.0, shape: {1, 5})

      result = ExBurn.Model.benchmark(compiled, input)

      assert result.warmup == 3
      assert result.runs == 10
    end

    @tag :nif
    test "median is between min and max" do
      model = axon_model(10, 5, 3)
      compiled = ExBurn.Model.compile(model)
      {input, _key} = Nx.Random.uniform(Nx.Random.key(1), -1.0, 1.0, shape: {1, 10})

      result = ExBurn.Model.benchmark(compiled, input, warmup: 2, runs: 10)

      assert result.median_ms >= result.min_ms
      assert result.median_ms <= result.max_ms
    end
  end

  describe "export/2 and import_params/2 with JSON" do
    @tag :nif
    test "round-trips model parameters through JSON" do
      model = axon_model(5, 3, 2)
      compiled = ExBurn.Model.compile(model)
      path = Path.join(System.tmp_dir!(), "test_json_export_#{System.unique_integer()}.json")

      try do
        assert ExBurn.Model.export(compiled, path, format: :json) == :ok
        assert File.exists?(path)

        {:ok, imported} = ExBurn.Model.import_params(compiled, path, format: :json)
        assert imported.compiled == true
        assert is_map(imported.params)
      after
        File.rm(path)
      end
    end

    @tag :nif
    test "JSON export produces valid JSON file" do
      model = axon_model(3, 2, 1)
      compiled = ExBurn.Model.compile(model)
      path = Path.join(System.tmp_dir!(), "test_json_#{System.unique_integer()}.json")

      try do
        ExBurn.Model.export(compiled, path, format: :json)

        {:ok, content} = File.read(path)
        assert is_binary(content)

        # Should be valid JSON
        assert {:ok, _} = Jason.decode(content)
      after
        File.rm(path)
      end
    end
  end
end
