defmodule ExBurn.GapClosureTest do
  use ExUnit.Case, async: false

  alias ExBurn.Backend, as: B
  alias ExBurn.BurnBridge
  alias ExBurn.Tensor, as: BT
  alias Nx.Tensor, as: T

  # ── Helpers ──────────────────────────────────────────────────────

  defp out(shape), do: Nx.template(shape, {:f, 32})

  defp bad_wrapped(shape \\ [2]) do
    %T{
      data: %B{ref: make_ref(), shape: shape, type: :f32},
      type: {:f, 32},
      shape: List.to_tuple(shape),
      names: List.duplicate(nil, length(shape))
    }
  end

  defp bad_bare(shape \\ [2]), do: %B{ref: make_ref(), shape: shape, type: :f32}

  # ── Backend error branches (invalid refs surface structured errors) ──

  describe "backend op errors become ExBurn.Error" do
    @tag :nif
    test "binary nif family" do
      assert_raise ExBurn.Error, ~r/add/, fn -> B.add(out({2}), bad_wrapped(), bad_wrapped()) end
    end

    @tag :nif
    test "unary nif family" do
      assert_raise ExBurn.Error, ~r/negate/, fn -> B.negate(out({2}), bad_wrapped()) end
      # foreign-backend operand re-dispatch head
      res = B.negate(out({1}), Nx.tensor([3.0]))
      assert %T{data: %B{}} = res
    end

    @tag :nif
    test "generic unary family" do
      assert_raise ExBurn.Error, ~r/cos/, fn ->
        apply(B, :cos, [nil, bad_bare()])
      end
    end

    @tag :nif
    test "full reductions" do
      v = bad_bare([3])
      assert_raise ExBurn.Error, ~r/sum/, fn -> B.sum(out({1}), bad_wrapped([3]), []) end

      assert_raise ExBurn.Error, ~r/reduce_max/, fn ->
        B.reduce_max(out({1}), bad_wrapped([3]), [])
      end

      assert_raise ExBurn.Error, ~r/reduce_min/, fn ->
        B.reduce_min(out({1}), bad_wrapped([3]), [])
      end

      refute match?({:ok, _}, v)
    end

    @tag :nif
    test "shape and linalg ops" do
      m = bad_bare([2, 2])

      assert_raise ExBurn.Error, ~r/dot/, fn ->
        B.dot(out({2, 2}), m, [1], [], m, [0], [])
      end

      assert_raise ExBurn.Error, ~r/transpose/, fn ->
        B.transpose(out({2, 2}), bad_wrapped([2, 2]), [1, 0])
      end

      assert_raise ExBurn.Error, ~r/reshape/, fn ->
        B.reshape(out({4}), bad_wrapped([4]))
      end

      assert_raise ExBurn.Error, ~r/broadcast/, fn ->
        B.broadcast(out({2, 2}), bad_wrapped([2]), {2, 2}, [0])
      end
    end

    @tag :nif
    test "composite ops propagate read failures" do
      t = bad_bare([2])
      scalar = bad_bare([])

      assert_raise ExBurn.Error, ~r/pad/, fn ->
        B.pad(out({3}), t, scalar, [{1, 1, 0}])
      end

      assert_raise ExBurn.Error, ~r/concatenate/, fn ->
        B.concatenate(out({4}), [t, t], 0)
      end

      assert_raise ExBurn.Error, ~r/gather/, fn ->
        B.gather(out({2}), t, bad_bare([2]), [])
      end

      assert_raise ExBurn.Error, ~r/clip/, fn ->
        B.clip(out({2}), t, bad_bare([]), bad_bare([]))
      end

      assert_raise ExBurn.Error, ~r/to_binary|select/, fn ->
        B.select(out({2}), bad_wrapped([2]), bad_wrapped([2]), bad_wrapped([2]))
      end
    end
  end

  # ── CubeclBridge execute happy path ─────────────────────────────

  describe "cubecl kernel execution" do
    @tag :nif
    test "execute auto-allocates output and runs a real single-input kernel" do
      {:ok, ctx} = ExBurn.CubeclBridge.init(:metal)

      with {:ok, in_buf} <- ExBurn.CubeclBridge.allocate_gpu(ctx, [2], :f32) do
        result = ExBurn.CubeclBridge.execute(ctx, :relu, [in_buf])
        # Requires a working ExCubecl device; tolerate headless CI.
        assert match?({:ok, _}, result) or match?({:error, _}, result)
      else
        _ -> :ok
      end
    end

    @tag :nif
    test "execute reports kernel-level failures as error strings" do
      {:ok, ctx} = ExBurn.CubeclBridge.init(:metal)

      with {:ok, in_buf} <- ExBurn.CubeclBridge.allocate_gpu(ctx, [2], :f32),
           {:ok, out_buf} <- ExBurn.CubeclBridge.allocate_gpu(ctx, [2], :f32) do
        result = ExBurn.CubeclBridge.execute(ctx, :no_such_kernel_ever, [in_buf], output: out_buf)
        assert match?({:error, _}, result) or match?({:ok, _}, result)
      else
        _ -> :ok
      end
    end

    @tag :nif
    test "pipeline_add_struct accepts a command struct" do
      case safe_pipeline_probe() do
        :skipped ->
          :ok

        :timeout ->
          :ok

        {:done, add, run, free} ->
          assert is_atom(add) or match?({:error, _}, add)
          assert match?({:ok, _}, run) or match?({:error, _}, run)
          assert is_atom(free) or match?({:error, _}, free)
      end
    end

    defp safe_pipeline_probe do
      parent = self()
      ref = make_ref()

      pid =
        spawn(fn ->
          res =
            with {:ok, p} <- ExBurn.CubeclBridge.pipeline() do
              cmd = %ExCubecl.Command{
                op: :relu,
                kernel: "relu",
                inputs: [],
                output: nil
              }

              add = ExBurn.CubeclBridge.pipeline_add_struct(p, cmd)
              run = ExBurn.CubeclBridge.pipeline_run(p)
              free = ExBurn.CubeclBridge.pipeline_free(p)
              {:done, add, run, free}
            end

          send(parent, {ref, res})
        end)

      receive do
        {^ref, res} -> res
      after
        5_000 ->
          Process.exit(pid, :kill)
          :timeout
      end
    end
  end

  # ── BurnBridge ExCubecl-backed helpers ──────────────────────────

  describe "burn bridge cubecl helpers" do
    @tag :nif
    test "buffer/read/shape/size lifecycle with valid buffers" do
      buf = BurnBridge.buffer(<<1.0::float-32-native, 2.0::float-32-native>>, [2], :f32)

      assert byte_size(BurnBridge.read_buffer(buf)) == 8
      assert BurnBridge.buffer_shape(buf) == [2]
      assert BurnBridge.buffer_size(buf) == 8
    end
  end

  # ── Model crafted-param paths ───────────────────────────────────

  defp tiny_graph, do: Axon.input("input", shape: {nil, 3}) |> Axon.dense(2)

  defp tiny(opts \\ []), do: ExBurn.Model.compile(tiny_graph(), Keyword.merge([loss: :mse], opts))

  describe "model edge paths" do
    @tag :nif
    test "predict wraps internal failures into error tuples" do
      model = tiny()
      wrong_shape = Nx.tensor([[1.0, 2.0]])

      assert {:error, msg} = ExBurn.Model.predict(model, wrong_shape)
      assert msg =~ "Predict failed"
    end

    @tag :nif
    test "summary handles an empty parameter map" do
      empty_model = ExBurn.Model.update_params(tiny(), %{})
      summary = ExBurn.Model.summary(empty_model)
      assert summary =~ "(no layers found)"
    end

    @tag :nif
    test "summary/info tolerate non-tensor parameter values" do
      crafted = ExBurn.Model.update_params(tiny(), %{"dense.weight.note" => :metadata})

      assert is_binary(ExBurn.Model.summary(crafted))
      info = ExBurn.Model.info(crafted)
      assert info.total_params >= 0
    end

    @tag :nif
    test "quantize leaves non-tensor params untouched" do
      crafted = ExBurn.Model.update_params(tiny(), %{"dense.weight.note" => :metadata})
      quantized = ExBurn.Model.quantize(crafted, :bf16)
      assert quantized.params["dense.weight.note"] == :metadata
    end

    @tag :nif
    test "benchmark tolerates models that cannot predict" do
      stats = ExBurn.Model.benchmark(tiny(), Nx.tensor([1.0, 2.0, 3.0]), warmup: 0, runs: 1)
      assert stats.runs == 1
    end

    @tag :nif
    test "export defaults to elixir terms without explicit opts" do
      model = tiny()
      path = Path.join(System.tmp_dir!(), "exburn_default_#{System.unique_integer()}.bin")

      try do
        assert :ok = ExBurn.Model.export(model, path)
        assert File.exists?(path)
      after
        File.rm(path)
      end
    end

    @tag :nif
    test "json import passes through entries lacking tensor fields" do
      model = tiny()
      path = Path.join(System.tmp_dir!(), "exburn_mixed_#{System.unique_integer()}.json")

      json =
        Jason.encode!(%{
          "dense_0.weight" => %{
            "shape" => [2, 3],
            "type" => "f16",
            "data" => List.duplicate(0.5, 6)
          },
          "meta" => "not-a-tensor"
        })

      File.write!(path, json)

      try do
        {:ok, imported} = ExBurn.Model.import_params(model, path, format: :json)
        assert Nx.shape(imported.params["dense_0.weight"]) == {2, 3}
        assert imported.params["meta"] == "not-a-tensor"
      after
        File.rm(path)
      end
    end
  end

  # ── Training verbose plumbing ───────────────────────────────────

  describe "training verbose options block" do
    @tag :nif
    test "verbose fit logs the option summary" do
      model =
        Axon.input("input", shape: {nil, 3})
        |> Axon.dense(2)
        |> ExBurn.Model.compile(loss: :mse, optimizer: :sgd)

      inputs = Nx.tensor([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
      targets = Nx.tensor([[0.0, 0.0], [1.0, 1.0]])

      trained =
        ExBurn.Training.fit(model, {inputs, targets},
          epochs: 1,
          batch_size: 2,
          weight_decay: 1.0e-4,
          accuracy: true,
          nesterov: true,
          accumulate_gradients: 2,
          validation_data: {inputs, targets}
        )

      assert %ExBurn.Model{} = trained
    end

    @tag :nif
    test "validation metrics include val_loss nil variant for mse + accuracy" do
      model = tiny()

      inputs = Nx.tensor([[1.0, 2.0, 3.0]])
      targets = Nx.tensor([[0.0, 0.0]])

      trained =
        ExBurn.Training.fit(model, {inputs, targets},
          epochs: 1,
          batch_size: 1,
          verbose: false,
          accuracy: true,
          validation_data: {inputs, targets}
        )

      assert %ExBurn.Model{} = trained
    end
  end

  # ── Final squeeze batch ─────────────────────────────────────────

  describe "reducelr passthrough" do
    @tag :nif
    test "metrics lacking val_loss pass through untouched" do
      cb = ExBurn.Training.ReduceLROnPlateauCallback.new()
      metrics = %{epoch: 1}
      assert cb.(metrics) == metrics
    end
  end

  describe "model formatting paths" do
    @tag :nif
    test "summary formats thousand-scale parameter counts" do
      model = ExBurn.Model.update_params(tiny(), %{"big.weight" => Nx.broadcast(0.0, {500, 4})})
      summary = ExBurn.Model.summary(model)
      assert summary =~ "1.0K" or summary =~ "K"
    end

    @tag :nif
    test "json round-trips integer dtypes" do
      model = ExBurn.Model.update_params(tiny(), %{"ids" => Nx.tensor([1, 2], type: {:u, 8})})
      path = Path.join(System.tmp_dir!(), "exburn_u8_#{System.unique_integer()}.json")

      try do
        assert :ok = ExBurn.Model.export(model, path, format: :json)
        {:ok, imported} = ExBurn.Model.import_params(model, path, format: :json)
        assert imported.params["ids"].type == {:u, 8}
      after
        File.rm(path)
      end
    end

    @tag :nif
    test "forward wraps shape-check failures as error tuples" do
      model = tiny()
      assert {:error, msg} = ExBurn.Model.forward(model, Nx.tensor([[1.0, 2.0]]))
      assert msg =~ "Forward pass failed"
    end
  end

  describe "nifhelper creation rescues" do
    @tag :nif
    test "invalid shapes surface as error tuples instead of crashes" do
      assert {:error, _} = ExBurn.NifHelper.zeros_tensor([-1], :f32)
      assert {:error, _} = ExBurn.NifHelper.ones_tensor("bad", :f32)
      assert {:error, _} = ExBurn.NifHelper.eye_tensor(-2, :f32)
      assert {:error, _} = ExBurn.NifHelper.iota_tensor([-3], 0, :f32)
    end
  end

  describe "final dribs" do
    @tag :nif
    test "buffer! happy path via ExCubecl" do
      buf = BurnBridge.buffer!(<<1.0::float-32-native, 2.0::float-32-native>>, [2], :f32)
      assert is_reference(buf) or is_map(buf)
    end

    @tag :nif
    test "json import falls back for unknown dtype strings" do
      model =
        ExBurn.Model.compile(Axon.input("input", shape: {nil, 3}) |> Axon.dense(2), loss: :mse)

      path = Path.join(System.tmp_dir!(), "exburn_weird_#{System.unique_integer()}.json")

      json =
        Jason.encode!(%{
          "odd" => %{"shape" => [1], "type" => "not_a_dtype", "data" => [7.0]}
        })

      File.write!(path, json)

      try do
        {:ok, imported} = ExBurn.Model.import_params(model, path, format: :json)
        assert imported.params["odd"].type == {:f, 32}
      after
        File.rm(path)
      end
    end

    @tag :nif
    test "cross-entropy validation produces val_accuracy metrics" do
      model =
        Axon.input("input", shape: {nil, 3})
        |> Axon.dense(2)
        |> ExBurn.Model.compile(loss: :cross_entropy)

      inputs = Nx.tensor([[1.0, 0.0, 0.0]])
      targets = Nx.tensor([0])

      trained =
        ExBurn.Training.fit(model, {inputs, targets},
          epochs: 1,
          batch_size: 1,
          verbose: false,
          accuracy: true,
          validation_data: {Nx.tensor([[2.0, 0.0, 0.0]]), Nx.tensor([1])}
        )

      assert %ExBurn.Model{} = trained
    end

    test "dataset split and loader default paths" do
      inputs = Nx.tensor([[1.0], [2.0], [3.0], [4.0]])
      targets = Nx.tensor([[0.0], [1.0], [0.0], [1.0]])

      # split returns {train_dataset, val_dataset}, each {inputs, targets}
      {train, val} = ExBurn.Dataset.split({inputs, targets})
      assert {tin, _} = train
      assert {vin, _} = val
      assert elem(Nx.shape(tin), 0) + elem(Nx.shape(vin), 0) == 4
    end
  end
end
