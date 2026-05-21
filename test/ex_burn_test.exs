defmodule ExBurnTest do
  use ExUnit.Case
  doctest ExBurn

  describe "version" do
    test "returns a version string" do
      assert is_binary(ExBurn.version())
    end
  end

  describe "default_device" do
    test "returns :cpu or :gpu" do
      device = ExBurn.default_device()
      assert device in [:cpu, :gpu]
    end
  end

  describe "Nx backend" do
    test "can set ExBurn as default backend" do
      # Save previous backend
      previous = Nx.default_backend()

      try do
        ExBurn.configure!()
        assert Nx.default_backend() == ExBurn.Backend
      after
        Nx.default_backend(previous)
      end
    end

    test "tensor operations work through backend" do
      previous = Nx.default_backend()

      try do
        Nx.default_backend(ExBurn.Backend)

        # Basic tensor creation and operations
        a = Nx.tensor([1.0, 2.0, 3.0])
        b = Nx.tensor([4.0, 5.0, 6.0])
        c = Nx.add(a, b)
        assert Nx.to_list(c) == [5.0, 7.0, 9.0]

        d = Nx.multiply(a, b)
        assert Nx.to_list(d) == [4.0, 10.0, 18.0]
      after
        Nx.default_backend(previous)
      end
    end
  end

  describe "BurnBridge" do
    test "creates zeros and ones tensors" do
      z = ExBurn.BurnBridge.zeros([2, 3], :f32)
      assert ExBurn.Tensor.shape(z) == [2, 3]

      o = ExBurn.BurnBridge.ones([3], :f32)
      assert ExBurn.Tensor.shape(o) == [3]
    end

    test "converts between Nx and Burn tensors" do
      nx = Nx.tensor([1.0, 2.0, 3.0])
      bt = ExBurn.BurnBridge.from_nx(nx)
      assert ExBurn.Tensor.shape(bt) == [3]

      back = ExBurn.BurnBridge.to_nx(bt)
      assert Nx.to_list(back) == [1.0, 2.0, 3.0]
    end
  end

  describe "Model" do
    test "compiles an Axon model" do
      model =
        Axon.input("input", shape: {nil, 10})
        |> Axon.dense(5)
        |> Axon.dense(1)

      compiled = ExBurn.Model.compile(model, loss: :mse, optimizer: :adam)
      assert compiled.compiled == true
      assert is_map(compiled.params)
    end
  end

  describe "DalaML" do
    test "returns compute config for iOS" do
      config = ExBurn.DalaML.compute_config(:ios)
      assert config.backend == :metal
      assert config.preferred_precision == :f16
    end

    test "returns compute config for Android" do
      config = ExBurn.DalaML.compute_config(:android)
      assert config.backend == :vulkan
      assert config.preferred_precision == :f16
    end

    test "returns available backends" do
      backends = ExBurn.CubeclBridge.available_backends()
      assert is_list(backends)
    end
  end
end
