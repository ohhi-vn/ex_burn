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

  describe "configure!" do
    test "sets ExBurn as the default backend" do
      previous = Nx.default_backend()

      try do
        ExBurn.configure!()
        assert Nx.default_backend() == ExBurn.Backend
      after
        Nx.default_backend(previous)
      end
    end
  end

  describe "Nx backend" do
    setup do
      previous = Nx.default_backend()
      Nx.default_backend(ExBurn.Backend)
      on_exit(fn -> Nx.default_backend(previous) end)
      :ok
    end

    test "tensor creation from binary" do
      t = Nx.tensor([1.0, 2.0, 3.0])
      assert Nx.to_list(t) == [1.0, 2.0, 3.0]
    end

    test "addition" do
      a = Nx.tensor([1.0, 2.0, 3.0])
      b = Nx.tensor([4.0, 5.0, 6.0])
      c = Nx.add(a, b)
      assert Nx.to_list(c) == [5.0, 7.0, 9.0]
    end

    test "subtraction" do
      a = Nx.tensor([4.0, 5.0, 6.0])
      b = Nx.tensor([1.0, 2.0, 3.0])
      c = Nx.subtract(a, b)
      assert Nx.to_list(c) == [3.0, 3.0, 3.0]
    end

    test "multiplication" do
      a = Nx.tensor([1.0, 2.0, 3.0])
      b = Nx.tensor([4.0, 5.0, 6.0])
      c = Nx.multiply(a, b)
      assert Nx.to_list(c) == [4.0, 10.0, 18.0]
    end

    test "division" do
      a = Nx.tensor([4.0, 10.0, 18.0])
      b = Nx.tensor([2.0, 5.0, 6.0])
      c = Nx.divide(a, b)
      assert Nx.to_list(c) == [2.0, 2.0, 3.0]
    end

    test "negation" do
      a = Nx.tensor([1.0, -2.0, 3.0])
      c = Nx.negate(a)
      assert Nx.to_list(c) == [-1.0, 2.0, -3.0]
    end

    test "absolute value" do
      a = Nx.tensor([-1.0, 2.0, -3.0])
      c = Nx.abs(a)
      assert Nx.to_list(c) == [1.0, 2.0, 3.0]
    end

    test "2D tensor operations" do
      a = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      b = Nx.tensor([[5.0, 6.0], [7.0, 8.0]])
      c = Nx.add(a, b)
      assert Nx.to_list(c) == [6.0, 8.0, 10.0, 12.0]
    end

    test "transpose" do
      a = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
      c = Nx.transpose(a)
      assert Nx.shape(c) == {3, 2}
    end

    test "reshape" do
      a = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
      c = Nx.reshape(a, {2, 3})
      assert Nx.shape(c) == {2, 3}
    end

    test "sum reduction" do
      a = Nx.tensor([1.0, 2.0, 3.0])
      c = Nx.sum(a)
      assert Nx.to_list(c) == [6.0]
    end

    test "mean reduction" do
      a = Nx.tensor([1.0, 2.0, 3.0, 4.0])
      c = Nx.mean(a)
      assert_in_delta Nx.to_list(c) |> hd(), 2.5, 1.0e-6
    end
  end

  describe "ExBurn.Error" do
    test "formats message without details" do
      error = ExBurn.Error.exception(op: :test, reason: "something failed")
      assert Exception.message(error) == "ExBurn.test: something failed"
    end

    test "formats message with details" do
      error = ExBurn.Error.exception(op: :matmul, reason: "shape mismatch", details: %{lhs: [3, 4], rhs: [5, 6]})
      msg = Exception.message(error)
      assert msg =~ "ExBurn.matmul: shape mismatch"
      assert msg =~ "[3, 4]"
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

    test "arithmetic operations" do
      a = ExBurn.BurnBridge.from_nx(Nx.tensor([1.0, 2.0, 3.0]))
      b = ExBurn.BurnBridge.from_nx(Nx.tensor([4.0, 5.0, 6.0]))

      c = ExBurn.BurnBridge.add(a, b)
      assert ExBurn.Tensor.shape(c) == [3]

      d = ExBurn.BurnBridge.mul(a, b)
      assert ExBurn.Tensor.shape(d) == [3]
    end
  end

  describe "Tensor utilities" do
    test "type conversion round-trip" do
      assert ExBurn.Tensor.nx_type_to_burn({:f, 32}) == :f32
      assert ExBurn.Tensor.burn_type_to_nx(:f32) == {:f, 32}
    end

    test "numel and rank" do
      t = ExBurn.BurnBridge.zeros([2, 3, 4], :f32)
      assert ExBurn.Tensor.numel(t) == 24
      assert ExBurn.Tensor.rank(t) == 3
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
  end

  describe "CubeclBridge" do
    test "returns available backends" do
      backends = ExBurn.CubeclBridge.available_backends()
      assert is_list(backends)
    end
  end
end
