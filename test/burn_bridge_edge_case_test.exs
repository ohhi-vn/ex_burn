defmodule ExBurn.BurnBridgeEdgeCaseTest do
  use ExUnit.Case

  describe "tensor creation edge cases" do
    @tag :nif
    test "zeros with various shapes" do
      t1 = ExBurn.BurnBridge.zeros([1], :f32)
      assert ExBurn.Tensor.shape(t1) == [1]

      t2 = ExBurn.BurnBridge.zeros([100], :f32)
      assert ExBurn.Tensor.shape(t2) == [100]

      t3 = ExBurn.BurnBridge.zeros([3, 4, 5], :f32)
      assert ExBurn.Tensor.shape(t3) == [3, 4, 5]
    end

    @tag :nif
    test "ones with various shapes" do
      t1 = ExBurn.BurnBridge.ones([1], :f32)
      assert ExBurn.Tensor.shape(t1) == [1]

      t2 = ExBurn.BurnBridge.ones([10, 10], :f32)
      assert ExBurn.Tensor.shape(t2) == [10, 10]
    end

    @tag :nif
    test "rand returns tensor with correct shape" do
      t = ExBurn.BurnBridge.rand([3, 4], :f32)
      assert ExBurn.Tensor.shape(t) == [3, 4]
      assert ExBurn.Tensor.type(t) == :f32
    end

    @tag :nif
    test "rand with default type" do
      t = ExBurn.BurnBridge.rand([5])
      assert ExBurn.Tensor.shape(t) == [5]
      assert ExBurn.Tensor.type(t) == :f32
    end
  end

  describe "arithmetic edge cases" do
    @tag :nif
    test "add with negative values" do
      a = ExBurn.BurnBridge.from_nx(Nx.tensor([-1.0, -2.0, -3.0]))
      b = ExBurn.BurnBridge.from_nx(Nx.tensor([1.0, 2.0, 3.0]))
      c = ExBurn.BurnBridge.add(a, b)
      assert ExBurn.Tensor.shape(c) == [3]
    end

    @tag :nif
    test "sub with zero" do
      a = ExBurn.BurnBridge.from_nx(Nx.tensor([1.0, 2.0, 3.0]))
      b = ExBurn.BurnBridge.zeros([3], :f32)
      c = ExBurn.BurnBridge.sub(a, b)
      assert ExBurn.Tensor.shape(c) == [3]
    end

    @tag :nif
    test "mul with zero" do
      a = ExBurn.BurnBridge.from_nx(Nx.tensor([1.0, 2.0, 3.0]))
      b = ExBurn.BurnBridge.zeros([3], :f32)
      c = ExBurn.BurnBridge.mul(a, b)
      assert ExBurn.Tensor.shape(c) == [3]
    end

    @tag :nif
    test "div with ones" do
      a = ExBurn.BurnBridge.from_nx(Nx.tensor([1.0, 2.0, 3.0]))
      b = ExBurn.BurnBridge.ones([3], :f32)
      c = ExBurn.BurnBridge.div(a, b)
      assert ExBurn.Tensor.shape(c) == [3]
    end

    @tag :nif
    test "neg of zeros" do
      a = ExBurn.BurnBridge.zeros([3], :f32)
      c = ExBurn.BurnBridge.neg(a)
      assert ExBurn.Tensor.shape(c) == [3]
    end

    @tag :nif
    test "abs of negative values" do
      a = ExBurn.BurnBridge.from_nx(Nx.tensor([-1.0, -2.0, -3.0]))
      c = ExBurn.BurnBridge.abs(a)
      assert ExBurn.Tensor.shape(c) == [3]
    end

    @tag :nif
    test "exp of zeros" do
      a = ExBurn.BurnBridge.zeros([3], :f32)
      c = ExBurn.BurnBridge.exp(a)
      assert ExBurn.Tensor.shape(c) == [3]
    end

    @tag :nif
    test "log of ones" do
      a = ExBurn.BurnBridge.ones([3], :f32)
      c = ExBurn.BurnBridge.log(a)
      assert ExBurn.Tensor.shape(c) == [3]
    end

    @tag :nif
    test "sqrt of ones" do
      a = ExBurn.BurnBridge.ones([3], :f32)
      c = ExBurn.BurnBridge.sqrt(a)
      assert ExBurn.Tensor.shape(c) == [3]
    end

    @tag :nif
    test "sigmoid of zeros" do
      a = ExBurn.BurnBridge.zeros([3], :f32)
      c = ExBurn.BurnBridge.sigmoid(a)
      assert ExBurn.Tensor.shape(c) == [3]
    end

    @tag :nif
    test "relu of mixed values" do
      a = ExBurn.BurnBridge.from_nx(Nx.tensor([-1.0, 0.0, 1.0]))
      c = ExBurn.BurnBridge.relu(a)
      assert ExBurn.Tensor.shape(c) == [3]
    end
  end

  describe "linear algebra edge cases" do
    @tag :nif
    test "matmul with identity-like matrices" do
      a = ExBurn.BurnBridge.from_nx(Nx.tensor([[1.0, 0.0], [0.0, 1.0]]))
      b = ExBurn.BurnBridge.from_nx(Nx.tensor([[1.0, 2.0], [3.0, 4.0]]))
      c = ExBurn.BurnBridge.matmul(a, b)
      assert ExBurn.Tensor.shape(c) == [2, 2]
    end

    @tag :nif
    test "matmul with non-square matrices" do
      a = ExBurn.BurnBridge.from_nx(Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
      b = ExBurn.BurnBridge.from_nx(Nx.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
      c = ExBurn.BurnBridge.matmul(a, b)
      assert ExBurn.Tensor.shape(c) == [2, 2]
    end

    @tag :nif
    test "transpose of 2D tensor" do
      a = ExBurn.BurnBridge.from_nx(Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
      t = ExBurn.BurnBridge.transpose(a)
      assert ExBurn.Tensor.shape(t) == [3, 2]
    end

    @tag :nif
    test "transpose with custom dims" do
      a = ExBurn.BurnBridge.from_nx(Nx.tensor([[1.0, 2.0], [3.0, 4.0]]))
      t = ExBurn.BurnBridge.transpose(a, 0, 1)
      assert ExBurn.Tensor.shape(t) == [2, 2]
    end
  end

  describe "reduction edge cases" do
    @tag :nif
    test "sum of ones" do
      a = ExBurn.BurnBridge.ones([5], :f32)
      s = ExBurn.BurnBridge.sum(a)
      assert ExBurn.Tensor.shape(s) == [1]
    end

    @tag :nif
    test "mean of constant tensor" do
      a = ExBurn.BurnBridge.from_nx(Nx.tensor([3.0, 3.0, 3.0]))
      m = ExBurn.BurnBridge.mean(a)
      assert ExBurn.Tensor.shape(m) == [1]
    end
  end

  describe "shape manipulation edge cases" do
    @tag :nif
    test "reshape 1D to 2D" do
      a = ExBurn.BurnBridge.ones([6], :f32)
      r = ExBurn.BurnBridge.reshape(a, [2, 3])
      assert ExBurn.Tensor.shape(r) == [2, 3]
    end

    @tag :nif
    test "reshape 2D to 1D" do
      a = ExBurn.BurnBridge.ones([2, 3], :f32)
      r = ExBurn.BurnBridge.reshape(a, [6])
      assert ExBurn.Tensor.shape(r) == [6]
    end

    @tag :nif
    test "softmax" do
      a = ExBurn.BurnBridge.from_nx(Nx.tensor([1.0, 2.0, 3.0]))
      s = ExBurn.BurnBridge.softmax(a)
      assert ExBurn.Tensor.shape(s) == [3]
    end

    @tag :nif
    test "dropout returns same tensor" do
      a = ExBurn.BurnBridge.ones([3], :f32)
      d = ExBurn.BurnBridge.dropout(a, 0.5, true)
      assert ExBurn.Tensor.shape(d) == [3]
    end
  end

  describe "loss function edge cases" do
    @tag :nif
    test "cross_entropy returns tensor" do
      pred = ExBurn.BurnBridge.from_nx(Nx.tensor([1.0, 2.0, 3.0]))
      target = ExBurn.BurnBridge.from_nx(Nx.tensor([0.0, 0.0, 1.0]))
      loss = ExBurn.BurnBridge.cross_entropy(pred, target)
      assert ExBurn.Tensor.shape(loss) == [1]
    end

    @tag :nif
    test "mse returns tensor" do
      pred = ExBurn.BurnBridge.from_nx(Nx.tensor([1.0, 2.0, 3.0]))
      target = ExBurn.BurnBridge.from_nx(Nx.tensor([1.0, 2.0, 3.0]))
      loss = ExBurn.BurnBridge.mse(pred, target)
      assert ExBurn.Tensor.shape(loss) == [1]
    end
  end

  describe "device management edge cases" do
    @tag :nif
    test "gpu_available? returns boolean" do
      assert is_boolean(ExBurn.BurnBridge.gpu_available?())
    end

    @tag :nif
    test "device_name returns string" do
      assert is_binary(ExBurn.BurnBridge.device_name())
    end

    @tag :nif
    test "device_info returns map with expected keys" do
      info = ExBurn.BurnBridge.device_info()
      assert is_map(info)
      assert Map.has_key?(info, :device)
      assert Map.has_key?(info, :gpu_available)
      assert Map.has_key?(info, :backend)
      assert Map.has_key?(info, :available_backends)
    end

    @tag :nif
    test "to_gpu returns tensor with same shape" do
      a = ExBurn.BurnBridge.ones([3, 4], :f32)
      g = ExBurn.BurnBridge.to_gpu(a)
      assert ExBurn.Tensor.shape(g) == [3, 4]
    end

    @tag :nif
    test "to_cpu returns tensor with same shape" do
      a = ExBurn.BurnBridge.ones([3, 4], :f32)
      c = ExBurn.BurnBridge.to_cpu(a)
      assert ExBurn.Tensor.shape(c) == [3, 4]
    end
  end

  describe "memory management" do
    @tag :nif
    test "free returns ok" do
      a = ExBurn.BurnBridge.ones([3], :f32)
      assert ExBurn.BurnBridge.free(a) == :ok
    end
  end

  describe "Nx ↔ Burn conversion edge cases" do
    @tag :nif
    test "from_nx with 2D tensor" do
      nx = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      bt = ExBurn.BurnBridge.from_nx(nx)
      assert ExBurn.Tensor.shape(bt) == [2, 2]
    end

    @tag :nif
    test "to_nx round-trip with 2D tensor" do
      nx = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      bt = ExBurn.BurnBridge.from_nx(nx)
      back = ExBurn.BurnBridge.to_nx(bt)
      assert Nx.shape(back) == {2, 2}
    end

    @tag :nif
    test "from_nx with single element" do
      nx = Nx.tensor([42.0])
      bt = ExBurn.BurnBridge.from_nx(nx)
      assert ExBurn.Tensor.shape(bt) == [1]
    end
  end
end
