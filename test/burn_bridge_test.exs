defmodule ExBurn.BurnBridgeTest do
  use ExUnit.Case

  describe "tensor creation" do
    @tag :nif
    test "zeros creates tensor with correct shape" do
      t = ExBurn.BurnBridge.zeros([2, 3], :f32)
      assert ExBurn.Tensor.shape(t) == [2, 3]
      assert ExBurn.Tensor.type(t) == :f32
    end

    @tag :nif
    test "ones creates tensor with correct shape" do
      t = ExBurn.BurnBridge.ones([3], :f32)
      assert ExBurn.Tensor.shape(t) == [3]
    end

    @tag :nif
    test "zeros with default type" do
      t = ExBurn.BurnBridge.zeros([4])
      assert ExBurn.Tensor.shape(t) == [4]
      assert ExBurn.Tensor.type(t) == :f32
    end
  end

  describe "Nx ↔ Burn conversion" do
    @tag :nif
    test "from_nx converts Nx tensor to Burn tensor" do
      nx = Nx.tensor([1.0, 2.0, 3.0])
      bt = ExBurn.BurnBridge.from_nx(nx)
      assert ExBurn.Tensor.shape(bt) == [3]
    end

    @tag :nif
    test "to_nx converts Burn tensor back to Nx" do
      nx = Nx.tensor([1.0, 2.0, 3.0])
      bt = ExBurn.BurnBridge.from_nx(nx)
      back = ExBurn.BurnBridge.to_nx(bt)
      assert Nx.to_list(back) == [1.0, 2.0, 3.0]
    end
  end

  describe "loss functions" do
    @tag :nif
    test "cross_entropy returns scalar tensor" do
      pred = ExBurn.BurnBridge.from_nx(Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
      target = ExBurn.BurnBridge.from_nx(Nx.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]))
      loss = ExBurn.BurnBridge.cross_entropy(pred, target)
      assert ExBurn.Tensor.shape(loss) == [1]
    end

    @tag :nif
    test "cross_entropy with 1D tensors" do
      pred = ExBurn.BurnBridge.from_nx(Nx.tensor([1.0, 2.0, 3.0]))
      target = ExBurn.BurnBridge.from_nx(Nx.tensor([0.0, 0.0, 1.0]))
      loss = ExBurn.BurnBridge.cross_entropy(pred, target)
      assert ExBurn.Tensor.shape(loss) == [1]
    end

    @tag :nif
    test "mse returns scalar tensor" do
      pred = ExBurn.BurnBridge.from_nx(Nx.tensor([1.0, 2.0, 3.0]))
      target = ExBurn.BurnBridge.from_nx(Nx.tensor([1.0, 2.0, 3.0]))
      loss = ExBurn.BurnBridge.mse(pred, target)
      assert ExBurn.Tensor.shape(loss) == [1]
    end

    @tag :nif
    test "mse with 2D tensors" do
      pred = ExBurn.BurnBridge.from_nx(Nx.tensor([[1.0, 2.0], [3.0, 4.0]]))
      target = ExBurn.BurnBridge.from_nx(Nx.tensor([[1.0, 2.0], [3.0, 4.0]]))
      loss = ExBurn.BurnBridge.mse(pred, target)
      assert ExBurn.Tensor.shape(loss) == [1]
    end

    @tag :nif
    test "mse with different values returns positive loss" do
      pred = ExBurn.BurnBridge.from_nx(Nx.tensor([1.0, 2.0, 3.0]))
      target = ExBurn.BurnBridge.from_nx(Nx.tensor([4.0, 5.0, 6.0]))
      loss = ExBurn.BurnBridge.mse(pred, target)
      {:ok, nx_loss} = ExBurn.Tensor.to_nx(loss)
      [val] = Nx.to_flat_list(nx_loss)
      assert val > 0.0
    end
  end

  describe "dropout" do
    @tag :nif
    test "dropout during training modifies tensor" do
      t = ExBurn.BurnBridge.from_nx(Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0]))
      result = ExBurn.BurnBridge.dropout(t, 0.5, true)
      assert ExBurn.Tensor.shape(result) == [5]
    end

    @tag :nif
    test "dropout during inference is identity" do
      t = ExBurn.BurnBridge.from_nx(Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0]))
      result = ExBurn.BurnBridge.dropout(t, 0.5, false)
      assert ExBurn.Tensor.shape(result) == [5]
      # When not training, tensor should be unchanged
      {:ok, original} = ExBurn.Tensor.to_nx(t)
      {:ok, output} = ExBurn.Tensor.to_nx(result)
      assert Nx.to_list(original) == Nx.to_list(output)
    end

    @tag :nif
    test "dropout with 2D tensor" do
      t = ExBurn.BurnBridge.from_nx(Nx.tensor([[1.0, 2.0], [3.0, 4.0]]))
      result = ExBurn.BurnBridge.dropout(t, 0.3, true)
      assert ExBurn.Tensor.shape(result) == [2, 2]
    end

    @tag :nif
    test "dropout with zero probability" do
      t = ExBurn.BurnBridge.from_nx(Nx.tensor([1.0, 2.0, 3.0]))
      result = ExBurn.BurnBridge.dropout(t, 0.0, true)
      assert ExBurn.Tensor.shape(result) == [3]
    end
  end

  describe "arithmetic operations" do
    @tag :nif
    test "add returns correct shape" do
      a = ExBurn.BurnBridge.from_nx(Nx.tensor([1.0, 2.0, 3.0]))
      b = ExBurn.BurnBridge.from_nx(Nx.tensor([4.0, 5.0, 6.0]))
      c = ExBurn.BurnBridge.add(a, b)
      assert ExBurn.Tensor.shape(c) == [3]
    end

    @tag :nif
    test "mul returns correct shape" do
      a = ExBurn.BurnBridge.from_nx(Nx.tensor([1.0, 2.0, 3.0]))
      b = ExBurn.BurnBridge.from_nx(Nx.tensor([4.0, 5.0, 6.0]))
      c = ExBurn.BurnBridge.mul(a, b)
      assert ExBurn.Tensor.shape(c) == [3]
    end

    @tag :nif
    test "sub returns correct shape" do
      a = ExBurn.BurnBridge.from_nx(Nx.tensor([4.0, 5.0, 6.0]))
      b = ExBurn.BurnBridge.from_nx(Nx.tensor([1.0, 2.0, 3.0]))
      c = ExBurn.BurnBridge.sub(a, b)
      assert ExBurn.Tensor.shape(c) == [3]
    end

    @tag :nif
    test "neg returns correct shape" do
      a = ExBurn.BurnBridge.from_nx(Nx.tensor([1.0, -2.0, 3.0]))
      c = ExBurn.BurnBridge.neg(a)
      assert ExBurn.Tensor.shape(c) == [3]
    end
  end
end
