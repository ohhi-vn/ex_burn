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
