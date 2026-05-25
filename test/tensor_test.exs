defmodule ExBurn.TensorTest do
  use ExUnit.Case

  describe "type conversion" do
    test "nx_type_to_burn round-trips" do
      assert ExBurn.Tensor.nx_type_to_burn({:f, 32}) == :f32
      assert ExBurn.Tensor.nx_type_to_burn({:f, 64}) == :f64
      assert ExBurn.Tensor.nx_type_to_burn({:s, 32}) == :i32
      assert ExBurn.Tensor.nx_type_to_burn({:s, 64}) == :i64
    end

    test "burn_type_to_nx round-trips" do
      assert ExBurn.Tensor.burn_type_to_nx(:f32) == {:f, 32}
      assert ExBurn.Tensor.burn_type_to_nx(:f64) == {:f, 64}
      assert ExBurn.Tensor.burn_type_to_nx(:i32) == {:s, 32}
      assert ExBurn.Tensor.burn_type_to_nx(:i64) == {:s, 64}
    end

    test "unknown types default to f32" do
      assert ExBurn.Tensor.nx_type_to_burn({:u, 32}) == :f32
      assert ExBurn.Tensor.burn_type_to_nx(:unknown) == {:f, 32}
    end
  end

  describe "inspection" do
    @tag :nif
    test "shape/1 returns shape" do
      t = ExBurn.BurnBridge.zeros([2, 3, 4], :f32)
      assert ExBurn.Tensor.shape(t) == [2, 3, 4]
    end

    @tag :nif
    test "type/1 returns type" do
      t = ExBurn.BurnBridge.zeros([3], :f32)
      assert ExBurn.Tensor.type(t) == :f32
    end

    @tag :nif
    test "ref/1 returns reference" do
      t = ExBurn.BurnBridge.zeros([3], :f32)
      assert is_reference(ExBurn.Tensor.ref(t))
    end

    @tag :nif
    test "numel/1 returns element count" do
      t = ExBurn.BurnBridge.zeros([2, 3, 4], :f32)
      assert ExBurn.Tensor.numel(t) == 24
    end

    @tag :nif
    test "rank/1 returns dimension count" do
      t = ExBurn.BurnBridge.zeros([2, 3, 4], :f32)
      assert ExBurn.Tensor.rank(t) == 3
    end
  end
end
