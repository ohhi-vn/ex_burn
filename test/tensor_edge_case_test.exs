defmodule ExBurn.TensorEdgeCaseTest do
  use ExUnit.Case

  describe "type conversion edge cases" do
    test "nx_type_to_burn preserves all float types" do
      assert ExBurn.Tensor.nx_type_to_burn({:f, 16}) == :f16
      assert ExBurn.Tensor.nx_type_to_burn({:bf, 16}) == :bf16
    end

    test "nx_type_to_burn preserves all integer types" do
      assert ExBurn.Tensor.nx_type_to_burn({:s, 8}) == :i8
      assert ExBurn.Tensor.nx_type_to_burn({:s, 16}) == :i16
      assert ExBurn.Tensor.nx_type_to_burn({:u, 8}) == :u8
    end

    test "nx_type_to_burn raises on unsupported types" do
      for bad <- [{:u, 32}, {:u, 64}, nil] do
        assert_raise(ExBurn.Error, fn -> ExBurn.Tensor.nx_type_to_burn(bad) end)
      end
    end

    test "burn_type_to_nx maps all supported burn types" do
      assert ExBurn.Tensor.burn_type_to_nx(:f16) == {:f, 16}
      assert ExBurn.Tensor.burn_type_to_nx(:bf16) == {:bf, 16}
      assert ExBurn.Tensor.burn_type_to_nx(:i16) == {:s, 16}
      assert ExBurn.Tensor.burn_type_to_nx(:i8) == {:s, 8}
      assert ExBurn.Tensor.burn_type_to_nx(:u8) == {:u, 8}
    end

    test "round-trip conversion for f32" do
      assert ExBurn.Tensor.burn_type_to_nx(ExBurn.Tensor.nx_type_to_burn({:f, 32})) == {:f, 32}
    end

    test "round-trip conversion for f64" do
      assert ExBurn.Tensor.burn_type_to_nx(ExBurn.Tensor.nx_type_to_burn({:f, 64})) == {:f, 64}
    end

    test "round-trip conversion for i32" do
      assert ExBurn.Tensor.burn_type_to_nx(ExBurn.Tensor.nx_type_to_burn({:s, 32})) == {:s, 32}
    end

    test "round-trip conversion for i64" do
      assert ExBurn.Tensor.burn_type_to_nx(ExBurn.Tensor.nx_type_to_burn({:s, 64})) == {:s, 64}
    end
  end

  describe "inspection edge cases" do
    @tag :nif
    test "numel for scalar-like tensor (shape [1])" do
      t = ExBurn.BurnBridge.zeros([1], :f32)
      assert ExBurn.Tensor.numel(t) == 1
    end

    @tag :nif
    test "numel for empty-like tensor (shape [0])" do
      t = ExBurn.BurnBridge.zeros([0], :f32)
      assert ExBurn.Tensor.numel(t) == 0
    end

    @tag :nif
    test "numel for large tensor" do
      t = ExBurn.BurnBridge.zeros([100, 100], :f32)
      assert ExBurn.Tensor.numel(t) == 10000
    end

    @tag :nif
    test "rank for 1D tensor" do
      t = ExBurn.BurnBridge.zeros([10], :f32)
      assert ExBurn.Tensor.rank(t) == 1
    end

    @tag :nif
    test "rank for 4D tensor" do
      t = ExBurn.BurnBridge.zeros([2, 3, 4, 5], :f32)
      assert ExBurn.Tensor.rank(t) == 4
    end

    @tag :nif
    test "shape returns correct shape for various dimensions" do
      t = ExBurn.BurnBridge.zeros([2, 3, 4], :f32)
      assert ExBurn.Tensor.shape(t) == [2, 3, 4]
    end
  end

  describe "batch conversion edge cases" do
    @tag :nif
    test "from_nx_batch with empty list" do
      assert {:ok, []} = ExBurn.Tensor.from_nx_batch([])
    end

    @tag :nif
    test "to_nx_batch with empty list" do
      assert {:ok, []} = ExBurn.Tensor.to_nx_batch([])
    end

    @tag :nif
    test "from_nx_batch with single tensor" do
      t = Nx.tensor([1.0, 2.0, 3.0])
      assert {:ok, [_]} = ExBurn.Tensor.from_nx_batch([t])
    end

    @tag :nif
    test "from_nx_batch with multiple tensors" do
      t1 = Nx.tensor([1.0, 2.0])
      t2 = Nx.tensor([3.0, 4.0])
      t3 = Nx.tensor([5.0, 6.0])
      assert {:ok, [_, _, _]} = ExBurn.Tensor.from_nx_batch([t1, t2, t3])
    end
  end

  describe "from_binary edge cases" do
    @tag :nif
    test "from_binary with valid data" do
      data = <<1.0::float-32-native, 2.0::float-32-native, 3.0::float-32-native>>
      assert {:ok, t} = ExBurn.Tensor.from_binary(data, [3], :f32)
      assert ExBurn.Tensor.shape(t) == [3]
      assert ExBurn.Tensor.type(t) == :f32
    end

    @tag :nif
    test "from_binary with 2D shape" do
      data =
        <<1.0::float-32-native, 2.0::float-32-native, 3.0::float-32-native, 4.0::float-32-native>>

      assert {:ok, t} = ExBurn.Tensor.from_binary(data, [2, 2], :f32)
      assert ExBurn.Tensor.shape(t) == [2, 2]
    end
  end
end
