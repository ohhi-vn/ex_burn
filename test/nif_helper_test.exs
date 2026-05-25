defmodule ExBurn.NifHelperTest do
  use ExUnit.Case

  describe "GPU availability" do
    @tag :nif
    test "gpu_available returns boolean" do
      assert is_boolean(ExBurn.NifHelper.gpu_available())
    end

    @tag :nif
    test "device_name returns a string" do
      assert is_binary(ExBurn.NifHelper.device_name())
    end
  end

  describe "tensor creation" do
    @tag :nif
    test "new_tensor returns ok tuple" do
      result = ExBurn.NifHelper.new_tensor(<<1.0::float-32-native>>, [1], "f32")
      assert match?({:ok, _}, result)
    end

    @tag :nif
    test "zeros_tensor returns ok tuple" do
      result = ExBurn.NifHelper.zeros_tensor([2, 3], "f32")
      assert match?({:ok, _}, result)
    end

    @tag :nif
    test "ones_tensor returns ok tuple" do
      result = ExBurn.NifHelper.ones_tensor([3], "f32")
      assert match?({:ok, _}, result)
    end
  end

  describe "tensor inspection" do
    @tag :nif
    test "tensor_shape returns ok tuple" do
      {:ok, ref} = ExBurn.NifHelper.zeros_tensor([2, 3], "f32")
      result = ExBurn.NifHelper.tensor_shape(ref)
      assert match?({:ok, _}, result)
    end

    @tag :nif
    test "tensor_to_binary returns ok tuple" do
      {:ok, ref} = ExBurn.NifHelper.new_tensor(<<1.0::float-32-native>>, [1], "f32")
      result = ExBurn.NifHelper.tensor_to_binary(ref)
      assert match?({:ok, _}, result)
    end
  end

  describe "arithmetic dispatch" do
    @tag :nif
    test "add_tensor returns ok tuple" do
      {:ok, a} = ExBurn.NifHelper.new_tensor(<<1.0::float-32-native>>, [1], "f32")
      {:ok, b} = ExBurn.NifHelper.new_tensor(<<2.0::float-32-native>>, [1], "f32")
      result = ExBurn.NifHelper.add_tensor(a, b)
      assert match?({:ok, _}, result)
    end

    @tag :nif
    test "mul_tensor returns ok tuple" do
      {:ok, a} = ExBurn.NifHelper.new_tensor(<<2.0::float-32-native>>, [1], "f32")
      {:ok, b} = ExBurn.NifHelper.new_tensor(<<3.0::float-32-native>>, [1], "f32")
      result = ExBurn.NifHelper.mul_tensor(a, b)
      assert match?({:ok, _}, result)
    end
  end

  describe "memory management" do
    @tag :nif
    test "free_tensor returns ok" do
      {:ok, ref} = ExBurn.NifHelper.zeros_tensor([2], "f32")
      assert ExBurn.NifHelper.free_tensor(ref) == :ok
    end
  end
end
