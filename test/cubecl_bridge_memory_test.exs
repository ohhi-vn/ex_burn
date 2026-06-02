defmodule ExBurn.CubeclBridgeMemoryTest do
  use ExUnit.Case

  describe "memory info" do
    test "memory_info returns a map or error" do
      result = ExBurn.CubeclBridge.memory_info()
      assert match?({:ok, %{}}, result) or match?({:error, _}, result)
    end

    test "memory_used returns a non-negative integer" do
      assert is_integer(ExBurn.CubeclBridge.memory_used(make_ref()))
      assert ExBurn.CubeclBridge.memory_used(make_ref()) >= 0
    end

    test "memory_total returns a non-negative integer" do
      assert is_integer(ExBurn.CubeclBridge.memory_total(make_ref()))
      assert ExBurn.CubeclBridge.memory_total(make_ref()) >= 0
    end
  end

  describe "device_summary/0" do
    test "returns a string" do
      summary = ExBurn.CubeclBridge.device_summary()
      assert is_binary(summary)
    end
  end

  describe "synchronize/1" do
    test "returns ok" do
      assert ExBurn.CubeclBridge.synchronize(make_ref()) == :ok
    end
  end

  describe "free/2" do
    test "returns ok" do
      assert ExBurn.CubeclBridge.free(make_ref(), make_ref()) == :ok
    end
  end

  describe "destroy/1" do
    test "returns ok" do
      assert ExBurn.CubeclBridge.destroy(make_ref()) == :ok
    end
  end

  describe "compile_kernel/3" do
    test "returns ok or error tuple" do
      {:ok, ctx} = ExBurn.CubeclBridge.init(:metal)
      result = ExBurn.CubeclBridge.compile_kernel(ctx, :add)
      assert match?({:ok, _}, result) or match?({:error, _}, result)
    end
  end

  describe "pipeline operations" do
    test "pipeline returns ok or error" do
      result = ExBurn.CubeclBridge.pipeline()
      assert match?({:ok, _}, result) or match?({:error, _}, result)
    end
  end
end
