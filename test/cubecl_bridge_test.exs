defmodule ExBurn.CubeclBridgeTest do
  use ExUnit.Case

  describe "availability" do
    test "available? returns boolean" do
      assert is_boolean(ExBurn.CubeclBridge.available?())
    end

    test "available_backends returns a list" do
      assert is_list(ExBurn.CubeclBridge.available_backends())
    end
  end

  describe "initialization" do
    test "init returns ok or error tuple" do
      result = ExBurn.CubeclBridge.init(:metal)
      assert match?({:ok, _}, result) or match?({:error, _}, result)
    end

    test "init with unsupported backend returns error when no GPU" do
      result = ExBurn.CubeclBridge.init(:cuda)
      assert match?({:ok, _}, result) or match?({:error, _}, result)
    end
  end

  describe "device info" do
    test "device_capabilities returns a map" do
      {:ok, ctx} = ExBurn.CubeclBridge.init(:metal)
      caps = ExBurn.CubeclBridge.device_capabilities(ctx)
      assert is_map(caps)
    end

    test "version returns a string" do
      assert is_binary(ExBurn.CubeclBridge.version())
    end
  end

  describe "kernel enumeration" do
    test "kernels returns ok or error tuple" do
      result = ExBurn.CubeclBridge.kernels()
      assert match?({:ok, [_ | _]}, result) or match?({:error, _}, result)
    end

    test "supported_dtypes returns a list" do
      dtypes = ExBurn.CubeclBridge.supported_dtypes()
      assert is_list(dtypes)
    end
  end
end
