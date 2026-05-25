defmodule ExBurnTest do
  use ExUnit.Case
  doctest ExBurn

  describe "version/0" do
    test "returns a version string" do
      assert is_binary(ExBurn.version())
    end
  end

  describe "default_device/0" do
    test "returns :cpu or :gpu" do
      assert ExBurn.default_device() in [:cpu, :gpu]
    end
  end

  describe "configure!/0" do
    test "sets ExBurn as the default backend" do
      previous = Nx.default_backend()

      try do
        assert ExBurn.configure!() == :ok
        assert Nx.default_backend() == {ExBurn.Backend, :ok}
      after
        Nx.default_backend(previous)
      end
    end
  end
end
