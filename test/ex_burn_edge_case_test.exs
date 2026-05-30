defmodule ExBurn.MainEdgeCaseTest do
  use ExUnit.Case
  doctest ExBurn

  describe "version/0" do
    test "returns a non-empty string" do
      version = ExBurn.version()
      assert is_binary(version)
      assert String.length(version) > 0
    end

    test "returns a valid semver-like string" do
      version = ExBurn.version()
      # Should contain at least one dot (e.g., "0.3.0")
      assert String.contains?(version, ".")
    end
  end

  describe "default_device/0" do
    test "returns :cpu or :gpu" do
      device = ExBurn.default_device()
      assert device in [:cpu, :gpu]
    end
  end

  describe "device_name/0" do
    test "returns a non-empty string" do
      name = ExBurn.device_name()
      assert is_binary(name)
      assert String.length(name) > 0
    end
  end

  describe "device_info/0" do
    @tag :nif
    test "returns a map" do
      info = ExBurn.device_info()
      assert is_map(info)
    end

    @tag :nif
    test "contains expected keys" do
      info = ExBurn.device_info()
      assert Map.has_key?(info, :device)
      assert Map.has_key?(info, :gpu_available)
      assert Map.has_key?(info, :backend)
      assert Map.has_key?(info, :available_backends)
    end

    @tag :nif
    test "device is a string" do
      info = ExBurn.device_info()
      assert is_binary(info.device)
    end

    @tag :nif
    test "gpu_available is a boolean" do
      info = ExBurn.device_info()
      assert is_boolean(info.gpu_available)
    end

    @tag :nif
    test "backend is an atom" do
      info = ExBurn.device_info()
      assert is_atom(info.backend)
    end

    @tag :nif
    test "available_backends is a list" do
      info = ExBurn.device_info()
      assert is_list(info.available_backends)
    end
  end

  describe "cuda_available?/0" do
    test "returns a boolean" do
      assert is_boolean(ExBurn.cuda_available?())
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

    test "returns :ok" do
      previous = Nx.default_backend()

      try do
        result = ExBurn.configure!()
        assert result == :ok
      after
        Nx.default_backend(previous)
      end
    end
  end

  describe "summary/0" do
    test "returns a non-empty string" do
      summary = ExBurn.summary()
      assert is_binary(summary)
      assert String.length(summary) > 0
    end

    test "contains version" do
      summary = ExBurn.summary()
      assert summary =~ "ExBurn v"
    end

    test "contains device info" do
      summary = ExBurn.summary()
      assert summary =~ "Device:"
    end

    test "contains GPU info" do
      summary = ExBurn.summary()
      assert summary =~ "GPU:"
    end

    test "contains backends info" do
      summary = ExBurn.summary()
      assert summary =~ "Backends:"
    end
  end

  describe "nif_loaded?/0" do
    test "returns a boolean" do
      assert is_boolean(ExBurn.nif_loaded?())
    end
  end

  describe "nif_function_count/0" do
    test "returns a non-negative integer" do
      count = ExBurn.nif_function_count()
      assert is_integer(count)
      assert count >= 0
    end
  end

  describe "smoke_test/0" do
    test "returns :ok or error tuple" do
      result = ExBurn.smoke_test()
      assert result == :ok or match?({:error, _}, result)
    end

    test "restores backend after completion" do
      previous = Nx.default_backend()
      ExBurn.smoke_test()
      # After smoke_test, backend should be restored to BinaryBackend
      assert Nx.default_backend() == {Nx.BinaryBackend, []}
      # Restore original
      Nx.default_backend(previous)
    end
  end
end
