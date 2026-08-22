defmodule ExBurn.CudaTest do
  @moduledoc """
  Tests for CUDA / GPU device support.

  Tests tagged `:nif` require the NIF to be compiled.
  Tests tagged `:cuda` require an actual NVIDIA GPU with CUDA runtime.
  Tests tagged `:metal` require an Apple GPU with Metal runtime.
  Tests tagged `:vulkan` require a GPU with Vulkan runtime.
  """

  use ExUnit.Case

  # ═══════════════════════════════════════════════════════════════════
  # Shared helpers
  # ═══════════════════════════════════════════════════════════════════

  defp active_backend do
    ExBurn.device_info()[:backend]
  end

  defp gpu? do
    ExBurn.Nif.gpu_available()
  end

  # ═══════════════════════════════════════════════════════════════════
  # ExBurn top-level API
  # ═══════════════════════════════════════════════════════════════════

  describe "ExBurn.device_info/0" do
    @tag :nif
    test "returns a map with expected keys" do
      info = ExBurn.device_info()
      assert is_map(info)
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
      assert info.backend in [:cuda, :metal, :vulkan, :cpu]
    end

    @tag :nif
    test "available_backends is a list of atoms" do
      info = ExBurn.device_info()
      assert is_list(info.available_backends)
      Enum.each(info.available_backends, fn b -> assert b in [:cuda, :metal, :vulkan, :rocm] end)
    end
  end

  describe "ExBurn.device_name/0" do
    @tag :nif
    test "returns a non-empty string" do
      name = ExBurn.device_name()
      assert is_binary(name)
      assert String.length(name) > 0
    end

    @tag :nif
    test "contains backend identifier" do
      name = ExBurn.device_name() |> String.downcase()

      assert name =~ "cuda" or name =~ "metal" or name =~ "vulkan" or name =~ "ndarray" or
               name =~ "cpu"
    end
  end

  describe "ExBurn.cuda_available?/0" do
    test "returns a boolean" do
      assert is_boolean(ExBurn.cuda_available?())
    end
  end

  describe "ExBurn.default_device/0" do
    @tag :nif
    test "returns :cpu or :gpu" do
      assert ExBurn.default_device() in [:cpu, :gpu]
    end

    @tag :nif
    test "is :gpu when NIF reports GPU available" do
      expected = if gpu?(), do: :gpu, else: :cpu
      assert ExBurn.default_device() == expected
    end
  end

  # ═══════════════════════════════════════════════════════════════════
  # NIF GPU primitives
  # ═══════════════════════════════════════════════════════════════════

  describe "NIF gpu_available / device_name" do
    @tag :nif
    test "gpu_available returns boolean" do
      assert is_boolean(ExBurn.Nif.gpu_available())
    end

    @tag :nif
    test "device_name returns a non-empty string" do
      name = ExBurn.Nif.device_name()
      assert is_binary(name)
      assert String.length(name) > 0
    end
  end

  # ═══════════════════════════════════════════════════════════════════
  # BurnBridge GPU API
  # ═══════════════════════════════════════════════════════════════════

  describe "BurnBridge GPU device functions" do
    @tag :nif
    test "gpu_available? returns boolean" do
      assert is_boolean(ExBurn.BurnBridge.gpu_available?())
    end

    @tag :nif
    test "device_name returns a string" do
      name = ExBurn.BurnBridge.device_name()
      assert is_binary(name)
      assert String.length(name) > 0
    end

    @tag :nif
    test "device_info returns a map with expected keys" do
      info = ExBurn.BurnBridge.device_info()
      assert is_map(info)
      assert Map.has_key?(info, :device)
      assert Map.has_key?(info, :gpu_available)
      assert Map.has_key?(info, :backend)
      assert Map.has_key?(info, :available_backends)
    end
  end

  # ═══════════════════════════════════════════════════════════════════
  # Consistency: NIF vs BurnBridge vs CubeclBridge
  # ═══════════════════════════════════════════════════════════════════

  describe "GPU availability consistency" do
    @tag :nif
    test "NIF and BurnBridge agree on gpu_available" do
      assert ExBurn.Nif.gpu_available() == ExBurn.BurnBridge.gpu_available?()
    end

    @tag :nif
    test "NIF and BurnBridge agree on device_name" do
      assert ExBurn.Nif.device_name() == ExBurn.BurnBridge.device_name()
    end

    @tag :nif
    test "ExBurn.device_info matches BurnBridge.device_info" do
      ex_info = ExBurn.device_info()
      bb_info = ExBurn.BurnBridge.device_info()
      assert ex_info.device == bb_info.device
      assert ex_info.gpu_available == bb_info.gpu_available
      assert ex_info.backend == bb_info.backend
    end
  end

  # ═══════════════════════════════════════════════════════════════════
  # Tensor device transfer (shared across all GPU backends)
  # ═══════════════════════════════════════════════════════════════════

  describe "tensor device transfer via BurnBridge" do
    @tag :nif
    test "to_gpu returns a Burn tensor with correct shape and type" do
      t = ExBurn.BurnBridge.from_nx(Nx.tensor([1.0, 2.0, 3.0]))
      gpu_t = ExBurn.BurnBridge.to_gpu(t)
      assert ExBurn.Tensor.shape(gpu_t) == [3]
      assert ExBurn.Tensor.type(gpu_t) == :f32
    end

    @tag :nif
    test "to_cpu returns a Burn tensor with correct shape and type" do
      t = ExBurn.BurnBridge.from_nx(Nx.tensor([1.0, 2.0, 3.0]))
      cpu_t = ExBurn.BurnBridge.to_cpu(t)
      assert ExBurn.Tensor.shape(cpu_t) == [3]
      assert ExBurn.Tensor.type(cpu_t) == :f32
    end

    @tag :nif
    test "to_gpu → to_cpu preserves 1D data" do
      original = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
      bt = ExBurn.BurnBridge.from_nx(original)
      gpu_bt = ExBurn.BurnBridge.to_gpu(bt)
      cpu_bt = ExBurn.BurnBridge.to_cpu(gpu_bt)
      result = ExBurn.BurnBridge.to_nx(cpu_bt)
      assert Nx.to_list(result) == [1.0, 2.0, 3.0, 4.0, 5.0]
    end

    @tag :nif
    test "to_gpu → to_cpu preserves 2D data" do
      original = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      bt = ExBurn.BurnBridge.from_nx(original)
      gpu_bt = ExBurn.BurnBridge.to_gpu(bt)
      cpu_bt = ExBurn.BurnBridge.to_cpu(gpu_bt)
      result = ExBurn.BurnBridge.to_nx(cpu_bt)

      assert ExBurn.Tensor.shape(cpu_bt) == [2, 2]
      assert Nx.to_list(result) == [[1.0, 2.0], [3.0, 4.0]]
    end
  end

  describe "tensor device transfer via NIF" do
    @tag :nif
    test "nif_to_gpu returns a resource reference" do
      t = ExBurn.BurnBridge.from_nx(Nx.tensor([1.0, 2.0, 3.0]))
      result = ExBurn.Nif.to_gpu(ExBurn.Tensor.ref(t))
      assert is_reference(result)
    end

    @tag :nif
    test "nif_to_cpu returns a resource reference" do
      t = ExBurn.BurnBridge.from_nx(Nx.tensor([1.0, 2.0, 3.0]))
      result = ExBurn.Nif.to_cpu(ExBurn.Tensor.ref(t))
      assert is_reference(result)
    end

    @tag :nif
    test "nif_to_gpu → nif_to_cpu round-trip preserves data" do
      original = Nx.tensor([10.0, 20.0, 30.0])
      bt = ExBurn.BurnBridge.from_nx(original)
      ref = ExBurn.Tensor.ref(bt)
      gpu_ref = ExBurn.Nif.to_gpu(ref)
      cpu_ref = ExBurn.Nif.to_cpu(gpu_ref)
      binary = ExBurn.Nif.tensor_to_binary(cpu_ref)
      vals = for <<x::float-32-native <- binary>>, do: x
      assert vals == [10.0, 20.0, 30.0]
    end
  end

  # ═══════════════════════════════════════════════════════════════════
  # Operations after GPU transfer (shared across all GPU backends)
  # ═══════════════════════════════════════════════════════════════════

  describe "arithmetic operations after GPU transfer" do
    @tag :nif
    test "add works on GPU-transferred tensors" do
      a = ExBurn.BurnBridge.from_nx(Nx.tensor([1.0, 2.0, 3.0]))
      b = ExBurn.BurnBridge.from_nx(Nx.tensor([4.0, 5.0, 6.0]))
      gpu_a = ExBurn.BurnBridge.to_gpu(a)
      gpu_b = ExBurn.BurnBridge.to_gpu(b)
      gpu_c = ExBurn.BurnBridge.add(gpu_a, gpu_b)
      cpu_c = ExBurn.BurnBridge.to_cpu(gpu_c)
      result = ExBurn.BurnBridge.to_nx(cpu_c)
      assert Nx.to_list(result) == [5.0, 7.0, 9.0]
    end

    @tag :nif
    test "mul works on GPU-transferred tensors" do
      a = ExBurn.BurnBridge.from_nx(Nx.tensor([2.0, 3.0, 4.0]))
      b = ExBurn.BurnBridge.from_nx(Nx.tensor([5.0, 6.0, 7.0]))
      gpu_a = ExBurn.BurnBridge.to_gpu(a)
      gpu_b = ExBurn.BurnBridge.to_gpu(b)
      gpu_c = ExBurn.BurnBridge.mul(gpu_a, gpu_b)
      cpu_c = ExBurn.BurnBridge.to_cpu(gpu_c)
      result = ExBurn.BurnBridge.to_nx(cpu_c)
      assert Nx.to_list(result) == [10.0, 18.0, 28.0]
    end

    @tag :nif
    test "sum works on GPU-transferred tensor" do
      a = ExBurn.BurnBridge.from_nx(Nx.tensor([1.0, 2.0, 3.0, 4.0]))
      gpu_a = ExBurn.BurnBridge.to_gpu(a)
      gpu_sum = ExBurn.BurnBridge.sum(gpu_a)
      cpu_sum = ExBurn.BurnBridge.to_cpu(gpu_sum)
      result = ExBurn.BurnBridge.to_nx(cpu_sum)
      [val] = Nx.to_list(result)
      assert_in_delta val, 10.0, 1.0e-4
    end
  end

  # ═══════════════════════════════════════════════════════════════════
  # CubeclBridge backend detection (no NIF required)
  # ═══════════════════════════════════════════════════════════════════

  describe "CubeclBridge backend detection" do
    test "cuda_available? returns boolean" do
      assert is_boolean(ExBurn.CubeclBridge.cuda_available?())
    end

    test "available_backends is a list" do
      assert is_list(ExBurn.CubeclBridge.available_backends())
    end

    test "available_backends entries are valid atoms" do
      backends = ExBurn.CubeclBridge.available_backends()
      valid = [:cuda, :metal, :vulkan, :wgpu, :rocm]
      Enum.each(backends, fn b -> assert b in valid end)
    end
  end

  # ═══════════════════════════════════════════════════════════════════
  # CUDA-specific tests (require NVIDIA GPU)
  # ═══════════════════════════════════════════════════════════════════

  describe "CUDA backend" do
    @tag :cuda
    test "gpu_available? returns true" do
      assert ExBurn.Nif.gpu_available() == true
    end

    @tag :cuda
    test "device_name mentions CUDA" do
      name = ExBurn.Nif.device_name()
      assert String.contains?(String.downcase(name), "cuda")
    end

    @tag :cuda
    test "backend is :cuda" do
      assert active_backend() == :cuda
    end

    @tag :cuda
    test "cuda_available? returns true" do
      assert ExBurn.cuda_available?() == true
    end

    @tag :cuda
    test "matmul on GPU produces correct result" do
      a = ExBurn.BurnBridge.from_nx(Nx.tensor([[1.0, 2.0], [3.0, 4.0]]))
      b = ExBurn.BurnBridge.from_nx(Nx.tensor([[5.0, 6.0], [7.0, 8.0]]))
      gpu_a = ExBurn.BurnBridge.to_gpu(a)
      gpu_b = ExBurn.BurnBridge.to_gpu(b)
      gpu_c = ExBurn.BurnBridge.matmul(gpu_a, gpu_b)
      cpu_c = ExBurn.BurnBridge.to_cpu(gpu_c)
      result = ExBurn.BurnBridge.to_nx(cpu_c)
      assert Nx.to_list(result) == [19.0, 22.0, 43.0, 50.0]
    end

    @tag :cuda
    test "large tensor operations complete without error" do
      vals = Enum.map(1..1000, &(&1 * 0.001))
      a = ExBurn.BurnBridge.from_nx(Nx.tensor(vals))
      gpu_a = ExBurn.BurnBridge.to_gpu(a)
      gpu_b = ExBurn.BurnBridge.add(gpu_a, gpu_a)
      cpu_b = ExBurn.BurnBridge.to_cpu(gpu_b)
      result = ExBurn.BurnBridge.to_nx(cpu_b)
      expected = Enum.map(vals, &(&1 * 2))
      actual = Nx.to_list(result)
      assert length(actual) == length(expected)

      Enum.zip(actual, expected)
      |> Enum.each(fn {a, e} ->
        assert_in_delta a, e, 1.0e-4
      end)
    end
  end

  # ═══════════════════════════════════════════════════════════════════
  # Metal-specific tests (require Apple GPU)
  # ═══════════════════════════════════════════════════════════════════

  describe "Metal backend" do
    @tag :metal
    test "gpu_available? returns true" do
      assert ExBurn.Nif.gpu_available() == true
    end

    @tag :metal
    test "device_name mentions Metal" do
      name = ExBurn.Nif.device_name()
      assert String.contains?(String.downcase(name), "metal")
    end

    @tag :metal
    test "backend is :metal" do
      assert active_backend() == :metal
    end

    @tag :metal
    test "matmul on GPU produces correct result" do
      a = ExBurn.BurnBridge.from_nx(Nx.tensor([[1.0, 2.0], [3.0, 4.0]]))
      b = ExBurn.BurnBridge.from_nx(Nx.tensor([[5.0, 6.0], [7.0, 8.0]]))
      gpu_a = ExBurn.BurnBridge.to_gpu(a)
      gpu_b = ExBurn.BurnBridge.to_gpu(b)
      gpu_c = ExBurn.BurnBridge.matmul(gpu_a, gpu_b)
      cpu_c = ExBurn.BurnBridge.to_cpu(gpu_c)
      result = ExBurn.BurnBridge.to_nx(cpu_c)
      assert Nx.to_list(result) == [19.0, 22.0, 43.0, 50.0]
    end

    @tag :metal
    test "large tensor operations complete without error" do
      vals = Enum.map(1..1000, &(&1 * 0.001))
      a = ExBurn.BurnBridge.from_nx(Nx.tensor(vals))
      gpu_a = ExBurn.BurnBridge.to_gpu(a)
      gpu_b = ExBurn.BurnBridge.add(gpu_a, gpu_a)
      cpu_b = ExBurn.BurnBridge.to_cpu(gpu_b)
      result = ExBurn.BurnBridge.to_nx(cpu_b)
      expected = Enum.map(vals, &(&1 * 2))
      actual = Nx.to_list(result)
      assert length(actual) == length(expected)

      Enum.zip(actual, expected)
      |> Enum.each(fn {a, e} ->
        assert_in_delta a, e, 1.0e-4
      end)
    end

    @tag :metal
    test "add on GPU-transferred tensors" do
      a = ExBurn.BurnBridge.from_nx(Nx.tensor([1.0, 2.0, 3.0]))
      b = ExBurn.BurnBridge.from_nx(Nx.tensor([4.0, 5.0, 6.0]))
      gpu_a = ExBurn.BurnBridge.to_gpu(a)
      gpu_b = ExBurn.BurnBridge.to_gpu(b)
      gpu_c = ExBurn.BurnBridge.add(gpu_a, gpu_b)
      cpu_c = ExBurn.BurnBridge.to_cpu(gpu_c)
      result = ExBurn.BurnBridge.to_nx(cpu_c)
      assert Nx.to_list(result) == [5.0, 7.0, 9.0]
    end

    @tag :metal
    test "mul on GPU-transferred tensors" do
      a = ExBurn.BurnBridge.from_nx(Nx.tensor([2.0, 3.0, 4.0]))
      b = ExBurn.BurnBridge.from_nx(Nx.tensor([5.0, 6.0, 7.0]))
      gpu_a = ExBurn.BurnBridge.to_gpu(a)
      gpu_b = ExBurn.BurnBridge.to_gpu(b)
      gpu_c = ExBurn.BurnBridge.mul(gpu_a, gpu_b)
      cpu_c = ExBurn.BurnBridge.to_cpu(gpu_c)
      result = ExBurn.BurnBridge.to_nx(cpu_c)
      assert Nx.to_list(result) == [10.0, 18.0, 28.0]
    end

    @tag :metal
    test "sum on GPU-transferred tensor" do
      a = ExBurn.BurnBridge.from_nx(Nx.tensor([1.0, 2.0, 3.0, 4.0]))
      gpu_a = ExBurn.BurnBridge.to_gpu(a)
      gpu_sum = ExBurn.BurnBridge.sum(gpu_a)
      cpu_sum = ExBurn.BurnBridge.to_cpu(gpu_sum)
      result = ExBurn.BurnBridge.to_nx(cpu_sum)
      [val] = Nx.to_list(result)
      assert_in_delta val, 10.0, 1.0e-4
    end

    @tag :metal
    test "to_gpu → to_cpu round-trip preserves 1D data" do
      original = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
      bt = ExBurn.BurnBridge.from_nx(original)
      gpu_bt = ExBurn.BurnBridge.to_gpu(bt)
      cpu_bt = ExBurn.BurnBridge.to_cpu(gpu_bt)
      result = ExBurn.BurnBridge.to_nx(cpu_bt)
      assert Nx.to_list(result) == [1.0, 2.0, 3.0, 4.0, 5.0]
    end

    @tag :metal
    test "to_gpu → to_cpu round-trip preserves 2D data" do
      original = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      bt = ExBurn.BurnBridge.from_nx(original)
      gpu_bt = ExBurn.BurnBridge.to_gpu(bt)
      cpu_bt = ExBurn.BurnBridge.to_cpu(gpu_bt)
      result = ExBurn.BurnBridge.to_nx(cpu_bt)
      assert Nx.to_list(result) == [1.0, 2.0, 3.0, 4.0]
    end
  end

  # ═══════════════════════════════════════════════════════════════════
  # Vulkan-specific tests (require Vulkan-capable GPU)
  # ═══════════════════════════════════════════════════════════════════

  describe "Vulkan backend" do
    @tag :vulkan
    test "gpu_available? returns true" do
      assert ExBurn.Nif.gpu_available() == true
    end

    @tag :vulkan
    test "device_name mentions Vulkan" do
      name = ExBurn.Nif.device_name()
      assert String.contains?(String.downcase(name), "vulkan")
    end

    @tag :vulkan
    test "backend is :vulkan" do
      assert active_backend() == :vulkan
    end

    @tag :vulkan
    test "matmul on GPU produces correct result" do
      a = ExBurn.BurnBridge.from_nx(Nx.tensor([[1.0, 2.0], [3.0, 4.0]]))
      b = ExBurn.BurnBridge.from_nx(Nx.tensor([[5.0, 6.0], [7.0, 8.0]]))
      gpu_a = ExBurn.BurnBridge.to_gpu(a)
      gpu_b = ExBurn.BurnBridge.to_gpu(b)
      gpu_c = ExBurn.BurnBridge.matmul(gpu_a, gpu_b)
      cpu_c = ExBurn.BurnBridge.to_cpu(gpu_c)
      result = ExBurn.BurnBridge.to_nx(cpu_c)
      assert Nx.to_list(result) == [19.0, 22.0, 43.0, 50.0]
    end

    @tag :vulkan
    test "large tensor operations complete without error" do
      vals = Enum.map(1..1000, &(&1 * 0.001))
      a = ExBurn.BurnBridge.from_nx(Nx.tensor(vals))
      gpu_a = ExBurn.BurnBridge.to_gpu(a)
      gpu_b = ExBurn.BurnBridge.add(gpu_a, gpu_a)
      cpu_b = ExBurn.BurnBridge.to_cpu(gpu_b)
      result = ExBurn.BurnBridge.to_nx(cpu_b)
      expected = Enum.map(vals, &(&1 * 2))
      actual = Nx.to_list(result)
      assert length(actual) == length(expected)

      Enum.zip(actual, expected)
      |> Enum.each(fn {a, e} ->
        assert_in_delta a, e, 1.0e-4
      end)
    end

    @tag :vulkan
    test "add on GPU-transferred tensors" do
      a = ExBurn.BurnBridge.from_nx(Nx.tensor([1.0, 2.0, 3.0]))
      b = ExBurn.BurnBridge.from_nx(Nx.tensor([4.0, 5.0, 6.0]))
      gpu_a = ExBurn.BurnBridge.to_gpu(a)
      gpu_b = ExBurn.BurnBridge.to_gpu(b)
      gpu_c = ExBurn.BurnBridge.add(gpu_a, gpu_b)
      cpu_c = ExBurn.BurnBridge.to_cpu(gpu_c)
      result = ExBurn.BurnBridge.to_nx(cpu_c)
      assert Nx.to_list(result) == [5.0, 7.0, 9.0]
    end

    @tag :vulkan
    test "mul on GPU-transferred tensors" do
      a = ExBurn.BurnBridge.from_nx(Nx.tensor([2.0, 3.0, 4.0]))
      b = ExBurn.BurnBridge.from_nx(Nx.tensor([5.0, 6.0, 7.0]))
      gpu_a = ExBurn.BurnBridge.to_gpu(a)
      gpu_b = ExBurn.BurnBridge.to_gpu(b)
      gpu_c = ExBurn.BurnBridge.mul(gpu_a, gpu_b)
      cpu_c = ExBurn.BurnBridge.to_cpu(gpu_c)
      result = ExBurn.BurnBridge.to_nx(cpu_c)
      assert Nx.to_list(result) == [10.0, 18.0, 28.0]
    end

    @tag :vulkan
    test "sum on GPU-transferred tensor" do
      a = ExBurn.BurnBridge.from_nx(Nx.tensor([1.0, 2.0, 3.0, 4.0]))
      gpu_a = ExBurn.BurnBridge.to_gpu(a)
      gpu_sum = ExBurn.BurnBridge.sum(gpu_a)
      cpu_sum = ExBurn.BurnBridge.to_cpu(gpu_sum)
      result = ExBurn.BurnBridge.to_nx(cpu_sum)
      [val] = Nx.to_list(result)
      assert_in_delta val, 10.0, 1.0e-4
    end

    @tag :vulkan
    test "to_gpu → to_cpu round-trip preserves 1D data" do
      original = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
      bt = ExBurn.BurnBridge.from_nx(original)
      gpu_bt = ExBurn.BurnBridge.to_gpu(bt)
      cpu_bt = ExBurn.BurnBridge.to_cpu(gpu_bt)
      result = ExBurn.BurnBridge.to_nx(cpu_bt)
      assert Nx.to_list(result) == [1.0, 2.0, 3.0, 4.0, 5.0]
    end

    @tag :vulkan
    test "to_gpu → to_cpu round-trip preserves 2D data" do
      original = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      bt = ExBurn.BurnBridge.from_nx(original)
      gpu_bt = ExBurn.BurnBridge.to_gpu(bt)
      cpu_bt = ExBurn.BurnBridge.to_cpu(gpu_bt)
      result = ExBurn.BurnBridge.to_nx(cpu_bt)
      assert Nx.to_list(result) == [1.0, 2.0, 3.0, 4.0]
    end
  end
end
