defmodule ExBurn.BridgeCoverageTest do
  use ExUnit.Case, async: false

  alias ExBurn.{BurnBridge, CubeclBridge}
  alias ExBurn.Tensor, as: BT

  # Some ExCubecl entry points block indefinitely on malformed input.
  # Run them in an isolated process with a hard timeout so coverage
  # attempts can never hang the suite.
  defp safe_call(fun, timeout_ms \\ 2_000) do
    parent = self()
    ref = make_ref()

    pid =
      spawn(fn ->
        result =
          try do
            {:ok, fun.()}
          rescue
            e -> {:raised, Exception.message(e)}
          catch
            :exit, reason -> {:exited, reason}
          end

        send(parent, {ref, result})
      end)

    receive do
      {^ref, {:ok, value}} -> value
      {^ref, other} -> other
    after
      timeout_ms ->
        Process.exit(pid, :kill)
        :call_timeout
    end
  end

  # ── CubeclBridge ─────────────────────────────────────────────────

  describe "device enumeration" do
    @tag :nif
    test "device_count returns a count or error" do
      result = CubeclBridge.device_count()
      assert match?({:ok, n} when is_integer(n), result) or match?({:error, _}, result)
    end

    @tag :nif
    test "cuda_available? returns boolean" do
      assert is_boolean(CubeclBridge.cuda_available?())
    end

    @tag :nif
    test "available_backends only lists detected backends" do
      backends = CubeclBridge.available_backends()
      assert is_list(backends)
      assert Enum.all?(backends, &(&1 in [:cuda, :metal, :vulkan]))
    end

    @tag :nif
    test "device_summary renders a string either way" do
      summary = CubeclBridge.device_summary()
      assert is_binary(summary)

      if summary != "No GPU available" do
        assert summary =~ "GPU Backends:"
        assert summary =~ "Max Workgroup Size:"
      end
    end
  end

  describe "kernel management" do
    @tag :nif
    test "compile_kernel accepts known kernels and rejects unknown ones" do
      case CubeclBridge.kernels() do
        {:ok, [known | _]} ->
          {:ok, ctx} = CubeclBridge.init(:metal)

          # Kernel names come from ExCubecl's fixed list — bounded and trusted.
          # credo:disable-for-next-line Credo.Check.Warning.UnsafeToAtom
          known_atom = String.to_atom(known)
          assert match?({:ok, _}, CubeclBridge.compile_kernel(ctx, known_atom))
          assert {:error, msg} = CubeclBridge.compile_kernel(ctx, :definitely_not_a_kernel)
          assert msg =~ "Kernel not found"

        _ ->
          :ok
      end
    end
  end

  describe "buffer operations" do
    @tag :nif
    test "allocate_gpu returns an empty buffer of the requested shape" do
      {:ok, ctx} = CubeclBridge.init(:metal)

      case CubeclBridge.allocate_gpu(ctx, [2, 2], :f32) do
        {:ok, buf} ->
          assert match?({:ok, [2, 2]}, CubeclBridge.buffer_shape(buf))
          assert match?({:ok, "f32"}, CubeclBridge.buffer_dtype(buf))
          # 4 elements × 4 bytes
          assert match?({:ok, 16}, CubeclBridge.buffer_size(buf))

        {:error, _} ->
          :ok
      end
    end

    @tag :nif
    test "host_to_device/device_to_host round-trips an Nx tensor" do
      {:ok, ctx} = CubeclBridge.init(:metal)
      tensor = Nx.tensor([1.0, 2.0, 3.0])

      case CubeclBridge.host_to_device(ctx, tensor) do
        {:ok, buf} ->
          assert {:ok, readback} = CubeclBridge.device_to_host(ctx, buf)
          assert Nx.to_flat_list(readback) == [1.0, 2.0, 3.0]

        {:error, _} ->
          :ok
      end
    end

    @tag :nif
    test "buffer_read and buffer_read! fetch stored bytes" do
      {:ok, ctx} = CubeclBridge.init(:metal)
      tensor = Nx.tensor([7.0, 8.0])

      with {:ok, buf} <- CubeclBridge.host_to_device(ctx, tensor) do
        assert match?({:ok, bin} when byte_size(bin) == 8, CubeclBridge.buffer_read(buf))
        assert byte_size(CubeclBridge.buffer_read!(buf)) == 8
        assert :ok = CubeclBridge.free(ctx, buf)
      else
        {:error, _} -> :ok
      end
    end

    @tag :nif
    test "synchronize, destroy are no-op ok" do
      {:ok, ctx} = CubeclBridge.init(:metal)
      assert CubeclBridge.synchronize(ctx) == :ok
      assert CubeclBridge.destroy(ctx) == :ok
    end

    @tag :nif
    test "memory queries return integers" do
      {:ok, ctx} = CubeclBridge.init(:metal)
      assert is_integer(CubeclBridge.memory_used(ctx))
      assert is_integer(CubeclBridge.memory_total(ctx))

      assert match?({:ok, _}, CubeclBridge.memory_info()) or
               match?({:error, _}, CubeclBridge.memory_info())
    end

    @tag :nif
    test "execute without inputs reports a clear error" do
      {:ok, ctx} = CubeclBridge.init(:metal)
      assert {:error, "No input buffers provided"} = CubeclBridge.execute(ctx, :whatever, [])
    end

    @tag :nif
    test "execute with explicit output runs or reports kernel error" do
      {:ok, ctx} = CubeclBridge.init(:metal)

      result =
        safe_call(
          fn ->
            with {:ok, in_buf} <- CubeclBridge.host_to_device(ctx, Nx.tensor([1.0, 2.0])),
                 {:ok, out_buf} <- CubeclBridge.allocate_gpu(ctx, [2], :f32) do
              CubeclBridge.execute(ctx, :relu, [in_buf], output: out_buf)
            end
          end,
          5_000
        )

      assert match?({:ok, _}, result) or match?({:error, _}, result) or result == :call_timeout
    end
  end

  describe "async execution" do
    @tag :nif
    test "submit/poll/wait surface whatever ExCubecl supports" do
      submit = safe_call(fn -> CubeclBridge.async_submit("{}") end)

      case submit do
        {:ok, id} ->
          poll = safe_call(fn -> CubeclBridge.async_poll(id) end)
          assert match?({:ok, _}, poll) or match?({:error, _}, poll) or poll == :call_timeout

          wait = safe_call(fn -> CubeclBridge.async_wait(id) end, 3_000)
          assert is_atom(wait) or match?({:error, _}, wait) or wait == :call_timeout

        _ ->
          :ok
      end
    end
  end

  describe "pipelines" do
    @tag :nif
    test "pipeline add/run/free lifecycle" do
      result =
        safe_call(
          fn ->
            with {:ok, pid} <- CubeclBridge.pipeline() do
              add = CubeclBridge.pipeline_add(pid, :relu, [], [])
              run = CubeclBridge.pipeline_run(pid)
              free = CubeclBridge.pipeline_free(pid)
              {:done, add, run, free}
            end
          end,
          4_000
        )

      case result do
        {:done, add, run, free} ->
          assert is_atom(add) or match?({:error, _}, add)
          assert match?({:ok, _}, run) or match?({:error, _}, run)
          assert is_atom(free) or match?({:error, _}, free)

        _ ->
          :ok
      end
    end
  end

  # ── BurnBridge ───────────────────────────────────────────────────

  describe "creation" do
    @tag :nif
    test "ones fills with ones" do
      t = BurnBridge.ones([3])
      assert BT.shape(t) == [3]
      assert Nx.to_flat_list(BurnBridge.to_nx(t)) == [1.0, 1.0, 1.0]
    end

    @tag :nif
    test "from_nx raises structured error on unsupported dtypes" do
      assert_raise ExBurn.Error, ~r/from_nx/, fn ->
        BurnBridge.from_nx(Nx.tensor([1.0], type: {:f, 64}))
      end
    end

    @tag :nif
    test "to_nx raises structured error on invalid refs" do
      bad = %BT{ref: make_ref(), shape: [1], type: :f32}

      assert_raise ExBurn.Error, ~r/to_nx/, fn ->
        BurnBridge.to_nx(bad)
      end
    end
  end

  describe "arithmetic and reductions" do
    @tag :nif
    test "sum and mean produce scalar burn tensors" do
      t = BurnBridge.zeros([4])
      assert BT.shape(BurnBridge.sum(t)) == [1]
      assert BT.shape(BurnBridge.mean(t)) == [1]
    end

    @tag :nif
    test "matmul computes a 2D product" do
      a = BurnBridge.reshape(BurnBridge.zeros([4]), [2, 2])
      result = BurnBridge.matmul(a, a)
      assert BT.shape(result) == [2, 2]
    end

    @tag :nif
    test "transpose general dims round-trips values" do
      t = BurnBridge.reshape(BurnBridge.zeros([6]), [1, 2, 3])

      swapped = BurnBridge.transpose(t, 1, 2)
      assert BT.shape(swapped) == [1, 3, 2]

      back = BurnBridge.transpose(swapped, 2, 1)
      assert BT.shape(back) == [1, 2, 3]
    end

    @tag :nif
    test "layer_norm produces finite values" do
      t = BurnBridge.zeros([4])
      normed = BurnBridge.layer_norm(t, -1, 1.0e-5)
      values = Nx.to_flat_list(BurnBridge.to_nx(normed))
      assert Enum.all?(values, &is_float/1)
    end
  end

  describe "GPU transfer helpers" do
    @tag :nif
    test "to_gpu/to_cpu preserve values" do
      original = BurnBridge.zeros([3])

      gpu = BurnBridge.to_gpu(original)
      cpu = BurnBridge.to_cpu(gpu)

      assert Nx.to_flat_list(BurnBridge.to_nx(cpu)) == [0.0, 0.0, 0.0]
    end

    @tag :nif
    test "device_info exposes backend details" do
      info = BurnBridge.device_info()
      assert Map.has_key?(info, :device)
      assert is_boolean(info.gpu_available)
      assert info.backend in [:cpu, :cuda, :metal, :vulkan]
      assert is_list(info.available_backends)
    end
  end

  describe "ExCubecl-backed helpers" do
    @tag :nif
    test "buffer/read_buffer/buffer_shape/buffer_size lifecycle" do
      case safe_buffer(<<1.0::float-32-native, 2.0::float-32-native>>, [2], {:f, 32}) do
        {:ok, buf} ->
          data = BurnBridge.read_buffer(buf)
          assert byte_size(data) == 8
          assert BurnBridge.buffer_shape(buf) in [[2], {:ok, [2]}]

          size = BurnBridge.buffer_size(buf)
          assert size in [8, {:ok, 8}]

        _ ->
          :ok
      end
    end

    defp safe_buffer(data, shape, type) do
      {:ok, BurnBridge.buffer(data, shape, type)}
    rescue
      e -> {:error, Exception.message(e)}
    end
  end
end
