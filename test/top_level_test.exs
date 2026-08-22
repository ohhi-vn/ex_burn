defmodule ExBurn.TopLevelTest do
  use ExUnit.Case, async: false

  @moduletag timeout: 20_000

  alias ExBurn.Backend, as: B
  alias Nx.Tensor, as: T

  # ── Top module helpers ───────────────────────────────────────────

  @tag :nif
  test "nif_loaded? reports health" do
    assert ExBurn.nif_loaded?() == true
  end

  @tag :nif
  test "nif_function_count returns the registered count" do
    count = ExBurn.nif_function_count()
    assert is_integer(count) and count > 10
  end

  @tag :nif
  test "smoke_test succeeds end-to-end" do
    previous = Nx.default_backend()
    assert :ok = ExBurn.smoke_test()
    # smoke_test restores whatever backend the caller had
    assert Nx.default_backend() == previous
  end

  # ── Nx dispatch through wrapped-arg bridge heads ─────────────────
  #
  # With the global default backend flipped, Nx hands every callback
  # %Nx.Tensor{} operands — exercising the wrapped clauses added for the
  # compiler path. All involved callbacks pin host views to BinaryBackend,
  # so this cannot recurse.

  # ── Wrapped-arg bridge heads ─────────────────────────────────────
  #
  # The dual-head callbacks accept both %Nx.Tensor{} wrappers (Nx dispatch
  # / compiler path) and bare structs (direct evaluator path). Exercise the
  # wrapped heads directly for deterministic coverage.

  defp btd(flat_values, shape) do
    data = Enum.reduce(flat_values, <<>>, fn v, acc -> <<acc::binary, v::float-32-native>> end)
    {:ok, ref} = ExBurn.NifHelper.new_tensor(data, shape, "f32")
    %B{ref: ref, shape: shape, type: :f32}
  end

  defp wt(flat_values, shape) do
    %T{
      data: btd(flat_values, shape),
      type: {:f, 32},
      shape: List.to_tuple(shape),
      names: List.duplicate(nil, length(shape))
    }
  end

  describe "wrapped-arg bridge heads" do
    @tag :nif
    test "clip accepts wrapped tensors" do
      result =
        B.clip(
          Nx.template({3}, {:f, 32}),
          wt([-5.0, 0.0, 5.0], [3]),
          wt([-1.0], []),
          wt([1.0], [])
        )

      assert match?(%B{}, result)
    end

    @tag :nif
    test "dot wrapped head delegates to the matmul fast path" do
      out = Nx.template({2, 2}, {:f, 32})

      result =
        B.dot(
          out,
          wt([1.0, 2.0, 3.0, 4.0], [2, 2]),
          [1],
          [],
          wt([5.0, 6.0, 7.0, 8.0], [2, 2]),
          [0],
          []
        )

      assert match?(%T{data: %B{}}, result)
      assert Nx.to_flat_list(result) == [19.0, 22.0, 43.0, 50.0]
    end

    @tag :nif
    test "pad wrapped head delegates with scalar value" do
      value = %T{data: btd([0.0], []), type: {:f, 32}, shape: {}, names: []}

      result = B.pad(Nx.template({4}, {:f, 32}), wt([1.0, 2.0], [2]), value, [{1, 1, 0}])
      assert match?(%B{}, result)
    end

    @tag :nif
    test "gather wrapped head takes by indices" do
      input = wt([10.0, 20.0, 30.0], [3])

      raw = Enum.reduce([2, 0], <<>>, fn v, acc -> <<acc::binary, v::signed-32-native>> end)
      {:ok, ref} = ExBurn.NifHelper.new_tensor(raw, [2], "f32")
      idx_data = %B{ref: ref, shape: [2], type: :i32}
      idx = %T{data: idx_data, type: {:s, 32}, shape: {2}, names: [nil]}

      result = B.gather(Nx.template({2}, {:f, 32}), input, idx, [])
      assert match?(%B{}, result)
    end

    @tag :nif
    test "stack expand handles wrapped inputs" do
      a = wt([1.0, 2.0], [2])
      b = wt([3.0, 4.0], [2])

      result = B.stack(Nx.template({2, 2}, {:f, 32}), [a, b], 0)
      assert match?(%B{}, result)
    end
  end
end
