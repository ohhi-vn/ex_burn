defmodule ExBurn.BackendCallbacksTest do
  # Direct-callback testing: exercises ExBurn.Backend behaviour callbacks
  # without touching the global default backend, so Nx-side fallback
  # computations always run on Nx.BinaryBackend and can never re-enter
  # the backend recursively.
  use ExUnit.Case, async: false

  alias ExBurn.Backend, as: B
  alias Nx.Tensor, as: T

  # ── Helpers ──────────────────────────────────────────────────────

  defp bt(values), do: Nx.backend_transfer(Nx.tensor(values, type: {:f, 32}), ExBurn.Backend)

  defp out(shape), do: Nx.template(shape, {:f, 32})

  # Creates a backend struct whose *Rust-side* shape matches `shape` —
  # required for NIF ops (matmul/transpose/conv) that read stored shapes.
  defp btd(flat_values, shape) do
    data =
      Enum.reduce(flat_values, <<>>, fn v, acc -> <<acc::binary, v::float-32-native>> end)

    {:ok, ref} = ExBurn.NifHelper.new_tensor(data, shape, "f32")
    %B{ref: ref, shape: shape, type: :f32}
  end

  # Wrapped variant of btd/2 for callbacks matching %T{data: %__MODULE__{}}.
  defp btw(flat_values, shape) do
    data = btd(flat_values, shape)

    %T{
      data: data,
      type: {:f, 32},
      shape: List.to_tuple(shape),
      names: List.duplicate(nil, length(shape))
    }
  end

  # Wraps a bare backend struct so value assertions work on it. Callbacks
  # that call wrap/2 return an already-wrapped %Nx.Tensor{} — those are
  # asserted with Nx.to_flat_list directly.
  defp unwrap_to_list(%B{shape: shape, type: type} = data) do
    %T{
      data: data,
      type: ExBurn.Tensor.burn_to_nx(type),
      shape: List.to_tuple(shape),
      names: List.duplicate(nil, length(shape))
    }
    |> Nx.to_flat_list()
  end

  # Raw bytes holding small signed-32 integers, tagged :i32 — lets the
  # bitwise/shift callbacks run on sane values.
  defp raw_int_data(values, shape \\ nil) do
    shape = shape || [length(values)]

    data = Enum.reduce(values, <<>>, fn v, acc -> <<acc::binary, v::signed-32-native>> end)

    {:ok, ref} = ExBurn.NifHelper.new_tensor(data, shape, "f32")
    %B{ref: ref, shape: shape, type: :i32}
  end

  # ── Creation & metadata ──────────────────────────────────────────

  describe "constant/3" do
    @tag :nif
    test "fills every element of the shape" do
      res = B.constant(out({2, 2}), 3.5, [])
      assert %T{data: %B{}} = res
      assert Nx.shape(res) == {2, 2}
      assert Nx.to_flat_list(res) == [3.5, 3.5, 3.5, 3.5]
    end

    @tag :nif
    test "supports scalar shapes" do
      res = B.constant(out({}), -1.5, [])
      assert Nx.shape(res) == {}
      assert Nx.to_number(res) == -1.5
    end
  end

  describe "from_binary/3" do
    @tag :nif
    test "stores f32 data" do
      t = bt([1.0, 2.0])
      assert Nx.to_flat_list(t) == [1.0, 2.0]
    end

    @tag :nif
    test "re-encodes non-f32 dtypes as f32" do
      s32 = Nx.backend_transfer(Nx.tensor([1, 2, 3], type: {:s, 32}), ExBurn.Backend)
      assert Nx.to_flat_list(s32) == [1.0, 2.0, 3.0]

      bf16 = Nx.backend_transfer(Nx.tensor([0.5], type: {:bf, 16}), ExBurn.Backend)
      assert Nx.to_flat_list(bf16) == [0.5]
    end
  end

  describe "to_binary/2" do
    @tag :nif
    test "round-trips stored bytes" do
      t = bt([1.5, -2.5])
      assert Nx.to_flat_list(t) == [1.5, -2.5]
    end

    @tag :nif
    test "raises structured error for invalid ref" do
      bad =
        %T{
          data: %B{ref: make_ref(), shape: [1], type: :f32},
          type: {:f, 32},
          shape: {1},
          names: [nil]
        }

      assert_raise ExBurn.Error, ~r/to_binary/, fn ->
        Nx.to_binary(bad)
      end
    end
  end

  describe "metadata helpers" do
    @tag :nif
    test "tensor_type/1 maps back to an Nx type" do
      assert B.tensor_type(bt([1.0]).data) == {:f, 32}
    end

    @tag :nif
    test "parameter_count/1 multiplies the shape" do
      assert btd([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]) |> B.parameter_count() == 6
    end

    @tag :nif
    test "describe/1 renders shape, type and element count" do
      desc = B.describe(btd([1.0, 2.0], [2, 1]))
      assert desc =~ "ExBurn.Tensor<shape:"
      assert desc =~ "[2, 1]"
      assert desc =~ "type: f32"
      assert desc =~ "elements: 2"
    end
  end

  # ── Element-wise arithmetic ──────────────────────────────────────

  describe "binary nif ops" do
    @tag :nif
    test "wrap falls back to nil names when out carries no names" do
      bare = %B{ref: make_ref(), shape: [3], type: :f32}
      res = B.add(bare, bt([1.0, 2.0, 3.0]), bt([1.0, 1.0, 1.0]))
      assert %T{data: %B{}} = res
      assert res.names == [nil]
      assert Nx.shape(res) == {3}
    end

    @tag :nif
    test "compute element-wise results" do
      o = out({2})
      a = bt([1.0, 2.0])
      b = bt([4.0, 5.0])

      assert Nx.to_flat_list(B.add(o, a, b)) == [5.0, 7.0]
      assert Nx.to_flat_list(B.subtract(o, b, a)) == [3.0, 3.0]
      assert Nx.to_flat_list(B.multiply(o, a, b)) == [4.0, 10.0]
      assert Nx.to_flat_list(B.divide(o, b, a)) == [4.0, 2.5]
    end

    @tag :nif
    test "pow raises a structured error for tensor exponents (NIF takes scalars)" do
      assert_raise ExBurn.Error, ~r/pow/, fn ->
        B.pow(out({2}), bt([1.0, 2.0]), bt([2.0, 3.0]))
      end
    end

    @tag :nif
    test "fallback clause converts foreign-backend operands" do
      plain_a = Nx.tensor([1.0, 2.0])
      plain_b = Nx.tensor([3.0, 4.0])

      res = B.add(out({2}), plain_a, plain_b)
      assert %T{data: %B{}} = res
      assert Nx.to_flat_list(res) == [4.0, 6.0]

      res2 = B.subtract(out({2}), plain_b, plain_a)
      assert Nx.to_flat_list(res2) == [2.0, 2.0]
    end
  end

  describe "unary nif ops" do
    @tag :nif
    test "compute element-wise results" do
      o = out({2})
      t = bt([-1.0, 2.0])
      one = bt([0.0])

      assert Nx.to_flat_list(B.negate(o, t)) == [1.0, -2.0]
      assert Nx.to_flat_list(B.abs(o, t)) == [1.0, 2.0]
      assert Nx.to_flat_list(B.exp(out({1}), one)) == [1.0]
      assert Nx.to_flat_list(B.log(out({1}), bt([1.0]))) == [0.0]
      assert Nx.to_flat_list(B.sqrt(out({1}), bt([9.0]))) == [3.0]
      assert_in_delta B.sigmoid(out({1}), one) |> Nx.to_flat_list() |> hd(), 0.5, 1.0e-6
      assert_in_delta B.tanh(out({1}), one) |> Nx.to_flat_list() |> hd(), 0.0, 1.0e-6
    end

    @tag :nif
    test "fallback clause converts foreign-backend operand" do
      plain = Nx.tensor([2.0])
      res = B.negate(out({1}), plain)
      assert %T{data: %B{}} = res
      assert Nx.to_flat_list(res) == [-2.0]
    end
  end

  describe "remainder and quotient" do
    @tag :nif
    test "truncated remainder matches Nx semantics" do
      result = B.remainder(out({2}), bt([7.0, -7.0]), bt([2.0, 2.0]))
      assert unwrap_to_list(result) == [1.0, -1.0]
    end

    @tag :nif
    test "quotient floors the division result" do
      result = B.quotient(out({2}), bt([7.0, -7.0]), bt([2.0, 2.0]))
      assert unwrap_to_list(result) == [3.0, -4.0]
    end
  end

  describe "generic binary ops (Nx fallback family)" do
    @tag :nif
    test "float-safe ops return backend tensors with correct values" do
      a = btd([1.0, 5.0, 2.0, 3.0], [2, 2])
      b = btd([4.0, 5.0, 2.0, 1.0], [2, 2])

      assert a |> then(&B.min(nil, &1, b)) |> unwrap_to_list() == [1.0, 5.0, 2.0, 1.0]
      assert B.max(nil, a, b) |> unwrap_to_list() == [4.0, 5.0, 2.0, 3.0]

      assert B.equal(nil, a, b) |> unwrap_to_list() == [0.0, 1.0, 1.0, 0.0]
      assert B.not_equal(nil, a, b) |> unwrap_to_list() == [1.0, 0.0, 0.0, 1.0]
      assert B.greater(nil, a, b) |> unwrap_to_list() == [0.0, 0.0, 0.0, 1.0]
      assert B.less(nil, a, b) |> unwrap_to_list() == [1.0, 0.0, 0.0, 0.0]
      assert B.greater_equal(nil, a, b) |> unwrap_to_list() == [0.0, 1.0, 1.0, 1.0]
      assert B.less_equal(nil, a, b) |> unwrap_to_list() == [1.0, 1.0, 1.0, 0.0]

      truthy = btd([1.0, 0.0, 2.0], [3])
      falsy = btd([0.0, 0.0, 0.0], [3])

      assert B.logical_and(nil, truthy, falsy) |> unwrap_to_list() == [0.0, 0.0, 0.0]
      assert B.logical_or(nil, truthy, falsy) |> unwrap_to_list() == [1.0, 0.0, 1.0]
      assert B.logical_xor(nil, truthy, truthy) |> unwrap_to_list() == [0.0, 0.0, 0.0]

      assert match?(%B{}, B.atan2(nil, a, b))
    end

    @tag :nif
    test "integer-tagged storage supports bitwise ops" do
      a = raw_int_data([1, 3])
      b = raw_int_data([3, 1])
      small_shift = raw_int_data([1, 0])

      assert match?(%B{}, B.bitwise_and(nil, a, b))
      assert match?(%B{}, B.bitwise_or(nil, a, b))
      assert match?(%B{}, B.bitwise_xor(nil, a, b))
      assert match?(%B{}, B.left_shift(nil, a, small_shift))
      assert match?(%B{}, B.right_shift(nil, a, small_shift))
    end
  end

  describe "generic unary ops (Nx fallback family)" do
    @tag :nif
    test "float-safe unary ops return backend tensors" do
      data = btd([0.0, 0.5, 1.0], [3])

      float_ops = [
        :acos,
        :acosh,
        :asin,
        :asinh,
        :atan,
        :atanh,
        :cbrt,
        :ceil,
        :conjugate,
        :cos,
        :cosh,
        :erf,
        :erfc,
        :erf_inv,
        :expm1,
        :floor,
        :log1p,
        :rsqrt,
        :sin,
        :sinh,
        :tan,
        :round,
        :sign,
        :real,
        :imag,
        :is_nan,
        :is_infinity
      ]

      for op <- float_ops do
        result = apply(B, op, [nil, data])
        assert match?(%B{}, result), "expected #{op} to return a backend struct"
      end
    end

    @tag :nif
    test "bitwise unary ops work on integer-tagged storage" do
      data = raw_int_data([1, 2])

      assert match?(%B{}, B.bitwise_not(nil, data))
      assert match?(%B{}, B.count_leading_zeros(nil, data))
      assert match?(%B{}, B.population_count(nil, data))
    end
  end

  # ── Reductions ───────────────────────────────────────────────────

  describe "reductions" do
    @tag :nif
    test "sum full goes through NIF, partial axes through Nx" do
      m = btw([1.0, 2.0, 3.0, 4.0], [2, 2])
      full = B.sum(out({1}), m, [])

      # Full reduction wraps into a tensor via wrap/2
      assert match?(%T{data: %B{}}, full)

      partial = B.sum(out({2}), m, axes: [0])
      assert unwrap_to_list(partial) == [4.0, 6.0]

      partial2 = B.sum(out({2}), m, axes: [1])
      assert unwrap_to_list(partial2) == [3.0, 7.0]
    end

    @tag :nif
    test "product reduces exactly" do
      v = btw([2.0, 3.0, 4.0], [3])
      assert B.product(out({1}), v, []) |> unwrap_to_list() == [24.0]

      m = btw([1.0, 2.0, 3.0, 4.0], [2, 2])
      assert B.product(out({2}), m, axes: [1]) |> unwrap_to_list() == [2.0, 12.0]
    end

    @tag :nif
    test "reduce_max full goes through NIF, partial through Nx" do
      v = btw([1.0, 5.0, 3.0], [3])
      full = B.reduce_max(out({1}), v, [])
      assert match?(%T{data: %B{}}, full)

      m = btw([1.0, 5.0, 3.0, 2.0], [2, 2])
      assert B.reduce_max(out({2}), m, axes: [1]) |> unwrap_to_list() == [5.0, 3.0]
    end

    @tag :nif
    test "reduce_min full goes through NIF, partial through Nx" do
      v = btw([4.0, 1.0, 3.0], [3])
      full = B.reduce_min(out({1}), v, [])
      assert match?(%T{data: %B{}}, full)

      m = btw([4.0, 1.0, 3.0, 2.0], [2, 2])
      assert B.reduce_min(out({2}), m, axes: [1]) |> unwrap_to_list() == [1.0, 2.0]
    end

    @tag :nif
    test "argmax and argmin return indices" do
      m = btw([1.0, 5.0, 3.0, 2.0], [2, 2])

      assert B.argmax(out({2}), m, axis: 1) |> unwrap_to_list() == [1.0, 0.0]
      assert B.argmin(out({2}), m, axis: 1) |> unwrap_to_list() == [0.0, 1.0]

      v = btw([7.0, 2.0], [2])
      assert B.argmax(out({1}), v, []) |> unwrap_to_list() == [0.0]
    end

    @tag :nif
    test "all and any reduce booleans" do
      yes = btw([1.0, 2.0], [2])
      no = btw([1.0, 0.0], [2])
      none = btw([0.0, 0.0], [2])
      some = btw([0.0, 2.0], [2])

      assert B.all(out({1}), yes, []) |> unwrap_to_list() == [1.0]
      assert B.all(out({1}), no, []) |> unwrap_to_list() == [0.0]
      assert B.any(out({1}), some, []) |> unwrap_to_list() == [1.0]
      assert B.any(out({1}), none, []) |> unwrap_to_list() == [0.0]
    end

    @tag :nif
    test "reduce applies the aggregation function via Nx" do
      m = btw([1.0, 2.0, 3.0, 4.0], [2, 2])
      acc = btd([0.0], [])

      result = B.reduce(out({2}), m, acc, [axes: [1]], fn x, y -> Nx.add(x, y) end)
      assert unwrap_to_list(result) == [3.0, 7.0]
    end

    @tag :nif
    test "gather takes elements by flat indices" do
      input = btd([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [6])
      indices = raw_int_data([0, 5], [2])

      result = B.gather(out({2}), input, indices, [])
      assert unwrap_to_list(result) == [1.0, 6.0]
    end
  end

  # ── Linear algebra ───────────────────────────────────────────────

  describe "dot" do
    @tag :nif
    test "2D contraction uses the matmul fast path and wraps the result" do
      a = btd([1.0, 2.0, 3.0, 4.0], [2, 2])
      b = btd([5.0, 6.0, 7.0, 8.0], [2, 2])

      result = B.dot(out({2, 2}), a, [1], [], b, [0], [])
      assert match?(%T{data: %B{}}, result)
      assert Nx.to_flat_list(result) == [19.0, 22.0, 43.0, 50.0]
    end

    @tag :nif
    test "batched contraction falls back to exact Nx dot" do
      a = btd([1.0, 2.0, 3.0, 4.0], [2, 1, 2])
      b = btd([1.0, 2.0, 1.0, 2.0], [2, 2, 1])

      result = B.dot(out({2, 1, 1}), a, [2], [0], b, [1], [0])
      assert unwrap_to_list(result) == [5.0, 11.0]
    end

    @tag :nif
    test "vector-vector dot falls back to exact Nx dot" do
      v = btd([1.0, 2.0, 3.0], [3])
      result = B.dot(out({1}), v, [0], [], v, [0], [])
      assert unwrap_to_list(result) == [14.0]
    end
  end

  describe "transpose" do
    @tag :nif
    test "2D transpose uses the NIF path and wraps the result" do
      m = btw([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])

      result = B.transpose(out({3, 2}), m, [1, 0])
      assert match?(%T{data: %B{}}, result)
      assert Nx.to_flat_list(result) == [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    end

    @tag :nif
    test "higher-rank transposes use the Nx fallback" do
      t = btw(Enum.map(0..23, &(&1 * 1.0)), [2, 3, 4])

      explicit = B.transpose(out({3, 2, 4}), t, [1, 0, 2])

      assert unwrap_to_list(explicit) ==
               Enum.flat_map(0..2, fn c ->
                 Enum.flat_map(0..1, fn r -> Enum.map(0..3, &(r * 12 + c * 4 + &1)) end)
               end)

      reversed = B.transpose(out({4, 3, 2}), t, nil)
      assert length(unwrap_to_list(reversed)) == 24
    end
  end

  # ── Shape manipulation ───────────────────────────────────────────

  describe "reshape / squeeze / broadcast" do
    @tag :nif
    test "reshape changes shape preserving order" do
      r = B.reshape(out({2, 3}), btw([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [6]))
      assert match?(%T{data: %B{}}, r)
      assert Nx.to_flat_list(r) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    end

    @tag :nif
    test "squeeze raises while the NIF rejects rank-changing reshapes" do
      assert_raise ExBurn.Error, ~r/squeeze/, fn ->
        B.squeeze(out({3}), btw([1.0, 2.0, 3.0], [1, 3, 1]), [0, 2])
      end
    end

    @tag :nif
    test "broadcast expands to the requested shape" do
      # Note: the NIF repeats the flat data rather than tiling rows like
      # Nx.broadcast; we assert the structural contract here.
      b = B.broadcast(out({2, 3}), btw([1.0, 2.0, 3.0], [3]), {2, 3}, [0])
      assert match?(%T{data: %B{}}, b)
      assert Nx.shape(b) == {2, 3}
      assert length(Nx.to_flat_list(b)) == 6
    end
  end

  describe "pad" do
    @tag :nif
    test "pads with the given scalar value" do
      v = btd([1.0, 2.0, 3.0], [3])
      pad_value = btd([0.0], [])

      padded = B.pad(out({5}), v, pad_value, [{1, 1, 0}])
      assert unwrap_to_list(padded) == [0.0, 1.0, 2.0, 3.0, 0.0]
    end
  end

  describe "slice" do
    @tag :nif
    test "extracts subranges with strides" do
      v = btw([1.0, 2.0, 3.0, 4.0, 5.0], [5])

      sliced = B.slice(out({3}), v, [1], [3], [1])
      assert unwrap_to_list(sliced) == [2.0, 3.0, 4.0]

      strided = B.slice(out({3}), v, [0], [5], [2])
      assert unwrap_to_list(strided) == [1.0, 3.0, 5.0]

      m = btw([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
      rows = B.slice(out({1, 3}), m, [0, 0], [1, 3], [1, 1])
      assert unwrap_to_list(rows) == [1.0, 2.0, 3.0]
    end
  end

  describe "concatenate and stack" do
    @tag :nif
    test "1D concatenation chains multiple tensors" do
      a = btd([1.0, 2.0], [2])
      b = btd([3.0], [1])
      c = btd([4.0, 5.0], [2])

      result = B.concatenate(out({5}), [a, b, c], 0)
      assert unwrap_to_list(result) == [1.0, 2.0, 3.0, 4.0, 5.0]
    end

    @tag :nif
    test "multi-dimensional concatenation handles non-zero axes" do
      a = btd([1.0, 2.0, 3.0, 4.0], [2, 2])
      b = btd([5.0, 6.0, 7.0, 8.0], [2, 2])

      axis1 = B.concatenate(out({2, 4}), [a, b], 1)
      assert unwrap_to_list(axis1) == [1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]
    end

    @tag :nif
    test "stack builds a new leading axis from bare structs" do
      a = btd([1.0, 2.0], [2])
      b = btd([3.0, 4.0], [2])

      result = B.stack(out({2, 2}), [a, b], 0)
      assert unwrap_to_list(%{result | shape: [2, 2]}) == [1.0, 2.0, 3.0, 4.0]
    end

    @tag :nif
    test "stack accepts wrapped tensors as inputs" do
      a = bt([1.0, 2.0])
      b = bt([3.0, 4.0])

      result = B.stack(out({2, 2}), [a.data, b.data], 0)
      assert unwrap_to_list(%{result | shape: [2, 2]}) == [1.0, 2.0, 3.0, 4.0]
    end

    @tag :nif
    test "reverse flips along given axes" do
      v = btw([1.0, 2.0, 3.0], [3])
      assert B.reverse(out({3}), v, [0]) |> unwrap_to_list() == [3.0, 2.0, 1.0]

      m = btw([1.0, 2.0, 3.0, 4.0], [2, 2])
      assert B.reverse(out({2, 2}), m, [1]) |> unwrap_to_list() == [2.0, 1.0, 4.0, 3.0]
    end
  end

  describe "select" do
    @tag :nif
    test "selects elements based on pred" do
      pred = btw([1.0, 0.0, 2.0], [3])
      on_true = btw([10.0, 20.0, 30.0], [3])
      on_false = btw([-10.0, -20.0, -30.0], [3])

      result = B.select(out({3}), pred, on_true, on_false)
      assert unwrap_to_list(result) == [10.0, -20.0, 30.0]
    end
  end

  # ── Random / creation ops ────────────────────────────────────────

  describe "random_uniform" do
    @tag :nif
    test "generates values within bounds" do
      res = B.random_uniform(out({2, 50}), low: -2.0, high: 3.0)
      assert match?(%T{data: %B{}}, res)

      values = Nx.to_flat_list(res)
      assert length(values) == 100
      assert Enum.all?(values, fn v -> v >= -2.0 and v < 3.0 end)
    end
  end

  describe "random_normal" do
    @tag :nif
    test "generates a normally distributed tensor" do
      res = B.random_normal(out({3, 3}), mean: 0.0, std: 1.0)
      assert match?(%B{}, res)

      values = unwrap_to_list(res)
      assert length(values) == 9
      assert Enum.all?(values, &is_float/1)
    end
  end

  describe "eye and iota" do
    @tag :nif
    test "eye produces identity" do
      res = B.eye(out({3, 3}), [])
      assert Nx.shape(res) == {3, 3}
      assert Nx.to_flat_list(res) == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    end

    @tag :nif
    test "iota fills sequentially along axis 0" do
      res = B.iota(out({4}), nil, [])
      assert Nx.to_flat_list(res) == [0.0, 1.0, 2.0, 3.0]

      res0 = B.iota(out({3}), 0, [])
      assert Nx.to_flat_list(res0) == [0.0, 1.0, 2.0]
    end
  end

  # ── Clip / conv ──────────────────────────────────────────────────

  describe "clip" do
    @tag :nif
    test "clamps values between min and max" do
      t = btd([-5.0, 0.0, 5.0], [3])
      min = btd([-1.0], [])
      max = btd([1.0], [])

      result = B.clip(out({3}), t, min, max)
      assert unwrap_to_list(result) == [-1.0, 0.0, 1.0]
    end
  end

  describe "conv" do
    @tag :nif
    test "computes a 2D convolution" do
      input = btd(List.duplicate(1.0, 16), [1, 1, 4, 4])
      kernel = btd(List.duplicate(1.0, 4), [1, 1, 2, 2])

      opts = [stride: [1, 1], padding: [{0, 0}, {0, 0}]]
      result = B.conv(out({1, 3, 3, 1}), input, kernel, opts)

      assert match?(%T{data: %B{}}, result)
      assert Nx.to_flat_list(result) == List.duplicate(4.0, 9)
    end
  end

  # ── Not-implemented callbacks raise ──────────────────────────────

  describe "callbacks without implementations raise" do
    @tag :nif
    test "window/indexed/sort/fft/solve/batched families raise ExBurn.Error" do
      raising = [
        {:window_sum, 4},
        {:window_max, 4},
        {:window_min, 4},
        {:window_product, 4},
        {:window_reduce, 6},
        {:window_scatter_max, 6},
        {:window_scatter_min, 6},
        {:indexed_add, 5},
        {:indexed_put, 5},
        {:put_slice, 4},
        {:fft, 3},
        {:sort, 3},
        {:argsort, 3},
        {:triangular_solve, 4},
        {:to_batched, 3}
      ]

      for {fun, arity} <- raising do
        args = List.duplicate(:dummy, arity)

        assert_raise ExBurn.Error, ~r/not implemented by ExBurn.Backend/, fn ->
          apply(B, fun, args)
        end
      end
    end
  end

  # ── Transfer / misc identity callbacks ───────────────────────────

  describe "backend transfer callbacks" do
    @tag :nif
    test "copy and transfer return the same struct unchanged" do
      data = bt([1.0]).data

      assert B.backend_copy(data, Nx.BinaryBackend, []) == data
      assert B.backend_transfer(data, Nx.BinaryBackend, []) == data
    end

    @tag :nif
    test "deallocate frees the underlying resource" do
      data = bt([42.0]).data
      assert B.backend_deallocate(data) == :ok
    end

    @tag :nif
    test "round-trip through BinaryBackend preserves values" do
      t = bt([1.5, -2.5])

      restored =
        t
        |> Nx.to_binary()
        |> Nx.from_binary({:f, 32})
        |> Nx.backend_transfer(ExBurn.Backend)

      assert Nx.to_flat_list(restored) == [1.5, -2.5]
    end
  end

  describe "type conversion callbacks" do
    @tag :nif
    test "as_type returns the struct unchanged when already f32" do
      data = bt([1.0]).data
      assert B.as_type(out({1}), data) == data
    end

    @tag :nif
    test "as_type re-encodes mistagged storage to f32" do
      mistagged = %{btd([1.0], [1]) | type: :i64}
      result = B.as_type(out({1}), mistagged)
      assert %B{type: :f32} = result
      refute result.ref == mistagged.ref
    end

    @tag :nif
    test "maybe_cast tolerates unreadable refs" do
      unreadable = %B{ref: make_ref(), shape: [1], type: :i64}
      result = B.as_type(out({1}), unreadable)
      assert %B{type: :f32} = result
      assert result.ref == unreadable.ref
    end

    @tag :nif
    test "mixed tagged types promote via result_type and still compute" do
      # f32 has higher precedence than i64, so both sides cast to f32
      a = %{btd([1.0, 2.0], [2]) | type: :i64}
      b = bt([3.0, 4.0])

      ta = %T{data: a, type: {:s, 64}, shape: {2}, names: [nil]}

      result = B.add(out({2}), ta, b)
      assert result.data.type == :f32
      assert Nx.to_flat_list(result) == [4.0, 6.0]
    end

    @tag :nif
    test "bitcast is an identity operation" do
      data = bt([1.0]).data
      assert B.bitcast(out({1}), data) == data
    end
  end

  describe "inspect" do
    @tag :nif
    test "small tensors include a data preview" do
      text = inspect(bt([1.0, 2.0]))
      assert text =~ "#ExBurn.Tensor<shape:"
      assert text =~ "data:"
    end

    @tag :nif
    test "large tensors omit the preview" do
      big = Nx.backend_transfer(Nx.tensor(Enum.map(1..150, &(&1 * 1.0))), ExBurn.Backend)
      text = inspect(big)
      assert text =~ "#ExBurn.Tensor<shape:"
      refute text =~ "data:"
    end
  end

  describe "pointer callbacks" do
    @tag :nif
    test "from_pointer raises" do
      assert_raise ExBurn.Error, ~r/from_pointer/, fn ->
        B.from_pointer(make_ref(), {:f, 32}, [1], [], [])
      end
    end

    @tag :nif
    test "to_pointer raises" do
      assert_raise ExBurn.Error, ~r/to_pointer/, fn ->
        B.to_pointer(bt([1.0]).data, [])
      end
    end
  end

  describe "block/4" do
    @tag :nif
    test "applies the function to output and args" do
      out_t = out({2})
      arg = bt([1.0])

      {o, a} = B.block(nil, out_t, [arg], fn out_arg, x -> {out_arg, x} end)
      assert o == out_t
      assert a == arg
    end
  end

  # ── Nx.Container protocol ────────────────────────────────────────

  describe "Nx.Container protocol" do
    @tag :nif
    test "the backend data is treated as an opaque leaf" do
      data = bt([1.0, 2.0]).data

      acc = Nx.Container.reduce(data, 0, fn _leaf, acc -> acc + 1 end)
      assert acc == 0

      {out_data, out_acc} =
        Nx.Container.traverse(data, :acc, fn leaf, acc -> {leaf, acc} end)

      assert out_data == data
      assert out_acc == :acc
    end

    @tag :nif
    test "unwrap treats the struct as having no tensor children" do
      data = bt([1.0]).data
      impl = Nx.Container.ExBurn.Backend
      assert impl.unwrap(data, fn leaf -> leaf end) == []
    end

    @tag :nif
    test "serialize is rejected" do
      data = bt([1.0]).data

      assert_raise RuntimeError, ~r/cannot serialize/, fn ->
        Nx.Container.ExBurn.Backend.serialize(data)
      end
    end

    @tag :nif
    test "protocol functions traverse composites containing backend leaves" do
      composite = {%{key: bt([1.0]).data}, bt([2.0]).data}
      count = Nx.Container.reduce(composite, 0, fn _leaf, acc -> acc + 1 end)
      assert count == 2
    end
  end
end
