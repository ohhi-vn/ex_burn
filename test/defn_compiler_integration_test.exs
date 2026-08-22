defmodule ExBurn.DefnCompilerIntegrationTest do
  use ExUnit.Case, async: false

  @compiler ExBurn.Defn.Compiler

  defp jit(fun, args, opts \\ []) do
    Nx.Defn.jit_apply(fun, List.wrap(args), Keyword.put(opts, :compiler, @compiler))
  end

  defmodule Programs do
    import Nx.Defn

    defn(add(x, y), do: Nx.add(x, y))
    defn(scale_by_const(x), do: Nx.multiply(x, 2.0))

    defn cache_reuse(x) do
      y = Nx.add(x, x)
      Nx.multiply(y, y)
    end

    defn(tuple_out(x), do: {Nx.add(x, 1.0), Nx.multiply(x, 2.0)})
    defn(map_out(x), do: %{a: Nx.add(x, 1.0), b: Nx.negate(x)})

    defn destructure(x) do
      {a, b} = {Nx.add(x, 1.0), Nx.multiply(x, 2.0)}
      Nx.add(a, b)
    end

    defn cond_true(x) do
      if Nx.sum(x) > 0.0 do
        Nx.add(x, 1.0)
      else
        Nx.subtract(x, 1.0)
      end
    end

    defn cond_last(x) do
      if Nx.sum(x) > 100.0 do
        Nx.add(x, 1.0)
      else
        Nx.subtract(x, 1.0)
      end
    end

    defn loop(x) do
      while x, Nx.less(Nx.sum(x), 10.0) do
        x + 3.0
      end
    end

    defn loop_never(x) do
      while x, Nx.greater(Nx.sum(x), 100.0) do
        x + 1.0
      end
    end

    defn(sum_reduce(x), do: Nx.reduce(x, 0.0, fn a, b -> Nx.add(a, b) end))
    defn(make_iota(_x), do: Nx.iota({4}))
    defn(concat_self(x), do: Nx.concatenate([x, x], axis: 0))
    defn(slice_mid(x), do: Nx.slice(x, [1], [2]))
    defn(hooked(x), do: Nx.Defn.Kernel.hook(Nx.multiply(x, 2.0), :doubler))
  end

  describe "arithmetic through the compiler" do
    @tag :nif
    test "parameters and element-wise ops evaluate" do
      result =
        jit(&Programs.add/2, [Nx.tensor([1.0, 2.0]), Nx.tensor([10.0, 20.0])])

      assert Nx.to_flat_list(result) == [11.0, 22.0]
    end

    @tag :nif
    test "scalar constants become burn tensors" do
      result = jit(&Programs.scale_by_const/1, Nx.tensor([1.0, 2.0, 3.0]))
      assert Nx.to_flat_list(result) == [2.0, 4.0, 6.0]
    end

    @tag :nif
    test "reused subexpressions hit the evaluation cache" do
      result = jit(&Programs.cache_reuse/1, Nx.tensor(2.0))
      assert Nx.to_number(result) == 16.0
    end

    @tag :nif
    test "garbage_collect option runs the same path" do
      result =
        jit(fn x -> Nx.add(x, 1.0) end, Nx.tensor(5.0), garbage_collect: true)

      assert Nx.to_number(result) == 6.0
    end
  end

  describe "composite outputs" do
    @tag :nif
    test "tuple outputs are zipped back into templates" do
      {a, b} = jit(&Programs.tuple_out/1, Nx.tensor(3.0))

      assert Nx.to_number(a) == 4.0
      assert Nx.to_number(b) == 6.0
    end

    @tag :nif
    test "map outputs are zipped back into templates" do
      %{a: a, b: b} = jit(&Programs.map_out/1, Nx.tensor(3.0))

      assert Nx.to_number(a) == 4.0
      assert Nx.to_number(b) == -3.0
    end

    @tag :nif
    test "tuple destructuring lowers to elem nodes" do
      result = jit(&Programs.destructure/1, Nx.tensor(3.0))
      assert Nx.to_number(result) == 10.0
    end
  end

  describe "control flow" do
    @tag :nif
    test "defn if lowers to cond and picks the true branch" do
      result = jit(&Programs.cond_true/1, Nx.tensor(5.0))
      assert Nx.to_number(result) == 6.0
    end

    @tag :nif
    test "cond falls through to the last clause" do
      result = jit(&Programs.cond_last/1, Nx.tensor(5.0))
      assert Nx.to_number(result) == 4.0
    end

    @tag :nif
    test "while loops iterate until the condition breaks" do
      # 0 → 3 → 6 → 9 → 12 (12 >= 10 stops)
      result = jit(&Programs.loop/1, Nx.tensor(0.0))
      assert Nx.to_number(result) == 12.0
    end

    @tag :nif
    test "while with a never-true condition returns the initial value" do
      result = jit(&Programs.loop_never/1, Nx.tensor(7.0))
      assert Nx.to_number(result) == 7.0
    end
  end

  describe "nested functions and reductions" do
    @tag :nif
    test "reduce lowers its body to a fun node" do
      result = jit(&Programs.sum_reduce/1, Nx.tensor([1.0, 2.0, 3.0]))
      assert Nx.to_number(result) == 6.0
    end
  end

  describe "creation / list / indices op dispatch" do
    @tag :nif
    test "iota inside defn uses the creation-op branch" do
      result = jit(&Programs.make_iota/1, Nx.tensor(0.0))
      assert Nx.to_flat_list(result) == [0.0, 1.0, 2.0, 3.0]
    end

    @tag :nif
    test "concatenate inside defn uses the list-op branch" do
      result = jit(&Programs.concat_self/1, Nx.tensor([1.0, 2.0]))

      assert Nx.shape(result) == {4}
      assert Nx.to_flat_list(result) == [1.0, 2.0, 1.0, 2.0]
    end

    @tag :nif
    test "slicing inside defn uses the indices-op branch" do
      result = jit(&Programs.slice_mid/1, Nx.tensor([1.0, 2.0, 3.0]))
      assert Nx.to_flat_list(result) == [2.0, 3.0]
    end
  end

  describe "hooks" do
    @tag :nif
    test "hook callbacks receive intermediate values" do
      parent = self()

      result =
        jit(&Programs.hooked/1, Nx.tensor([1.0]),
          hooks: %{doubler: fn v -> send(parent, {:hook_fired, v}) end}
        )

      assert Nx.to_flat_list(result) == [2.0]

      assert_receive {:hook_fired, _value}, 1_000
    end
  end

  describe "__compile__/4" do
    @tag :nif
    test "compiled function executes against parameter thunks" do
      # Vars must be parameter expressions so they fetch from the thunks
      var = Nx.Defn.Expr.parameter(:exburn_test_ctx, {:f, 32}, {2}, 0)

      compiled =
        @compiler.__compile__(:test_key, [var], fn [x] -> Nx.add(x, 1.0) end, [])

      [result] = compiled.([[fn -> Nx.tensor([10.0, 20.0]) end]])

      assert Nx.to_flat_list(result) == [11.0, 21.0]
    end

    @tag :nif
    test "embedded tensor vars evaluate through the :tensor clause" do
      concrete = Nx.tensor([5.0])

      [result] =
        @compiler.__jit__(
          :tensor_var_key,
          [Nx.Defn.Expr.tensor(concrete)],
          fn [x] -> Nx.add(x, 1.0) end,
          [[fn -> Nx.tensor([5.0]) end]],
          []
        )

      assert Nx.to_flat_list(result) == [6.0]
    end
  end

  describe "uncompilable arguments raise" do
    @tag :nif
    test "passing an expression as an argument raises ArgumentError" do
      # A thunk that yields a parameter expression node (not a concrete
      # tensor) must trigger the structured ArgumentError.
      expr = Nx.Defn.Expr.parameter(:exburn_test_ctx2, {:f, 32}, {1}, 1)

      assert_raise ArgumentError, ~r/cannot pass a tensor expression/, fn ->
        @compiler.__jit__(
          :bad_args_key,
          [Nx.Defn.Expr.parameter(:exburn_test_ctx2, {:f, 32}, {1}, 0)],
          fn [x] -> Nx.add(x, 1.0) end,
          [[fn -> expr end]],
          []
        )
      end
    end
  end
end
