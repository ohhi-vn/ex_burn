defmodule ExBurn.Defn.CompilerEdgeCaseTest do
  use ExUnit.Case

  describe "__partitions_options__ edge cases" do
    test "with max_concurrency of 1" do
      opts = [max_concurrency: 1]
      result = ExBurn.Defn.Compiler.__partitions_options__(opts)
      assert length(result) == 1
      assert hd(result) == opts
    end

    test "with max_concurrency of 8" do
      opts = [max_concurrency: 8, garbage_collect: true]
      result = ExBurn.Defn.Compiler.__partitions_options__(opts)
      assert length(result) == 8
      assert Enum.all?(result, &(&1 == opts))
    end

    test "with garbage_collect option" do
      opts = [garbage_collect: true]
      result = ExBurn.Defn.Compiler.__partitions_options__(opts)
      assert length(result) == 1
      assert hd(result) == opts
    end

    test "with empty opts" do
      result = ExBurn.Defn.Compiler.__partitions_options__([])
      assert length(result) == 1
    end
  end

  describe "__to_backend__ edge cases" do
    test "always returns ExBurn.Backend regardless of opts" do
      assert ExBurn.Defn.Compiler.__to_backend__([]) == {ExBurn.Backend, []}
      assert ExBurn.Defn.Compiler.__to_backend__(compiler: :other) == {ExBurn.Backend, []}
      assert ExBurn.Defn.Compiler.__to_backend__(garbage_collect: true) == {ExBurn.Backend, []}
    end
  end

  describe "__compile__ edge cases" do
    test "returns a function" do
      fun = fn [x] -> Nx.add(x, x) end
      vars = [Nx.tensor([1.0, 2.0, 3.0])]
      compiled = ExBurn.Defn.Compiler.__compile__(:test_key, vars, fun, [])
      assert is_function(compiled, 1)
    end

    test "compiled function with multiple params" do
      fun = fn [x, y] -> Nx.add(x, y) end
      vars = [Nx.tensor([1.0]), Nx.tensor([2.0])]
      compiled = ExBurn.Defn.Compiler.__compile__(:test_key, vars, fun, [])
      assert is_function(compiled, 1)
    end

    test "compiled function with single param" do
      fun = fn [x] -> Nx.multiply(x, 2.0) end
      vars = [Nx.tensor([1.0, 2.0])]
      compiled = ExBurn.Defn.Compiler.__compile__(:test_key, vars, fun, [])
      assert is_function(compiled, 1)
    end
  end

  describe "__jit__ edge cases" do
    @tag :nif
    test "evaluates with single arg list" do
      fun = fn [x] -> Nx.add(x, x) end
      vars = [Nx.tensor([1.0, 2.0])]
      args_list = [[fn -> Nx.tensor([1.0, 2.0]) end]]

      [result] = ExBurn.Defn.Compiler.__jit__(:test_key, vars, fun, args_list, [])
      assert is_struct(result, Nx.Tensor)
    end

    @tag :nif
    test "evaluates with multiple arg lists" do
      fun = fn [x] -> Nx.multiply(x, 2.0) end
      vars = [Nx.tensor([1.0, 2.0])]

      args_list = [
        [fn -> Nx.tensor([1.0, 2.0]) end],
        [fn -> Nx.tensor([3.0, 4.0]) end]
      ]

      [r1, r2] = ExBurn.Defn.Compiler.__jit__(:test_key, vars, fun, args_list, [])
      assert is_struct(r1, Nx.Tensor)
      assert is_struct(r2, Nx.Tensor)
    end
  end

  describe "__shard_jit__ edge cases" do
    test "always raises RuntimeError" do
      assert_raise RuntimeError, "sharding is not supported by ExBurn.Defn.Compiler", fn ->
        ExBurn.Defn.Compiler.__shard_jit__(:key, nil, [], fn _ -> Nx.tensor(0) end, [[]], [])
      end
    end

    test "raises even with valid-looking args" do
      assert_raise RuntimeError, "sharding is not supported by ExBurn.Defn.Compiler", fn ->
        ExBurn.Defn.Compiler.__shard_jit__(
          :key,
          :mesh,
          [Nx.tensor([1.0])],
          fn [x] -> x end,
          [[fn -> Nx.tensor([1.0]) end]],
          []
        )
      end
    end
  end
end
