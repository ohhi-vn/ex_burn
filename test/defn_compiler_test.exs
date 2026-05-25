defmodule ExBurn.Defn.CompilerTest do
  use ExUnit.Case

  describe "Nx.Defn.Compiler behaviour callbacks" do
    test "__to_backend__ returns ExBurn.Backend" do
      assert ExBurn.Defn.Compiler.__to_backend__([]) == {ExBurn.Backend, []}
    end

    test "__to_backend__ ignores options" do
      assert ExBurn.Defn.Compiler.__to_backend__(compiler: SomeOther) == {ExBurn.Backend, []}
    end

    test "__partitions_options__ duplicates opts" do
      opts = [garbage_collect: true]
      assert ExBurn.Defn.Compiler.__partitions_options__(opts) == [opts]
    end

    test "__partitions_options__ respects max_concurrency" do
      opts = [max_concurrency: 4]
      result = ExBurn.Defn.Compiler.__partitions_options__(opts)
      assert length(result) == 4
    end

    test "__partitions_options__ defaults to 1 partition" do
      assert length(ExBurn.Defn.Compiler.__partitions_options__([])) == 1
    end
  end

  describe "__compile__ returns a callable" do
    test "compiled function accepts params list" do
      fun = fn [x] -> Nx.add(x, x) end
      vars = [Nx.tensor([1.0, 2.0, 3.0])]

      compiled = ExBurn.Defn.Compiler.__compile__(:test_key, vars, fun, [])
      assert is_function(compiled, 1)
    end
  end

  describe "__jit__ evaluates expressions" do
    @tag :nif
    test "basic arithmetic" do
      fun = fn [x, y] -> Nx.add(x, y) end
      vars = [Nx.tensor([1.0, 2.0]), Nx.tensor([3.0, 4.0])]
      args_list = [[fn -> Nx.tensor([1.0, 2.0]) end, fn -> Nx.tensor([3.0, 4.0]) end]]

      [result] = ExBurn.Defn.Compiler.__jit__(:test_key, vars, fun, args_list, [])
      assert is_struct(result, Nx.Tensor)
    end
  end

  describe "__shard_jit__ raises" do
    test "sharding is not supported" do
      assert_raise RuntimeError, "sharding is not supported by ExBurn.Defn.Compiler", fn ->
        ExBurn.Defn.Compiler.__shard_jit__(:key, nil, [], fn _ -> Nx.tensor(0) end, [[]], [])
      end
    end
  end

  describe "defn integration" do
    @tag :nif
    test "simple defn function runs through ExBurn compiler" do
      defmodule SimpleMath do
        import Nx.Defn

        defn double(x) do
          Nx.multiply(x, 2.0)
        end
      end

      result = SimpleMath.double(Nx.tensor([1.0, 2.0, 3.0]))
      assert Nx.to_list(result) == [2.0, 4.0, 6.0]
    end
  end
end
