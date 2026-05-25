defmodule ExBurn.BackendTest do
  use ExUnit.Case

  setup do
    previous = Nx.default_backend()
    Nx.default_backend(ExBurn.Backend)
    on_exit(fn -> Nx.default_backend(previous) end)
    :ok
 end

  describe "tensor creation" do
    @tag :nif
    test "from binary" do
      t = Nx.tensor([1.0, 2.0, 3.0])
      assert Nx.to_list(t) == [1.0, 2.0, 3.0]
    end
  end

  describe "element-wise arithmetic" do
    @tag :nif
    test "addition" do
      a = Nx.tensor([1.0, 2.0, 3.0])
      b = Nx.tensor([4.0, 5.0, 6.0])
      assert Nx.to_list(Nx.add(a, b)) == [5.0, 7.0, 9.0]
    end

    @tag :nif
    test "subtraction" do
      a = Nx.tensor([4.0, 5.0, 6.0])
      b = Nx.tensor([1.0, 2.0, 3.0])
      assert Nx.to_list(Nx.subtract(a, b)) == [3.0, 3.0, 3.0]
    end

    @tag :nif
    test "multiplication" do
      a = Nx.tensor([1.0, 2.0, 3.0])
      b = Nx.tensor([4.0, 5.0, 6.0])
      assert Nx.to_list(Nx.multiply(a, b)) == [4.0, 10.0, 18.0]
    end

    @tag :nif
    test "division" do
      a = Nx.tensor([4.0, 10.0, 18.0])
      b = Nx.tensor([2.0, 5.0, 6.0])
      assert Nx.to_list(Nx.divide(a, b)) == [2.0, 2.0, 3.0]
    end

    @tag :nif
    test "negation" do
      a = Nx.tensor([1.0, -2.0, 3.0])
      assert Nx.to_list(Nx.negate(a)) == [-1.0, 2.0, -3.0]
    end

    @tag :nif
    test "absolute value" do
      a = Nx.tensor([-1.0, 2.0, -3.0])
      assert Nx.to_list(Nx.abs(a)) == [1.0, 2.0, 3.0]
    end
  end

  describe "2D tensor operations" do
    @tag :nif
    test "addition" do
      a = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      b = Nx.tensor([[5.0, 6.0], [7.0, 8.0]])
      assert Nx.to_list(Nx.add(a, b)) == [6.0, 8.0, 10.0, 12.0]
    end

    @tag :nif
    test "transpose" do
      a = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
      assert Nx.shape(Nx.transpose(a)) == {3, 2}
    end

    @tag :nif
    test "reshape" do
      a = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
      assert Nx.shape(Nx.reshape(a, {2, 3})) == {2, 3}
    end
  end

  describe "reductions" do
    @tag :nif
    test "sum" do
      a = Nx.tensor([1.0, 2.0, 3.0])
      assert Nx.to_list(Nx.sum(a)) == [6.0]
    end

    @tag :nif
    test "mean" do
      a = Nx.tensor([1.0, 2.0, 3.0, 4.0])
      [val] = Nx.to_list(Nx.mean(a))
      assert_in_delta val, 2.5, 1.0e-6
    end
  end
end
