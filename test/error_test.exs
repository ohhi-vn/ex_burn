defmodule ExBurn.ErrorTest do
  use ExUnit.Case

  describe "message formatting" do
    test "formats without details" do
      error = ExBurn.Error.exception(op: :test, reason: "something failed")
      assert Exception.message(error) == "ExBurn.test: something failed"
    end

    test "formats with details" do
      error =
        ExBurn.Error.exception(
          op: :matmul,
          reason: "shape mismatch",
          details: %{lhs: [3, 4], rhs: [5, 6]}
        )

      msg = Exception.message(error)
      assert msg =~ "ExBurn.matmul: shape mismatch"
      assert msg =~ "[3, 4]"
    end

    test "formats with nil details" do
      error = ExBurn.Error.exception(op: :conv, reason: "kernel too large", details: nil)
      assert Exception.message(error) == "ExBurn.conv: kernel too large"
    end
  end

  describe "exception struct" do
    test "stores fields" do
      error = ExBurn.Error.exception(op: :add, reason: "bad input", details: %{x: 1})
      assert error.op == :add
      assert error.reason == "bad input"
      assert error.details == %{x: 1}
    end
  end
end
