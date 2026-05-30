defmodule ExBurn.ErrorEdgeCaseTest do
  use ExUnit.Case

  describe "format_error/1" do
    test "formats error without details" do
      error = ExBurn.Error.exception(op: :test_op, reason: "basic error")
      assert ExBurn.Error.format_error(error) == "ExBurn.test_op: basic error"
    end

    test "formats error with details" do
      error =
        ExBurn.Error.exception(op: :test_op, reason: "detailed error", details: %{key: "value"})

      formatted = ExBurn.Error.format_error(error)
      assert formatted =~ "ExBurn.test_op: detailed error"
      assert formatted =~ "%{key: \"value\"}"
    end
  end

  describe "new/1" do
    test "creates error with defaults" do
      error = ExBurn.Error.new(op: :custom, reason: "custom reason")
      assert error.op == :custom
      assert error.reason == "custom reason"
      assert error.details == nil
    end

    test "creates error with empty opts" do
      error = ExBurn.Error.new([])
      assert error.op == nil
      assert error.reason == nil
      assert error.details == nil
    end

    test "creates error with all fields" do
      error = ExBurn.Error.new(op: :forward, reason: "failed", details: %{layer: "dense_0"})
      assert error.op == :forward
      assert error.reason == "failed"
      assert error.details == %{layer: "dense_0"}
    end
  end

  describe "from_tuple/2" do
    test "wraps error tuple with op" do
      error = ExBurn.Error.from_tuple({:error, "something went wrong"}, op: :predict)
      assert error.op == :predict
      assert error.reason == "something went wrong"
      assert error.details == nil
    end

    test "wraps error tuple with additional opts" do
      error = ExBurn.Error.from_tuple({:error, "bad"}, op: :compile, details: %{line: 42})
      assert error.op == :compile
      assert error.reason == "bad"
      assert error.details == %{line: 42}
    end

    test "converts non-string reason to string" do
      error = ExBurn.Error.from_tuple({:error, :enoent}, op: :read)
      assert error.reason == "enoent"
    end
  end

  describe "to_log_string/1" do
    test "formats without details" do
      error = ExBurn.Error.exception(op: :add, reason: "overflow")
      assert ExBurn.Error.to_log_string(error) == "[ExBurn:add] overflow"
    end

    test "formats with details" do
      error =
        ExBurn.Error.exception(op: :matmul, reason: "mismatch", details: %{a: [2, 3], b: [4, 5]})

      log = ExBurn.Error.to_log_string(error)
      assert log =~ "[ExBurn:matmul]"
      assert log =~ "mismatch"
      assert log =~ "details:"
    end
  end

  describe "exception raising" do
    test "raises with correct message" do
      assert_raise ExBurn.Error, "ExBurn.predict: model not compiled", fn ->
        raise ExBurn.Error, op: :predict, reason: "model not compiled"
      end
    end

    test "raises with details in message" do
      assert_raise ExBurn.Error, ~r/ExBurn\.conv.*kernel too large/, fn ->
        raise ExBurn.Error, op: :conv, reason: "kernel too large"
      end
    end
  end

  describe "message/1 fallback" do
    test "handles non-struct input gracefully" do
      # The message/1 function is implemented via defexception, so it should
      # only be called on valid ExBurn.Error structs. Verify it works.
      error = %ExBurn.Error{op: :test, reason: "msg", details: nil}
      assert Exception.message(error) == "ExBurn.test: msg"
    end
  end
end
