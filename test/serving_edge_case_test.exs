defmodule ExBurn.ServingEdgeCaseTest do
  use ExUnit.Case

  describe "new/2" do
    test "creates serving with defaults" do
      model = ExBurn.Model.new()
      serving = ExBurn.Serving.new(model)
      assert serving.batch_size == 32
      assert serving.batch_timeout == 50
      assert serving.padding == false
    end

    test "creates serving with custom options" do
      model = ExBurn.Model.new()
      serving = ExBurn.Serving.new(model, batch_size: 64, batch_timeout: 100, padding: true)
      assert serving.batch_size == 64
      assert serving.batch_timeout == 100
      assert serving.padding == true
    end

    test "partitions defaults to scheduler count" do
      model = ExBurn.Model.new()
      serving = ExBurn.Serving.new(model)
      assert serving.partitions == System.schedulers_online()
    end

    test "partitions can be overridden" do
      model = ExBurn.Model.new()
      serving = ExBurn.Serving.new(model, partitions: 2)
      assert serving.partitions == 2
    end
  end

  describe "build/2" do
    test "builds Nx.Serving" do
      model = ExBurn.Model.new()
      serving = ExBurn.Serving.build(model)
      assert is_struct(serving, Nx.Serving)
    end

    test "builds with custom options" do
      model = ExBurn.Model.new()
      serving = ExBurn.Serving.build(model, batch_size: 16, batch_timeout: 200)
      assert is_struct(serving, Nx.Serving)
    end
  end

  describe "status/1" do
    test "returns status map" do
      model = ExBurn.Model.new()
      serving = ExBurn.Serving.new(model, batch_size: 16, batch_timeout: 200, padding: true)
      status = ExBurn.Serving.status(serving)

      assert status.batch_size == 16
      assert status.batch_timeout == 200
      assert status.padding == true
      assert is_integer(status.partitions)
    end
  end

  describe "with_batch_size/2" do
    test "returns new serving with updated batch size" do
      model = ExBurn.Model.new()
      serving = ExBurn.Serving.new(model, batch_size: 32)
      updated = ExBurn.Serving.with_batch_size(serving, 64)

      assert updated.batch_size == 64
      assert updated.batch_timeout == serving.batch_timeout
      assert updated.partitions == serving.partitions
      assert updated.padding == serving.padding
    end
  end

  describe "with_timeout/2" do
    test "returns new serving with updated timeout" do
      model = ExBurn.Model.new()
      serving = ExBurn.Serving.new(model, batch_timeout: 50)
      updated = ExBurn.Serving.with_timeout(serving, 200)

      assert updated.batch_timeout == 200
      assert updated.batch_size == serving.batch_size
      assert updated.partitions == serving.partitions
      assert updated.padding == serving.padding
    end
  end
end
