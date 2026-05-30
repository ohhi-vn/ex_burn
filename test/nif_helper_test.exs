defmodule ExBurn.NifHelperTest do
  use ExUnit.Case

  describe "GPU availability" do
    @tag :nif
    test "gpu_available returns boolean" do
      assert is_boolean(ExBurn.NifHelper.gpu_available())
    end

    @tag :nif
    test "device_name returns a string" do
      assert is_binary(ExBurn.NifHelper.device_name())
    end
  end

  describe "tensor creation" do
    @tag :nif
    test "new_tensor returns ok tuple" do
      result = ExBurn.NifHelper.new_tensor(<<1.0::float-32-native>>, [1], "f32")
      assert match?({:ok, _}, result)
    end

    @tag :nif
    test "zeros_tensor returns ok tuple" do
      result = ExBurn.NifHelper.zeros_tensor([2, 3], "f32")
      assert match?({:ok, _}, result)
    end

    @tag :nif
    test "ones_tensor returns ok tuple" do
      result = ExBurn.NifHelper.ones_tensor([3], "f32")
      assert match?({:ok, _}, result)
    end
  end

  describe "tensor inspection" do
    @tag :nif
    test "tensor_shape returns ok tuple" do
      {:ok, ref} = ExBurn.NifHelper.zeros_tensor([2, 3], "f32")
      result = ExBurn.NifHelper.tensor_shape(ref)
      assert match?({:ok, _}, result)
    end

    @tag :nif
    test "tensor_to_binary returns ok tuple" do
      {:ok, ref} = ExBurn.NifHelper.new_tensor(<<1.0::float-32-native>>, [1], "f32")
      result = ExBurn.NifHelper.tensor_to_binary(ref)
      assert match?({:ok, _}, result)
    end
  end

  describe "arithmetic dispatch" do
    @tag :nif
    test "add_tensor returns ok tuple" do
      {:ok, a} = ExBurn.NifHelper.new_tensor(<<1.0::float-32-native>>, [1], "f32")
      {:ok, b} = ExBurn.NifHelper.new_tensor(<<2.0::float-32-native>>, [1], "f32")
      result = ExBurn.NifHelper.add_tensor(a, b)
      assert match?({:ok, _}, result)
    end

    @tag :nif
    test "mul_tensor returns ok tuple" do
      {:ok, a} = ExBurn.NifHelper.new_tensor(<<2.0::float-32-native>>, [1], "f32")
      {:ok, b} = ExBurn.NifHelper.new_tensor(<<3.0::float-32-native>>, [1], "f32")
      result = ExBurn.NifHelper.mul_tensor(a, b)
      assert match?({:ok, _}, result)
    end
  end

  describe "neural network operations" do
    @tag :nif
    test "softmax_tensor returns ok tuple" do
      {:ok, a} =
        ExBurn.NifHelper.new_tensor(<<1.0::float-32-native, 2.0::float-32-native>>, [2], "f32")

      result = ExBurn.NifHelper.softmax_tensor(a, 0)
      assert match?({:ok, _}, result)
    end

    @tag :nif
    test "layer_norm_tensor returns ok tuple" do
      {:ok, a} =
        ExBurn.NifHelper.new_tensor(<<1.0::float-32-native, 2.0::float-32-native>>, [2], "f32")

      result = ExBurn.NifHelper.layer_norm_tensor(a, 0, 1.0e-5)
      assert match?({:ok, _}, result)
    end

    @tag :nif
    test "layer_norm_tensor with negative dim" do
      {:ok, a} =
        ExBurn.NifHelper.new_tensor(<<1.0::float-32-native, 2.0::float-32-native>>, [2], "f32")

      result = ExBurn.NifHelper.layer_norm_tensor(a, -1, 1.0e-5)
      assert match?({:ok, _}, result)
    end

    @tag :nif
    test "layer_norm_tensor with custom epsilon" do
      {:ok, a} =
        ExBurn.NifHelper.new_tensor(<<1.0::float-32-native, 2.0::float-32-native>>, [2], "f32")

      result = ExBurn.NifHelper.layer_norm_tensor(a, 0, 0.01)
      assert match?({:ok, _}, result)
    end
  end

  describe "loss functions" do
    @tag :nif
    test "cross_entropy_loss returns ok tuple" do
      pred =
        <<1.0::float-32-native, 2.0::float-32-native, 3.0::float-32-native, 4.0::float-32-native,
          5.0::float-32-native, 6.0::float-32-native>>

      {:ok, pred_ref} = ExBurn.NifHelper.new_tensor(pred, [2, 3], "f32")

      target =
        <<0.0::float-32-native, 0.0::float-32-native, 1.0::float-32-native, 1.0::float-32-native,
          0.0::float-32-native, 0.0::float-32-native>>

      {:ok, target_ref} = ExBurn.NifHelper.new_tensor(target, [2, 3], "f32")
      result = ExBurn.NifHelper.cross_entropy_loss(pred_ref, target_ref)
      assert match?({:ok, _}, result)
    end

    @tag :nif
    test "mse_loss returns ok tuple" do
      {:ok, pred} =
        ExBurn.NifHelper.new_tensor(
          <<1.0::float-32-native, 2.0::float-32-native, 3.0::float-32-native>>,
          [3],
          "f32"
        )

      {:ok, target} =
        ExBurn.NifHelper.new_tensor(
          <<1.0::float-32-native, 2.0::float-32-native, 3.0::float-32-native>>,
          [3],
          "f32"
        )

      result = ExBurn.NifHelper.mse_loss(pred, target)
      assert match?({:ok, _}, result)
    end

    @tag :nif
    test "mse_loss with identical tensors returns near-zero" do
      {:ok, pred} =
        ExBurn.NifHelper.new_tensor(
          <<1.0::float-32-native, 2.0::float-32-native, 3.0::float-32-native>>,
          [3],
          "f32"
        )

      {:ok, target} =
        ExBurn.NifHelper.new_tensor(
          <<1.0::float-32-native, 2.0::float-32-native, 3.0::float-32-native>>,
          [3],
          "f32"
        )

      {:ok, loss_ref} = ExBurn.NifHelper.mse_loss(pred, target)
      {:ok, loss_binary} = ExBurn.NifHelper.tensor_to_binary(loss_ref)
      <<loss_val::float-32-native>> = loss_binary
      assert_in_delta loss_val, 0.0, 1.0e-5
    end

    @tag :nif
    test "mse_loss with different tensors returns positive" do
      {:ok, pred} =
        ExBurn.NifHelper.new_tensor(<<1.0::float-32-native, 2.0::float-32-native>>, [2], "f32")

      {:ok, target} =
        ExBurn.NifHelper.new_tensor(<<3.0::float-32-native, 4.0::float-32-native>>, [2], "f32")

      {:ok, loss_ref} = ExBurn.NifHelper.mse_loss(pred, target)
      {:ok, loss_binary} = ExBurn.NifHelper.tensor_to_binary(loss_ref)
      <<loss_val::float-32-native>> = loss_binary
      assert loss_val > 0.0
    end
  end

  describe "dropout" do
    @tag :nif
    test "dropout returns ok tuple" do
      {:ok, a} =
        ExBurn.NifHelper.new_tensor(
          <<1.0::float-32-native, 2.0::float-32-native, 3.0::float-32-native>>,
          [3],
          "f32"
        )

      result = ExBurn.NifHelper.dropout(a, 0.5)
      assert match?({:ok, _}, result)
    end

    @tag :nif
    test "dropout with 2D tensor" do
      data =
        <<1.0::float-32-native, 2.0::float-32-native, 3.0::float-32-native, 4.0::float-32-native>>

      {:ok, a} = ExBurn.NifHelper.new_tensor(data, [2, 2], "f32")
      result = ExBurn.NifHelper.dropout(a, 0.3)
      assert match?({:ok, _}, result)
    end

    @tag :nif
    test "dropout with zero probability" do
      {:ok, a} =
        ExBurn.NifHelper.new_tensor(<<1.0::float-32-native, 2.0::float-32-native>>, [2], "f32")

      result = ExBurn.NifHelper.dropout(a, 0.0)
      assert match?({:ok, _}, result)
    end
  end

  describe "autodiff" do
    @tag :nif
    test "backward_tensor returns ok" do
      {:ok, a} = ExBurn.NifHelper.new_tensor(<<1.0::float-32-native>>, [1], "f32")
      assert ExBurn.NifHelper.backward_tensor(a) == :ok
    end

    @tag :nif
    test "grad_tensor returns ok tuple" do
      {:ok, a} =
        ExBurn.NifHelper.new_tensor(<<1.0::float-32-native, 2.0::float-32-native>>, [2], "f32")

      result = ExBurn.NifHelper.grad_tensor(a, a)
      assert match?({:ok, _}, result)
    end

    @tag :nif
    test "backward then grad returns gradient" do
      # Create a simple computation: y = x^2, dy/dx = 2x
      {:ok, x} = ExBurn.NifHelper.new_tensor(<<3.0::float-32-native>>, [1], "f32")
      {:ok, x_squared} = ExBurn.NifHelper.mul_tensor(x, x)

      # Trigger backward pass
      ExBurn.NifHelper.backward_tensor(x_squared)

      # Extract gradient
      {:ok, grad} = ExBurn.NifHelper.grad_tensor(x, x)
      {:ok, grad_binary} = ExBurn.NifHelper.tensor_to_binary(grad)
      <<grad_val::float-32-native>> = grad_binary

      # dy/dx = 2*3 = 6
      assert_in_delta grad_val, 6.0, 0.1
    end
  end

  describe "memory management" do
    @tag :nif
    test "free_tensor returns ok" do
      {:ok, ref} = ExBurn.NifHelper.zeros_tensor([2], "f32")
      assert ExBurn.NifHelper.free_tensor(ref) == :ok
    end
  end
end
