defmodule ExBurn.NifApiTest do
  use ExUnit.Case, async: false

  # ── Helpers ──────────────────────────────────────────────────────

  defp vec_ref(values) do
    shape = [length(values)]
    data = Enum.reduce(values, <<>>, fn v, acc -> <<acc::binary, v::float-32-native>> end)
    ExBurn.Nif.new_tensor(data, shape, "f32")
  end

  defp bytes_to_list(bin) do
    for <<v::float-32-native <- bin>>, do: v
  end

  describe "ExBurn.Nif dtype validation" do
    @tag :nif
    test "rejects unsupported atom dtypes with a structured error" do
      assert_raise ExBurn.Error,
                   ~r/dtype :f64 is not supported by the NIF yet/,
                   fn -> ExBurn.Nif.new_tensor(<<1.0::float-32-native>>, [1], :f64) end
    end

    @tag :nif
    test "rejects unsupported binary dtypes" do
      assert_raise ExBurn.Error, ~r/dtype "f16" is not supported/, fn ->
        ExBurn.Nif.zeros_tensor([2], "f16")
      end
    end

    @tag :nif
    test "accepts atom and binary spellings of f32" do
      a = ExBurn.Nif.new_tensor(<<1.0::float-32-native>>, [1], :f32)
      b = ExBurn.Nif.new_tensor(<<2.0::float-32-native>>, [1], "f32")
      assert byte_size(ExBurn.Nif.tensor_to_binary(a)) == 4
      assert byte_size(ExBurn.Nif.tensor_to_binary(b)) == 4
    end
  end

  describe "ExBurn.Nif tensor inspection" do
    @tag :nif
    test "tensor_dtype reports the stored dtype" do
      ref = vec_ref([1.0])
      assert ExBurn.Nif.tensor_dtype(ref) == "f32"
    end

    @tag :nif
    test "tensor_numel counts elements" do
      ref = vec_ref(List.duplicate(1.0, 7))
      assert ExBurn.Nif.tensor_numel(ref) == 7
    end
  end

  describe "ExBurn.Nif arithmetic coverage" do
    @tag :nif
    test "pow raises for non-float exponent but works with scalars" do
      ref = vec_ref([2.0])
      out = ExBurn.Nif.pow_tensor(ref, 3.0)
      assert bytes_to_list(ExBurn.Nif.tensor_to_binary(out)) == [8.0]

      assert_raise ArgumentError, fn -> ExBurn.Nif.pow_tensor(ref, ref) end
    end

    @tag :nif
    test "sigmoid and tanh transform values" do
      z = vec_ref([0.0])

      sig = ExBurn.Nif.sigmoid_tensor(z)
      assert_in_delta bytes_to_list(ExBurn.Nif.tensor_to_binary(sig)) |> hd(), 0.5, 1.0e-6

      tanh = ExBurn.Nif.tanh_tensor(z)
      assert_in_delta bytes_to_list(ExBurn.Nif.tensor_to_binary(tanh)) |> hd(), 0.0, 1.0e-6
    end

    @tag :nif
    test "max and min reduce to scalars" do
      r = vec_ref([3.0, 1.0, 2.0])

      max = ExBurn.Nif.max_tensor(r)
      assert bytes_to_list(ExBurn.Nif.tensor_to_binary(max)) == [3.0]

      min = ExBurn.Nif.min_tensor(r)
      assert bytes_to_list(ExBurn.Nif.tensor_to_binary(min)) == [1.0]
    end

    @tag :nif
    test "dot computes inner product" do
      a = vec_ref([1.0, 2.0])
      b = vec_ref([3.0, 4.0])

      dot = ExBurn.Nif.dot_tensor(a, b)
      assert bytes_to_list(ExBurn.Nif.tensor_to_binary(dot)) == [11.0]
    end

    @tag :nif
    test "broadcast and concat reshape data" do
      r = vec_ref([1.0, 2.0])

      broadcasted = ExBurn.Nif.broadcast_tensor(r, [2, 2])
      assert ExBurn.Nif.tensor_shape(broadcasted) == [2, 2]
      assert bytes_to_list(ExBurn.Nif.tensor_to_binary(broadcasted)) == [1.0, 2.0, 1.0, 2.0]

      c = ExBurn.Nif.concat_tensor(r, vec_ref([3.0]))
      assert bytes_to_list(ExBurn.Nif.tensor_to_binary(c)) == [1.0, 2.0, 3.0]
    end

    @tag :nif
    test "eye and iota create fresh tensors" do
      eye = ExBurn.Nif.eye_tensor(2, :f32)
      assert bytes_to_list(ExBurn.Nif.tensor_to_binary(eye)) == [1.0, 0.0, 0.0, 1.0]

      iota = ExBurn.Nif.iota_tensor([3], 0, :f32)
      assert bytes_to_list(ExBurn.Nif.tensor_to_binary(iota)) == [0.0, 1.0, 2.0]
    end

    @tag :nif
    test "empty_tensor allocates uninitialized storage" do
      empty = ExBurn.Nif.empty_tensor([4], :f32)
      assert ExBurn.Nif.tensor_shape(empty) == [4]
      assert byte_size(ExBurn.Nif.tensor_to_binary(empty)) == 16
    end

    @tag :nif
    test "debug_nif_count reports registered NIFs" do
      count = ExBurn.Nif.debug_nif_count()
      assert is_integer(count) and count > 10
    end
  end

  describe "ExBurn.NifHelper wrappers" do
    @tag :nif
    test "creation wrappers return ok tuples" do
      {:ok, zeros} = ExBurn.NifHelper.zeros_tensor([2], :f32)
      {:ok, ones} = ExBurn.NifHelper.ones_tensor([2], :f32)
      {:ok, eye} = ExBurn.NifHelper.eye_tensor(3, :f64)
      {:ok, iota} = ExBurn.NifHelper.iota_tensor([4], 0, :f16)

      assert bytes_to_list(ExBurn.Nif.tensor_to_binary(zeros)) == [0.0, 0.0]
      assert bytes_to_list(ExBurn.Nif.tensor_to_binary(ones)) == [1.0, 1.0]
      assert ExBurn.Nif.tensor_shape(eye) == [3, 3]
      assert bytes_to_list(ExBurn.Nif.tensor_to_binary(iota)) == [0.0, 1.0, 2.0, 3.0]
    end

    @tag :nif
    test "guard converts raised NIF errors into error tuples" do
      assert {:error, _} = ExBurn.NifHelper.exp_tensor(make_ref())
      assert {:error, _} = ExBurn.NifHelper.tensor_shape(:not_a_resource)
      assert {:error, _} = ExBurn.NifHelper.tensor_to_binary("nope")
    end

    @tag :nif
    test "unary and reduction wrappers compute values" do
      a = vec_ref([1.0, 4.0])

      checks = [
        {:exp_tensor, [a]},
        {:log_tensor, [vec_ref([1.0])]},
        {:sqrt_tensor, [vec_ref([9.0])]},
        {:sigmoid_tensor, [a]},
        {:tanh_tensor, [a]},
        {:relu_tensor, [vec_ref([-1.0, 2.0])]}
      ]

      for {fun, args} <- checks do
        assert {:ok, _ref} = apply(ExBurn.NifHelper, fun, args), "expected #{fun} to succeed"
      end

      {:ok, mean} = ExBurn.NifHelper.mean_tensor(a)
      assert_in_delta bytes_to_list(ExBurn.Nif.tensor_to_binary(mean)) |> hd(), 2.5, 1.0e-6
    end

    @tag :nif
    test "linear algebra wrappers" do
      a = vec_ref([1.0, 2.0, 3.0, 4.0])
      {:ok, a2d} = ExBurn.NifHelper.reshape_tensor(a, [2, 2])

      {:ok, m} = ExBurn.NifHelper.matmul_tensor(a2d, a2d)
      assert ExBurn.Nif.tensor_shape(m) == [2, 2]
      # [1 2; 3 4] * [1 2; 3 4] = [7 10; 15 22]
      m_data =
        ExBurn.Nif.tensor_to_binary(m)
        |> bytes_to_list()

      assert m_data == [7.0, 10.0, 15.0, 22.0]

      {:ok, t} = ExBurn.NifHelper.transpose_tensor(a2d)
      assert is_list(ExBurn.Nif.tensor_shape(t))

      {:ok, dot} = ExBurn.NifHelper.dot_tensor(a, a)
      assert is_reference(dot)

      assert match?({:ok, _}, ExBurn.NifHelper.reshape_tensor(m, [4]))

      # Rust broadcast expands an existing leading dimension
      two = vec_ref([5.0, 6.0])
      {:ok, a_1x2} = ExBurn.NifHelper.reshape_tensor(two, [1, 2])
      assert match?({:ok, _}, ExBurn.NifHelper.broadcast_tensor(a_1x2, [2, 2]))
      assert match?({:ok, _}, ExBurn.NifHelper.concat_tensor(a, a))
    end

    @tag :nif
    test "device info helpers return usable values" do
      assert is_boolean(ExBurn.NifHelper.gpu_available())

      name = ExBurn.NifHelper.device_name()
      assert is_binary(name) and name != ""

      t = vec_ref([1.0])
      {:ok, gpu} = ExBurn.NifHelper.to_gpu(t)
      {:ok, back} = ExBurn.NifHelper.to_cpu(gpu)
      assert bytes_to_list(ExBurn.Nif.tensor_to_binary(back)) == [1.0]
    end

    @tag :nif
    test "free_tensor releases resources" do
      ref = vec_ref([1.0])
      assert :ok == ExBurn.NifHelper.free_tensor(ref)
    end

    @tag :nif
    test "neural network wrappers run" do
      logits = vec_ref([1.0, 2.0, 3.0])

      {:ok, soft} = ExBurn.NifHelper.softmax_tensor(logits, -1)
      probs = bytes_to_list(ExBurn.Nif.tensor_to_binary(soft))
      assert_in_delta Enum.sum(probs), 1.0, 1.0e-5

      {:ok, normed} = ExBurn.NifHelper.layer_norm_tensor(logits, -1, 1.0e-5)
      assert bytes_to_list(ExBurn.Nif.tensor_to_binary(normed)) |> length() == 3
    end

    @tag :nif
    test "loss and dropout wrappers" do
      pred = vec_ref([1.0, 0.0])
      target = vec_ref([1.0, 0.0])

      assert match?({:ok, _}, ExBurn.NifHelper.cross_entropy_loss(pred, target))
      assert match?({:ok, _}, ExBurn.NifHelper.mse_loss(pred, target))

      {:ok, dropped} = ExBurn.NifHelper.dropout(pred, 0.0)
      # zero dropout keeps every element
      assert bytes_to_list(ExBurn.Nif.tensor_to_binary(dropped)) == [1.0, 0.0]
    end

    @tag :nif
    test "backward/grad surface their current state" do
      # backward_tensor currently returns :ok (no gradient graph yet);
      # grad_tensor returns a reference or an error tuple depending on the
      # NIF build. Both must not crash the VM.
      result = ExBurn.NifHelper.backward_tensor(vec_ref([1.0]))
      assert result == :ok or match?({:error, _}, result) or match?({:ok, _}, result)

      grad = ExBurn.NifHelper.grad_tensor(vec_ref([1.0]), vec_ref([1.0]))
      assert match?({:error, _}, grad) or match?({:ok, _}, grad)
    end

    @tag :nif
    test "random_tensor generates in-range uniforms and rejects bad dtypes" do
      {:ok, ref} = ExBurn.NifHelper.random_tensor([50], "f32", -1.0, 1.0)
      values = bytes_to_list(ExBurn.Nif.tensor_to_binary(ref))
      assert length(values) == 50
      assert Enum.all?(values, fn v -> v >= -1.0 and v < 1.0 end)

      assert {:error, "unknown dtype \"i8\""} =
               ExBurn.NifHelper.random_tensor([2], "i8", 0.0, 1.0)
    end

    @tag :nif
    test "conv2d_tensor runs an NHWC convolution through Nx" do
      input = vec_ref(List.duplicate(1.0, 16))

      {:ok, input4d} = ExBurn.NifHelper.reshape_tensor(input, [1, 1, 4, 4])
      kernel = vec_ref(List.duplicate(1.0, 4))

      {:ok, kernel4d} = ExBurn.NifHelper.reshape_tensor(kernel, [1, 1, 2, 2])

      assert {:ok, out} =
               ExBurn.NifHelper.conv2d_tensor(input4d, kernel4d, [1, 1], [{0, 0}, {0, 0}])

      assert ExBurn.Nif.tensor_shape(out) == [1, 1, 3, 3]
      assert bytes_to_list(ExBurn.Nif.tensor_to_binary(out)) == List.duplicate(4.0, 9)
    end
  end
end
