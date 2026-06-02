# BurnBridge Direct Tensor Operations with ExBurn
# Run with: mix run examples/burn_bridge_ops.exs
#
# Demonstrates the ExBurn.BurnBridge module for direct Burn tensor
# operations, bypassing the Nx abstraction layer:
#   1. Creating tensors (zeros, ones, rand, from_nx)
#   2. Arithmetic operations (add, sub, mul, div, neg, abs)
#   3. Math functions (exp, log, sqrt, sigmoid, relu)
#   4. Linear algebra (matmul, transpose, dot)
#   5. Reductions (sum, mean)
#   6. Shape manipulation (reshape, softmax)
#   7. Loss functions (mse, cross_entropy)
#   8. Device management (gpu_available?)
#   9. Converting between Burn tensors and Nx tensors

Mix.install([
  {:nx, "~> 0.7"},
  {:axon, "~> 0.7"},
  {:ex_burn, path: Path.expand("..", __DIR__)}
])

defmodule BurnBridgeOps do
  @moduledoc """
  Showcases ExBurn.BurnBridge for direct tensor operations.
  """

  def run do
    IO.puts("=== BurnBridge Direct Tensor Operations ===\n")

    # 1. Device info
    IO.puts("Device:")
    IO.puts("  GPU available: #{ExBurn.BurnBridge.gpu_available?()}")
    IO.puts("  Device name:   #{ExBurn.BurnBridge.device_name()}")
    IO.puts("  Device info:   #{inspect(ExBurn.BurnBridge.device_info())}\n")

    # 2. Tensor creation
    IO.puts("Tensor creation:")

    zeros = ExBurn.BurnBridge.zeros([2, 3], :f32)

    IO.puts(
      "  zeros([2,3]): shape=#{inspect(ExBurn.Tensor.shape(zeros))}, type=#{ExBurn.Tensor.type(zeros)}"
    )

    ones = ExBurn.BurnBridge.ones([2, 3], :f32)

    IO.puts(
      "  ones([2,3]):  shape=#{inspect(ExBurn.Tensor.shape(ones))}, type=#{ExBurn.Tensor.type(ones)}"
    )

    rand = ExBurn.BurnBridge.rand([2, 3], :f32, 0.0, 1.0)

    IO.puts(
      "  rand([2,3]):  shape=#{inspect(ExBurn.Tensor.shape(rand))}, type=#{ExBurn.Tensor.type(rand)}"
    )

    nx_tensor = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
    from_nx = ExBurn.BurnBridge.from_nx(nx_tensor)
    IO.puts("  from_nx([[1,2],[3,4]]): shape=#{inspect(ExBurn.Tensor.shape(from_nx))}")

    # 3. Arithmetic
    IO.puts("\nArithmetic:")
    a = ExBurn.BurnBridge.from_nx(Nx.tensor([[1.0, 2.0], [3.0, 4.0]]))
    b = ExBurn.BurnBridge.from_nx(Nx.tensor([[5.0, 6.0], [7.0, 8.0]]))

    add_result = ExBurn.BurnBridge.add(a, b) |> ExBurn.BurnBridge.to_nx() |> Nx.to_flat_list()
    IO.puts("  add: #{inspect(add_result)}")

    sub_result = ExBurn.BurnBridge.sub(a, b) |> ExBurn.BurnBridge.to_nx() |> Nx.to_flat_list()
    IO.puts("  sub: #{inspect(sub_result)}")

    mul_result = ExBurn.BurnBridge.mul(a, b) |> ExBurn.BurnBridge.to_nx() |> Nx.to_flat_list()
    IO.puts("  mul (element-wise): #{inspect(mul_result)}")

    div_result = ExBurn.BurnBridge.div(a, b) |> ExBurn.BurnBridge.to_nx() |> Nx.to_flat_list()
    IO.puts("  div (element-wise): #{inspect(div_result)}")

    neg_result = ExBurn.BurnBridge.neg(a) |> ExBurn.BurnBridge.to_nx() |> Nx.to_flat_list()
    IO.puts("  neg: #{inspect(neg_result)}")

    abs_result =
      ExBurn.BurnBridge.abs(ExBurn.BurnBridge.neg(a))
      |> ExBurn.BurnBridge.to_nx()
      |> Nx.to_flat_list()

    IO.puts("  abs(neg(a)): #{inspect(abs_result)}")

    # 4. Math functions
    IO.puts("\nMath functions:")
    c = ExBurn.BurnBridge.from_nx(Nx.tensor([[1.0, 2.0, 3.0]]))

    exp_result = ExBurn.BurnBridge.exp(c) |> ExBurn.BurnBridge.to_nx() |> Nx.to_flat_list()
    IO.puts("  exp([1,2,3]): #{inspect(exp_result)}")

    log_result = ExBurn.BurnBridge.log(c) |> ExBurn.BurnBridge.to_nx() |> Nx.to_flat_list()
    IO.puts("  log([1,2,3]): #{inspect(log_result)}")

    sqrt_result = ExBurn.BurnBridge.sqrt(c) |> ExBurn.BurnBridge.to_nx() |> Nx.to_flat_list()
    IO.puts("  sqrt([1,2,3]): #{inspect(sqrt_result)}")

    sigmoid_result =
      ExBurn.BurnBridge.sigmoid(c) |> ExBurn.BurnBridge.to_nx() |> Nx.to_flat_list()

    IO.puts("  sigmoid([1,2,3]): #{inspect(sigmoid_result)}")

    relu_result =
      ExBurn.BurnBridge.relu(ExBurn.BurnBridge.from_nx(Nx.tensor([[-1.0, 0.0, 1.0]])))
      |> ExBurn.BurnBridge.to_nx()
      |> Nx.to_flat_list()

    IO.puts("  relu([-1,0,1]): #{inspect(relu_result)}")

    # 5. Linear algebra
    IO.puts("\nLinear algebra:")
    m1 = ExBurn.BurnBridge.from_nx(Nx.tensor([[1.0, 2.0], [3.0, 4.0]]))
    m2 = ExBurn.BurnBridge.from_nx(Nx.tensor([[5.0, 6.0], [7.0, 8.0]]))

    matmul_result =
      ExBurn.BurnBridge.matmul(m1, m2) |> ExBurn.BurnBridge.to_nx() |> Nx.to_flat_list()

    IO.puts("  matmul([[1,2],[3,4]], [[5,6],[7,8]]): #{inspect(matmul_result)}")

    transpose_result =
      ExBurn.BurnBridge.transpose(m1) |> ExBurn.BurnBridge.to_nx() |> Nx.to_flat_list()

    IO.puts("  transpose([[1,2],[3,4]]): #{inspect(transpose_result)}")

    v1 = Nx.tensor([1.0, 2.0, 3.0])
    v2 = Nx.tensor([4.0, 5.0, 6.0])
    dot_result = Nx.dot(v1, v2) |> Nx.to_number()
    IO.puts("  dot([1,2,3], [4,5,6]): #{dot_result}")

    # 6. Reductions
    IO.puts("\nReductions:")
    d = ExBurn.BurnBridge.from_nx(Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

    sum_result = ExBurn.BurnBridge.sum(d) |> ExBurn.BurnBridge.to_nx() |> Nx.to_number()
    IO.puts("  sum([[1,2,3],[4,5,6]]): #{sum_result}")

    mean_result = ExBurn.BurnBridge.mean(d) |> ExBurn.BurnBridge.to_nx() |> Nx.to_number()
    IO.puts("  mean([[1,2,3],[4,5,6]]): #{mean_result}")

    # 7. Shape manipulation
    IO.puts("\nShape manipulation:")
    e = ExBurn.BurnBridge.from_nx(Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))

    reshape_result = ExBurn.BurnBridge.reshape(e, [2, 3])
    IO.puts("  reshape([1..6], [2,3]): shape=#{inspect(ExBurn.Tensor.shape(reshape_result))}")

    softmax_result =
      ExBurn.BurnBridge.softmax(ExBurn.BurnBridge.from_nx(Nx.tensor([[1.0, 2.0, 3.0]])))
      |> ExBurn.BurnBridge.to_nx()
      |> Nx.to_flat_list()

    IO.puts("  softmax([1,2,3]): #{inspect(softmax_result)}")

    # 8. Loss functions
    IO.puts("\nLoss functions:")
    pred = ExBurn.BurnBridge.from_nx(Nx.tensor([[0.1, 0.2, 0.7], [0.8, 0.1, 0.1]]))
    target = ExBurn.BurnBridge.from_nx(Nx.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]))

    ce_result =
      ExBurn.BurnBridge.cross_entropy(pred, target) |> ExBurn.BurnBridge.to_nx() |> Nx.to_number()

    IO.puts("  cross_entropy: #{Float.round(ce_result, 4)}")

    pred_mse = ExBurn.BurnBridge.from_nx(Nx.tensor([[1.0, 2.0], [3.0, 4.0]]))
    target_mse = ExBurn.BurnBridge.from_nx(Nx.tensor([[1.5, 2.5], [2.5, 3.5]]))

    mse_result =
      ExBurn.BurnBridge.mse(pred_mse, target_mse) |> ExBurn.BurnBridge.to_nx() |> Nx.to_number()

    IO.puts("  mse: #{Float.round(mse_result, 4)}")

    # 9. Nx <-> Burn round-trip
    IO.puts("\nNx <-> Burn round-trip:")
    original = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
    burn_tensor = ExBurn.BurnBridge.from_nx(original)
    roundtrip = ExBurn.BurnBridge.to_nx(burn_tensor)
    IO.puts("  original:  #{inspect(Nx.to_flat_list(original))}")
    IO.puts("  roundtrip: #{inspect(Nx.to_flat_list(roundtrip))}")
    IO.puts("  match: #{Nx.all(Nx.equal(original, roundtrip)) |> Nx.to_number() == 1}")

    # 10. Tensor inspection
    IO.puts("\nTensor inspection:")
    t = ExBurn.BurnBridge.from_nx(Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    IO.puts("  shape: #{inspect(ExBurn.Tensor.shape(t))}")
    IO.puts("  type:  #{ExBurn.Tensor.type(t)}")
    IO.puts("  rank:  #{ExBurn.Tensor.rank(t)}")
    IO.puts("  numel: #{ExBurn.Tensor.numel(t)}")

    IO.puts("\n=== Done ===")
  end
end

BurnBridgeOps.run()
