# Linear Regression with ExBurn
# Run with: mix run examples/linear_regression.exs
#
# Demonstrates the simplest possible ML workflow:
#   1. Generate synthetic data: y = 2x + 1 + noise
#   2. Define a single-neuron model (linear regression)
#   3. Train using SGD
#   4. Verify the learned weights approximate w=2.0, b=1.0

Mix.install([
  {:nx, "~> 0.7"},
  {:axon, "~> 0.7"},
  {:ex_burn, path: Path.expand("..", __DIR__)}
])

defmodule LinearRegression do
  @moduledoc """
  Simple linear regression: y = wx + b
  Learns w ≈ 2.0, b ≈ 1.0 from noisy data.
  """

  def run do
    IO.puts("=== Linear Regression with ExBurn ===\n")

    # ── 1. Generate synthetic data ──────────────────────────────
    num_samples = 200
    key = Nx.Random.key(42)

    # x ~ Uniform(-3, 3)
    {x, key} = Nx.Random.uniform(key, -3.0, 3.0, shape: {num_samples, 1})

    # y = 2x + 1 + noise(σ=0.1)
    {noise, _key} = Nx.Random.normal(key, 0.0, 0.1, shape: {num_samples, 1})
    y = Nx.add(Nx.multiply(x, 2.0), Nx.add(1.0, noise))

    IO.puts("Data: #{num_samples} samples, y = 2x + 1 + ε")
    IO.puts("  x range: [#{Nx.reduce_min(x) |> Nx.to_number() |> Float.round(2)}, #{Nx.reduce_max(x) |> Nx.to_number() |> Float.round(2)}]")
    IO.puts("  y range: [#{Nx.reduce_min(y) |> Nx.to_number() |> Float.round(2)}, #{Nx.reduce_max(y) |> Nx.to_number() |> Float.round(2)}]")

    # ── 2. Define model ─────────────────────────────────────────
    # Single dense layer with no activation = linear regression
    model =
      Axon.input("input", shape: {nil, 1})
      |> Axon.dense(1, use_bias: true, name: "linear")

    IO.puts("\nModel:")
    IO.puts(Axon.Display.display(model, []))

    # ── 3. Compile ──────────────────────────────────────────────
    compiled = ExBurn.Model.compile(model,
      loss: :mse,
      optimizer: :sgd,
      learning_rate: 0.05
    )

    IO.puts("Initial prediction (x=1.0):")
    pred0 = predict(compiled, Nx.tensor([[1.0]]))
    IO.puts("  f(1.0) = #{Float.round(Nx.to_number(pred0), 4)} (expected ≈ 3.0)\n")

    # ── 4. Train ────────────────────────────────────────────────
    IO.puts("Training...")

    trained =
      ExBurn.Training.fit(compiled, {x, y},
        epochs: 100,
        batch_size: 32,
        verbose: false,
        callbacks: [
          fn
            %{epoch: epoch, loss: loss} when rem(epoch, 20) == 0 ->
              IO.puts("  Epoch #{String.pad_leading("#{epoch}", 3)}: loss = #{Float.round(loss, 6)}")
              %{epoch: epoch, loss: loss}

            metrics ->
              metrics
          end
        ]
      )

    # ── 5. Evaluate ─────────────────────────────────────────────
    IO.puts("\nResults:")

    pred_final = predict(trained, Nx.tensor([[1.0]]))
    IO.puts("  f(1.0) = #{Float.round(Nx.to_number(pred_final), 4)} (expected ≈ 3.0)")

    pred_zero = predict(trained, Nx.tensor([[0.0]]))
    IO.puts("  f(0.0) = #{Float.round(Nx.to_number(pred_zero), 4)} (expected ≈ 1.0)")

    pred_neg = predict(trained, Nx.tensor([[-2.0]]))
    IO.puts("  f(-2.0) = #{Float.round(Nx.to_number(pred_neg), 4)} (expected ≈ -3.0)")

    # Compute final MSE
    y_pred = predict_batch(trained, x)
    mse = Nx.mean(Nx.power(Nx.subtract(y_pred, y), 2.0)) |> Nx.to_number()
    IO.puts("\n  Final MSE: #{Float.round(mse, 6)}")

    # Extract learned weights
    params = ExBurn.Model.parameters(trained)
    {w, b} = extract_linear_params(params)
    IO.puts("  Learned: w = #{Float.round(w, 4)}, b = #{Float.round(b, 4)}")
    IO.puts("  Target:  w = 2.0000, b = 1.0000")

    IO.puts("\n=== Done ===")
  end

  defp predict(%ExBurn.Model{params: params}, input) do
    # Forward pass: y = x @ w + b
    w = params["linear"]["weight"]
    b = params["linear"]["bias"]
    Nx.add(Nx.dot(input, w), b)
  end

  defp predict_batch(%ExBurn.Model{params: params}, input) do
    w = params["linear"]["weight"]
    b = params["linear"]["bias"]
    Nx.add(Nx.dot(input, Nx.transpose(w)), b)
  end

  defp extract_linear_params(params) do
    w = params["linear"]["weight"] |> Nx.to_number()
    b = params["linear"]["bias"] |> Nx.to_number()
    {w, b}
  end
end

LinearRegression.run()
