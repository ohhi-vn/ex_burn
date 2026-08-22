# Model Management with ExBurn
# Run with: mix run examples/model_management.exs
#
# Demonstrates ExBurn.Model utilities for model inspection,
# serialization, and manipulation:
#   1. Model compilation with different optimizers
#   2. Model.summary() — Keras-style architecture table
#   3. Model.info() — parameter counts, memory estimates
#   4. Model.benchmark() — forward pass timing
#   5. Model.save/2 and Model.load/2 — serialization
#   6. Model.export/3 and Model.import_params/3 — JSON format
#   7. Model.quantize/2 — reduce precision for deployment
#   8. Model.freeze/2 and Model.unfreeze/2 — transfer learning
#   9. Model.clone/2 — deep copy
#  10. Model.update_params/2 — manual parameter updates

Mix.install([
  {:nx, "~> 0.7"},
  {:axon, "~> 0.7"},
  {:ex_burn, path: Path.expand("..", __DIR__)}
])

defmodule ModelManagement do
  @moduledoc """
  Showcases ExBurn.Model utilities for model management.
  """

  def run do
    IO.puts("=== Model Management with ExBurn ===\n")

    # ── 1. Create a model ──────────────────────────────────────
    model =
      Axon.input("input", shape: {nil, 10})
      |> Axon.dense(64, activation: :relu, name: "encoder")
      |> Axon.dropout(rate: 0.3)
      |> Axon.dense(32, activation: :relu, name: "bottleneck")
      |> Axon.dense(5, name: "classifier")

    IO.puts("Model architecture:")
    IO.inspect(Axon.get_output_shape(model, Nx.template({1, 10}, :f32)), label: "Output shape")

    # ── 2. Compile with Adam ────────────────────────────────────
    compiled =
      ExBurn.Model.compile(model,
        loss: :cross_entropy,
        optimizer: :adam,
        learning_rate: 0.001,
        weight_decay: 1.0e-4
      )

    # ── 3. Model summary ────────────────────────────────────────
    IO.puts(ExBurn.Model.summary(compiled))

    # ── 4. Model info ───────────────────────────────────────────
    info = ExBurn.Model.info(compiled)
    IO.puts("Model info:")
    IO.puts("  Total params:     #{info.total_params}")
    IO.puts("  Layer count:      #{info.layer_count}")
    IO.puts("  Loss function:    #{info.loss_function}")
    IO.puts("  Optimizer:        #{info.optimizer}")
    IO.puts("  Learning rate:    #{info.learning_rate}")
    IO.puts("  Device:           #{info.device}")
    IO.puts("  Weight decay:     #{info.weight_decay}")
    IO.puts("  Est. memory:      #{info.estimated_memory_mb} MB")
    IO.puts("  Compiled:         #{info.compiled}\n")

    # ── 5. Benchmark forward pass ──────────────────────────────
    {input, _key} = Nx.Random.uniform(Nx.Random.key(1), 0.0, 1.0, shape: {1, 10})
    bench = ExBurn.Model.benchmark(compiled, input, warmup: 3, runs: 10)

    IO.puts("Benchmark (10 runs, 3 warmup):")
    IO.puts("  Avg:   #{bench.avg_ms} ms")
    IO.puts("  Min:   #{bench.min_ms} ms")
    IO.puts("  Max:   #{bench.max_ms} ms")
    IO.puts("  Median: #{bench.median_ms} ms")
    IO.puts("  Std:   #{bench.std_ms} ms\n")

    # ── 6. Save and load ────────────────────────────────────────
    model_path = "/tmp/ex_burn_model.model"
    IO.puts("Saving model to #{model_path}...")
    :ok = ExBurn.Model.save(compiled, model_path)

    {:ok, loaded} = ExBurn.Model.load(compiled, model_path)
    IO.puts("  Loaded successfully: #{loaded.compiled}")
    IO.puts("  Params match: #{map_size(loaded.params) == map_size(compiled.params)}\n")

    # ── 7. Export and import (JSON) ─────────────────────────────
    json_path = "/tmp/ex_burn_model.json"
    IO.puts("Exporting to JSON: #{json_path}...")
    :ok = ExBurn.Model.export(compiled, json_path, format: :json)

    {:ok, imported} = ExBurn.Model.import_params(compiled, json_path, format: :json)
    IO.puts("  Imported successfully: #{imported.compiled}\n")

    # ── 8. Serialize / deserialize params ──────────────────────
    binary = ExBurn.Model.serialize_params(compiled)
    {:ok, deserialized} = ExBurn.Model.deserialize_params(binary)
    IO.puts("Serialize/deserialize params:")
    IO.puts("  Binary size: #{byte_size(binary)} bytes")
    IO.puts("  Keys match:  #{map_size(deserialized) == map_size(compiled.params)}\n")

    # ── 9. Quantize ─────────────────────────────────────────────
    quantized = ExBurn.Model.quantize(compiled, :f16)
    q_info = ExBurn.Model.info(quantized)
    IO.puts("Quantized model (f16):")
    IO.puts("  Total params: #{q_info.total_params}")
    IO.puts("  Est. memory:  #{q_info.estimated_memory_mb} MB\n")

    # ── 10. Layer freezing (transfer learning) ──────────────────
    frozen = ExBurn.Model.freeze(compiled, ["encoder", "bottleneck"])
    IO.puts("Frozen layers: encoder, bottleneck")
    IO.puts("  encoder frozen:    #{ExBurn.Model.frozen?(frozen, "encoder")}")
    IO.puts("  bottleneck frozen: #{ExBurn.Model.frozen?(frozen, "bottleneck")}")
    IO.puts("  classifier frozen: #{ExBurn.Model.frozen?(frozen, "classifier")}")

    IO.puts(
      "  All frozen:        #{ExBurn.Model.frozen_layers(frozen) |> MapSet.to_list() |> Enum.join(", ")}"
    )

    unfrozen = ExBurn.Model.unfreeze(frozen, ["encoder"])
    IO.puts("\nUnfrozen: encoder")
    IO.puts("  encoder frozen:    #{ExBurn.Model.frozen?(unfrozen, "encoder")}")
    IO.puts("  bottleneck frozen: #{ExBurn.Model.frozen?(unfrozen, "bottleneck")}\n")

    # ── 11. Clone ───────────────────────────────────────────────
    snapshot = ExBurn.Model.clone(compiled)
    IO.puts("Cloned model:")
    IO.puts("  Params count: #{map_size(snapshot.params)}")

    IO.puts(
      "  Same config:  #{snapshot.optimizer == compiled.optimizer && snapshot.loss_fn == compiled.loss_fn}\n"
    )

    # ── 12. Update params manually ─────────────────────────────
    # Zero out all parameters
    zero_params =
      Enum.map(compiled.params, fn {key, tensor} ->
        {key, Nx.broadcast(Nx.tensor(0.0, type: Nx.type(tensor)), Nx.shape(tensor))}
      end)
      |> Map.new()

    zeroed_model = ExBurn.Model.update_params(compiled, zero_params)
    zeroed_info = ExBurn.Model.info(zeroed_model)
    IO.puts("Model with zeroed params:")
    IO.puts("  Total params: #{zeroed_info.total_params}")
    IO.puts("  (params are zeroed but count is preserved)\n")

    # ── 13. Compare optimizers ─────────────────────────────────
    IO.puts("Optimizer comparison:")

    for opt <- [:adam, :sgd, :rmsprop] do
      m = ExBurn.Model.compile(model, loss: :cross_entropy, optimizer: opt, learning_rate: 0.001)
      info = ExBurn.Model.info(m)

      IO.puts(
        "  #{String.pad_trailing("#{opt}", 8)} lr=#{info.learning_rate} params=#{info.total_params}"
      )
    end

    IO.puts("\n=== Done ===")
  end
end

ModelManagement.run()
