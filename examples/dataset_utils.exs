# Dataset Utilities with ExBurn
# Run with: mix run examples/dataset_utils.exs
#
# Demonstrates the ExBurn.Dataset module for common data preprocessing:
#   1. Splitting data into train/validation sets
#   2. Creating batched data loaders with shuffling
#   3. Normalization (standard, minmax, l2)
#   4. One-hot encoding
#   5. Dataset statistics

Mix.install([
  {:nx, "~> 0.7"},
  {:axon, "~> 0.7"},
  {:ex_burn, path: Path.expand("..", __DIR__)}
])

defmodule DatasetUtils do
  @moduledoc """
  Showcases ExBurn.Dataset utilities for data preprocessing.
  """

  def run do
    IO.puts("=== Dataset Utilities with ExBurn ===\n")

    # ── 1. Create sample dataset ───────────────────────────────
    key = Nx.Random.key(42)

    # 100 samples, 5 features
    {x, key} = Nx.Random.normal(key, 0.0, 1.0, shape: {100, 5})

    # Integer class labels (3 classes)
    {labels, _key} = Nx.Random.uniform(key, 0, 2.999, shape: {100})
    labels = Nx.as_type(labels, {:s, 64})

    IO.puts("Original dataset:")
    stats = ExBurn.Dataset.stats({x, labels})
    IO.puts("  Samples: #{stats.num_samples}")
    IO.puts("  Input shape: #{inspect(stats.input_shape)}")
    IO.puts("  Target shape: #{inspect(stats.target_shape)}")
    IO.puts("  Input type: #{inspect(stats.input_type)}")
    IO.puts("  Target type: #{inspect(stats.target_type)}\n")

    # ── 2. Split into train / validation ───────────────────────
    {train, val} = ExBurn.Dataset.split({x, labels}, val_split: 0.25, seed: 42)
    {train_x, train_y} = train
    {val_x, val_y} = val

    IO.puts("After split (val_split=0.25, seed=42):")
    IO.puts("  Train: #{Nx.shape(train_x) |> elem(0)} samples")
    IO.puts("  Val:   #{Nx.shape(val_x) |> elem(0)} samples\n")

    # ── 3. Normalization ───────────────────────────────────────
    IO.puts("Normalization:")

    # Standard (z-score) normalization
    {train_std, std_stats} = ExBurn.Dataset.normalize(train_x, method: :standard)
    _val_std = ExBurn.Dataset.normalize_with_stats(val_x, std_stats)

    IO.puts("  Standard (z-score):")

    IO.puts(
      "    Train mean (first feature): #{train_std |> Nx.slice([0, 0], [1, 1]) |> Nx.squeeze() |> Nx.to_number() |> Float.round(4)}"
    )

    # Min-max normalization
    {train_minmax, minmax_stats} = ExBurn.Dataset.normalize(train_x, method: :minmax)
    _val_minmax = ExBurn.Dataset.normalize_with_stats(val_x, minmax_stats)

    IO.puts("  Min-Max [0, 1]:")

    IO.puts(
      "    Train min: #{train_minmax |> Nx.reduce_min() |> Nx.to_number() |> Float.round(4)}"
    )

    IO.puts(
      "    Train max: #{train_minmax |> Nx.reduce_max() |> Nx.to_number() |> Float.round(4)}"
    )

    # L2 normalization
    {train_l2, l2_stats} = ExBurn.Dataset.normalize(train_x, method: :l2)
    _val_l2 = ExBurn.Dataset.normalize_with_stats(val_x, l2_stats)

    IO.puts("  L2 (row-wise unit norm):")
    # Check that rows have unit norm
    row_norms = Nx.sqrt(Nx.sum(Nx.multiply(train_l2, train_l2), axes: [-1]))
    IO.puts("    Mean row norm: #{Nx.mean(row_norms) |> Nx.to_number() |> Float.round(4)}")

    # ── 4. One-hot encoding ────────────────────────────────────
    IO.puts("\nOne-hot encoding (3 classes):")
    one_hot = ExBurn.Dataset.one_hot(train_y, num_classes: 3)
    IO.puts("  Shape: #{inspect(Nx.shape(one_hot))}")
    IO.puts("  First 3 rows:")
    one_hot_slice = Nx.slice(one_hot, [0, 0], [3, 3])

    Enum.each(0..2, fn i ->
      row = Nx.slice(one_hot_slice, [i, 0], [1, 3]) |> Nx.to_flat_list()

      IO.puts(
        "    Class #{Enum.at(row, 0) |> round()} → #{Enum.map_join(row, ", ", &Float.round(&1, 1))}"
      )
    end)

    # ── 5. Batched data loader ─────────────────────────────────
    IO.puts("\nBatched data loader (batch_size=16, shuffle=true):")
    loader = ExBurn.Dataset.loader({train_x, train_y}, batch_size: 16, shuffle: true)

    loader
    |> Stream.take(3)
    |> Enum.with_index(1)
    |> Enum.each(fn {{batch_x, batch_y}, idx} ->
      IO.puts("  Batch #{idx}: x=#{inspect(Nx.shape(batch_x))}, y=#{inspect(Nx.shape(batch_y))}")
    end)

    # ── 6. Data loader with drop_last ─────────────────────────
    IO.puts("\nData loader with drop_last=true:")
    loader_drop = ExBurn.Dataset.loader({train_x, train_y}, batch_size: 16, drop_last: true)
    batches = Enum.to_list(loader_drop)
    IO.puts("  Total batches: #{length(batches)}")

    # Without drop_last
    loader_keep = ExBurn.Dataset.loader({train_x, train_y}, batch_size: 16, drop_last: false)
    batches_keep = Enum.to_list(loader_keep)
    IO.puts("  Total batches (keep last): #{length(batches_keep)}")

    IO.puts("\n=== Done ===")
  end
end

DatasetUtils.run()
