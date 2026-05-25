defmodule ExBurn.Dataset do
  @moduledoc """
  Dataset utilities for ExBurn.

  Provides common data loading, splitting, and preprocessing helpers
  for machine learning workflows.

  ## Usage

      # Split data into train/validation sets
      {train, val} = ExBurn.Dataset.split({x, y}, val_split: 0.2)

      # Create a batched data loader
      loader = ExBurn.Dataset.loader({x, y}, batch_size: 32, shuffle: true)

      # Normalize features
      {normalized, stats} = ExBurn.Dataset.normalize(x, method: :standard)
  """

  @type dataset :: {Nx.Tensor.t(), Nx.Tensor.t()}

  @doc """
  Splits a dataset into training and validation sets.

  ## Options

    * `:val_split` — Fraction of data for validation (default: 0.2)
    * `:shuffle` — Shuffle before splitting (default: true)
    * `:seed` — Random seed for reproducibility (default: nil)

  ## Returns

    `{train_data, val_data}` where each is `{inputs, targets}`.

  ## Example

      {train, val} = ExBurn.Dataset.split({x, y}, val_split: 0.2, seed: 42)
  """
  @spec split(dataset(), keyword()) :: {dataset(), dataset()}
  def split({inputs, targets}, opts \\ []) do
    val_split = Keyword.get(opts, :val_split, 0.2)
    shuffle = Keyword.get(opts, :shuffle, true)
    seed = Keyword.get(opts, :seed)

    num_samples = Nx.shape(inputs) |> elem(0)
    num_val = round(num_samples * val_split)
    num_train = num_samples - num_val

    indices =
      if shuffle do
        if seed, do: :rand.seed(:exsss, seed)
        Enum.shuffle(0..(num_samples - 1))
      else
        Enum.to_list(0..(num_samples - 1))
      end

    {train_indices, val_indices} = Enum.split(indices, num_train)

    train_inputs = gather_indices(inputs, train_indices)
    train_targets = gather_indices(targets, train_indices)
    val_inputs = gather_indices(inputs, val_indices)
    val_targets = gather_indices(targets, val_indices)

    {{train_inputs, train_targets}, {val_inputs, val_targets}}
  end

  @doc """
  Creates a batched data loader from a dataset.

  Returns a `Stream` of `{batch_inputs, batch_targets}` tuples.

  ## Options

    * `:batch_size` — Batch size (default: 32)
    * `:shuffle` — Shuffle data each epoch (default: true)
    * `:drop_last` — Drop the last incomplete batch (default: false)
    * `:seed` — Random seed for shuffling

  ## Example

      ExBurn.Dataset.loader({x, y}, batch_size: 64)
      |> Enum.each(fn {batch_x, batch_y} ->
        # process batch
      end)
  """
  @spec loader(dataset(), keyword()) :: Enumerable.t()
  def loader({inputs, targets}, opts \\ []) do
    batch_size = Keyword.get(opts, :batch_size, 32)
    shuffle = Keyword.get(opts, :shuffle, true)
    drop_last = Keyword.get(opts, :drop_last, false)

    num_samples = Nx.shape(inputs) |> elem(0)

    indices =
      if shuffle do
        Enum.shuffle(0..(num_samples - 1))
      else
        Enum.to_list(0..(num_samples - 1))
      end

    indices
    |> Stream.chunk_every(batch_size, batch_size, if(drop_last, do: :discard, else: []))
    |> Stream.map(fn batch_indices ->
      batch_inputs = gather_indices(inputs, batch_indices)
      batch_targets = gather_indices(targets, batch_indices)
      {batch_inputs, batch_targets}
    end)
  end

  @doc """
  Normalizes a tensor using the specified method.

  ## Options

    * `:method` — Normalization method: `:standard` (z-score), `:minmax`, or `:l2` (default: `:standard`)
    * `:axes` — Axes to compute statistics over (default: [0])

  ## Returns

    `{normalized_tensor, stats_map}` where stats can be used to normalize
    new data with `normalize_with_stats/3`.

  ## Example

      {train_norm, stats} = ExBurn.Dataset.normalize(train_x, method: :standard)
      test_norm = ExBurn.Dataset.normalize_with_stats(test_x, stats)
  """
  @spec normalize(Nx.Tensor.t(), keyword()) :: {Nx.Tensor.t(), map()}
  def normalize(tensor, opts \\ []) do
    method = Keyword.get(opts, :method, :standard)

    case method do
      :standard ->
        mean = Nx.mean(tensor, axes: [0], keep_axes: true)
        std = Nx.standard_deviation(tensor, axes: [0], keep_axes: true)
        # Avoid division by zero
        std_safe = Nx.select(Nx.equal(std, 0.0), Nx.tensor(1.0), std)
        normalized = Nx.divide(Nx.subtract(tensor, mean), std_safe)
        {normalized, %{method: :standard, mean: mean, std: std}}

      :minmax ->
        min = Nx.reduce_min(tensor, axes: [0], keep_axes: true)
        max = Nx.reduce_max(tensor, axes: [0], keep_axes: true)
        range = Nx.subtract(max, min)
        range_safe = Nx.select(Nx.equal(range, 0.0), Nx.tensor(1.0), range)
        normalized = Nx.divide(Nx.subtract(tensor, min), range_safe)
        {normalized, %{method: :minmax, min: min, max: max}}

      :l2 ->
        norm = Nx.sqrt(Nx.sum(Nx.multiply(tensor, tensor), axes: [-1], keep_axes: true))
        norm_safe = Nx.select(Nx.equal(norm, 0.0), Nx.tensor(1.0), norm)
        normalized = Nx.divide(tensor, norm_safe)
        {normalized, %{method: :l2}}

      other ->
        raise ArgumentError, "Unknown normalization method: #{inspect(other)}"
    end
  end

  @doc """
  Normalizes a tensor using pre-computed statistics.

  Useful for applying the same normalization to test/validation data.

  ## Example

      {train_norm, stats} = ExBurn.Dataset.normalize(train_x)
      test_norm = ExBurn.Dataset.normalize_with_stats(test_x, stats)
  """
  @spec normalize_with_stats(Nx.Tensor.t(), map()) :: Nx.Tensor.t()
  def normalize_with_stats(tensor, %{method: :standard, mean: mean, std: std}) do
    std_safe = Nx.select(Nx.equal(std, 0.0), Nx.tensor(1.0), std)
    Nx.divide(Nx.subtract(tensor, mean), std_safe)
  end

  def normalize_with_stats(tensor, %{method: :minmax, min: min, max: max}) do
    range = Nx.subtract(max, min)
    range_safe = Nx.select(Nx.equal(range, 0.0), Nx.tensor(1.0), range)
    Nx.divide(Nx.subtract(tensor, min), range_safe)
  end

  def normalize_with_stats(tensor, %{method: :l2}) do
    norm = Nx.sqrt(Nx.sum(Nx.multiply(tensor, tensor), axes: [-1], keep_axes: true))
    norm_safe = Nx.select(Nx.equal(norm, 0.0), Nx.tensor(1.0), norm)
    Nx.divide(tensor, norm_safe)
  end

  @doc """
  Applies one-hot encoding to integer class labels.

  ## Parameters

    * `labels` — 1D tensor of integer class indices
    * `num_classes` — Total number of classes

  ## Returns

    2D tensor of one-hot encoded labels.

  ## Example

      one_hot = ExBurn.Dataset.one_hot(Nx.tensor([0, 2, 1]), num_classes: 3)
      # [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
  """
  @spec one_hot(Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  def one_hot(labels, opts \\ []) do
    num_classes =
      Keyword.get(opts, :num_classes) ||
        compute_num_classes(labels)

    Nx.equal(Nx.new_axis(labels, -1), Nx.iota({num_classes}))
    |> Nx.as_type(:f32)
  end

  @doc ~S"""
  Returns basic statistics about a dataset.

  ## Returns

    A map with `:num_samples`, `:input_shape`, `:target_shape`,
    `:input_type`, `:target_type`.

  ## Example

      s = ExBurn.Dataset.stats({x, y})
      IO.puts("Samples: #{s.num_samples}")
  """
  @spec stats(dataset()) :: map()
  def stats({inputs, targets}) do
    %{
      num_samples: Nx.shape(inputs) |> elem(0),
      input_shape: Nx.shape(inputs),
      target_shape: Nx.shape(targets),
      input_type: Nx.type(inputs),
      target_type: Nx.type(targets)
    }
  end

  # ── Private ──────────────────────────────────────────────────────

  defp gather_indices(tensor, indices) when is_list(indices) do
    indices_tensor = Nx.tensor(indices, type: {:s, 64})
    Nx.take(tensor, indices_tensor)
  end

  defp compute_num_classes(labels) do
    max_label = Nx.reduce_max(labels) |> Nx.to_number() |> round()
    max_label + 1
  end
end
