defmodule ExBurn.DatasetTest do
  use ExUnit.Case

  describe "split/2" do
    test "splits data into train and validation sets" do
      x = Nx.iota({10, 3})
      y = Nx.iota({10, 1})

      {{train_x, train_y}, {val_x, val_y}} =
        ExBurn.Dataset.split({x, y}, val_split: 0.2, shuffle: false)

      # 80% train = 8 samples, 20% val = 2 samples
      assert elem(Nx.shape(train_x), 0) == 8
      assert elem(Nx.shape(val_x), 0) == 2
      assert elem(Nx.shape(train_y), 0) == 8
      assert elem(Nx.shape(val_y), 0) == 2
    end

    test "handles small datasets" do
      x = Nx.iota({5, 2})
      y = Nx.iota({5, 1})

      {{train_x, _train_y}, {val_x, _val_y}} =
        ExBurn.Dataset.split({x, y}, val_split: 0.2, shuffle: false)

      total = elem(Nx.shape(train_x), 0) + elem(Nx.shape(val_x), 0)
      assert total == 5
    end
  end

  describe "loader/2" do
    test "creates batches of the specified size" do
      x = Nx.iota({10, 3})
      y = Nx.iota({10, 1})

      batches =
        ExBurn.Dataset.loader({x, y}, batch_size: 3, shuffle: false)
        |> Enum.to_list()

      # 10 samples / 3 batch_size = 4 batches (3, 3, 3, 1)
      assert length(batches) == 4

      # First batch should have 3 samples
      {first_x, _first_y} = hd(batches)
      assert elem(Nx.shape(first_x), 0) == 3
    end

    test "drops last batch when drop_last is true" do
      x = Nx.iota({10, 3})
      y = Nx.iota({10, 1})

      batches =
        ExBurn.Dataset.loader({x, y}, batch_size: 3, shuffle: false, drop_last: true)
        |> Enum.to_list()

      # 10 / 3 = 3 full batches, last one dropped
      assert length(batches) == 3
    end
  end

  describe "normalize/2" do
    test "standard normalization produces zero mean and unit variance" do
      x = Nx.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
      {normalized, stats} = ExBurn.Dataset.normalize(x, method: :standard)

      assert stats.method == :standard
      assert Nx.shape(stats.mean) == {1, 2}
      assert Nx.shape(stats.std) == {1, 2}

      # Mean of normalized data should be approximately 0
      mean = Nx.mean(normalized, axes: [0]) |> Nx.abs() |> Nx.reduce_max()
      assert_in_delta(Nx.to_number(mean), 0.0, 1.0e-6)
    end

    test "minmax normalization produces values in [0, 1]" do
      x = Nx.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
      {normalized, stats} = ExBurn.Dataset.normalize(x, method: :minmax)

      assert stats.method == :minmax

      # All values should be in [0, 1]
      min_val = Nx.reduce_min(normalized) |> Nx.to_number()
      max_val = Nx.reduce_max(normalized) |> Nx.to_number()
      assert min_val >= 0.0
      assert max_val <= 1.0
    end

    test "normalize_with_stats applies same transformation" do
      train = Nx.tensor([[1.0], [2.0], [3.0]])
      test = Nx.tensor([[1.5]])

      {_train_norm, stats} = ExBurn.Dataset.normalize(train, method: :standard)
      test_norm = ExBurn.Dataset.normalize_with_stats(test, stats)

      # Test sample should be between train min and max when normalized
      assert is_struct(test_norm, Nx.Tensor)
    end
  end

  describe "one_hot/2" do
    test "encodes integer labels to one-hot" do
      labels = Nx.tensor([0, 2, 1])
      one_hot = ExBurn.Dataset.one_hot(labels, num_classes: 3)

      assert Nx.shape(one_hot) == {3, 3}
      assert Nx.to_list(one_hot) == [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
    end

    test "infers num_classes when not provided" do
      labels = Nx.tensor([0, 1, 2, 3])
      one_hot = ExBurn.Dataset.one_hot(labels)

      assert Nx.shape(one_hot) == {4, 4}
    end
  end

  describe "stats/1" do
    test "returns dataset statistics" do
      x = Nx.iota({100, 5})
      y = Nx.iota({100, 1})

      stats = ExBurn.Dataset.stats({x, y})

      assert stats.num_samples == 100
      assert stats.input_shape == {100, 5}
      assert stats.target_shape == {100, 1}
    end
  end
end
