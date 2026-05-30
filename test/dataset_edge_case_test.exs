defmodule ExBurn.DatasetEdgeCaseTest do
  use ExUnit.Case

  describe "split/2 edge cases" do
    test "handles small val_split" do
      x = Nx.iota({10, 3})
      y = Nx.iota({10, 1})

      {{train_x, _train_y}, {val_x, _val_y}} =
        ExBurn.Dataset.split({x, y}, val_split: 0.1, shuffle: false)

      # 10 * 0.1 = 1 val sample, 9 train samples
      assert elem(Nx.shape(train_x), 0) == 9
      assert elem(Nx.shape(val_x), 0) == 1
    end

    test "handles large val_split" do
      x = Nx.iota({10, 3})
      y = Nx.iota({10, 1})

      {{train_x, _train_y}, {val_x, _val_y}} =
        ExBurn.Dataset.split({x, y}, val_split: 0.9, shuffle: false)

      # 10 * 0.9 = 9 val samples, 1 train sample
      assert elem(Nx.shape(train_x), 0) == 1
      assert elem(Nx.shape(val_x), 0) == 9
    end

    test "handles small dataset" do
      x = Nx.iota({5, 3})
      y = Nx.iota({5, 1})

      {{train_x, _train_y}, {val_x, _val_y}} =
        ExBurn.Dataset.split({x, y}, val_split: 0.2, shuffle: false)

      total = elem(Nx.shape(train_x), 0) + elem(Nx.shape(val_x), 0)
      assert total == 5
    end

    test "handles two sample dataset" do
      x = Nx.iota({2, 3})
      y = Nx.iota({2, 1})

      {{train_x, _train_y}, {val_x, _val_y}} =
        ExBurn.Dataset.split({x, y}, val_split: 0.5, shuffle: false)

      assert elem(Nx.shape(train_x), 0) == 1
      assert elem(Nx.shape(val_x), 0) == 1
    end

    test "split preserves total sample count" do
      x = Nx.iota({20, 3})
      y = Nx.iota({20, 1})

      {{t_x, _}, {v_x, _}} = ExBurn.Dataset.split({x, y}, val_split: 0.2, shuffle: true, seed: 42)

      total = elem(Nx.shape(t_x), 0) + elem(Nx.shape(v_x), 0)
      assert total == 20
    end

    test "shuffle without seed produces different results" do
      x = Nx.iota({100, 3})
      y = Nx.iota({100, 1})

      result1 = ExBurn.Dataset.split({x, y}, val_split: 0.2, shuffle: true)
      result2 = ExBurn.Dataset.split({x, y}, val_split: 0.2, shuffle: true)

      {{t1_x, _}, _} = result1
      {{t2_x, _}, _} = result2

      # With 100 samples, it's extremely unlikely to get the same shuffle
      assert Nx.to_list(t1_x) != Nx.to_list(t2_x)
    end

    test "no shuffle preserves order" do
      x = Nx.iota({10, 3})
      y = Nx.iota({10, 1})

      {{train_x, _train_y}, _} =
        ExBurn.Dataset.split({x, y}, val_split: 0.2, shuffle: false)

      # Without shuffle, first 8 rows should be the first 8 rows of the original
      assert elem(Nx.shape(train_x), 0) == 8
      first_val = train_x |> Nx.slice([0, 0], [1, 1]) |> Nx.to_list() |> List.flatten() |> hd()
      assert first_val == 0.0
    end
  end

  describe "loader/2 edge cases" do
    test "handles batch_size larger than dataset" do
      x = Nx.iota({5, 3})
      y = Nx.iota({5, 1})

      batches =
        ExBurn.Dataset.loader({x, y}, batch_size: 100, shuffle: false)
        |> Enum.to_list()

      assert length(batches) == 1
      {first_x, _first_y} = hd(batches)
      assert elem(Nx.shape(first_x), 0) == 5
    end

    test "handles batch_size of 1" do
      x = Nx.iota({5, 3})
      y = Nx.iota({5, 1})

      batches =
        ExBurn.Dataset.loader({x, y}, batch_size: 1, shuffle: false)
        |> Enum.to_list()

      assert length(batches) == 5

      Enum.each(batches, fn {bx, _by} ->
        assert elem(Nx.shape(bx), 0) == 1
      end)
    end

    test "handles exact multiple batch_size" do
      x = Nx.iota({12, 3})
      y = Nx.iota({12, 1})

      batches =
        ExBurn.Dataset.loader({x, y}, batch_size: 4, shuffle: false)
        |> Enum.to_list()

      assert length(batches) == 3

      Enum.each(batches, fn {bx, _by} ->
        assert elem(Nx.shape(bx), 0) == 4
      end)
    end

    test "drop_last with exact multiple" do
      x = Nx.iota({12, 3})
      y = Nx.iota({12, 1})

      batches =
        ExBurn.Dataset.loader({x, y}, batch_size: 4, shuffle: false, drop_last: true)
        |> Enum.to_list()

      assert length(batches) == 3
    end

    test "returns a Stream" do
      x = Nx.iota({10, 3})
      y = Nx.iota({10, 1})

      loader = ExBurn.Dataset.loader({x, y}, batch_size: 32)
      assert is_struct(loader, Stream)
    end
  end

  describe "normalize/2 edge cases" do
    test "standard normalization with constant feature" do
      # All values in column 0 are the same → std = 0
      x = Nx.tensor([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
      {normalized, stats} = ExBurn.Dataset.normalize(x, method: :standard)

      assert stats.method == :standard
      # Column 0 should be 0 (mean subtracted, divided by 1 to avoid div by zero)
      col_0 = Nx.slice(normalized, [0, 0], [3, 1])
      assert Nx.to_list(col_0) == [[0.0], [0.0], [0.0]]
    end

    test "minmax normalization with constant feature" do
      # All values in column 0 are the same → range = 0
      x = Nx.tensor([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
      {normalized, stats} = ExBurn.Dataset.normalize(x, method: :minmax)

      assert stats.method == :minmax
      # Column 0 should be 0 (min subtracted, divided by 1 to avoid div by zero)
      col_0 = Nx.slice(normalized, [0, 0], [3, 1])
      assert Nx.to_list(col_0) == [[0.0], [0.0], [0.0]]
    end

    test "l2 normalization with zero vector" do
      x = Nx.tensor([[0.0, 0.0], [1.0, 0.0]])
      {normalized, stats} = ExBurn.Dataset.normalize(x, method: :l2)

      assert stats.method == :l2
      # First row should remain zero (norm_safe prevents div by zero)
      row_0 = Nx.slice(normalized, [0, 0], [1, 2])
      assert Nx.to_list(row_0) == [[0.0, 0.0]]
    end

    test "l2 normalization produces unit vectors" do
      x = Nx.tensor([[3.0, 4.0], [1.0, 0.0]])
      {normalized, _stats} = ExBurn.Dataset.normalize(x, method: :l2)

      # First row: norm = 5, so [3/5, 4/5] = [0.6, 0.8]
      row_0 = Nx.to_list(Nx.slice(normalized, [0, 0], [1, 2])) |> List.flatten()
      [v0, v1] = row_0
      assert_in_delta v0, 0.6, 1.0e-6
      assert_in_delta v1, 0.8, 1.0e-6
    end

    test "normalize_with_stats for standard method" do
      train = Nx.tensor([[1.0], [2.0], [3.0]])
      test = Nx.tensor([[2.0]])

      {_train_norm, stats} = ExBurn.Dataset.normalize(train, method: :standard)
      test_norm = ExBurn.Dataset.normalize_with_stats(test, stats)

      # Test sample at mean should normalize to ~0
      val = test_norm |> Nx.to_list() |> List.flatten() |> hd()
      assert_in_delta val, 0.0, 1.0e-6
    end

    test "normalize_with_stats for minmax method" do
      train = Nx.tensor([[0.0], [10.0]])
      test = Nx.tensor([[5.0]])

      {_train_norm, stats} = ExBurn.Dataset.normalize(train, method: :minmax)
      test_norm = ExBurn.Dataset.normalize_with_stats(test, stats)

      # 5.0 is halfway between 0 and 10, so should be 0.5
      val = test_norm |> Nx.to_list() |> List.flatten() |> hd()
      assert_in_delta val, 0.5, 1.0e-6
    end

    test "normalize_with_stats for l2 method" do
      test = Nx.tensor([[3.0, 4.0]])
      stats = %{method: :l2}
      test_norm = ExBurn.Dataset.normalize_with_stats(test, stats)

      # norm = 5, so [3/5, 4/5]
      [v0, v1] = test_norm |> Nx.to_list() |> List.flatten()
      assert_in_delta v0, 0.6, 1.0e-6
      assert_in_delta v1, 0.8, 1.0e-6
    end

    test "raises for unknown normalization method" do
      x = Nx.tensor([[1.0, 2.0]])

      assert_raise ArgumentError, ~r/Unknown normalization method/, fn ->
        ExBurn.Dataset.normalize(x, method: :unknown)
      end
    end
  end

  describe "one_hot/2 edge cases" do
    test "handles single class" do
      labels = Nx.tensor([0])
      one_hot = ExBurn.Dataset.one_hot(labels, num_classes: 1)
      assert Nx.shape(one_hot) == {1, 1}
      assert Nx.to_list(one_hot) == [[1.0]]
    end

    test "handles large number of classes" do
      labels = Nx.tensor([0, 99])
      one_hot = ExBurn.Dataset.one_hot(labels, num_classes: 100)
      assert Nx.shape(one_hot) == {2, 100}
    end

    test "handles non-sequential labels" do
      labels = Nx.tensor([0, 5, 10])
      one_hot = ExBurn.Dataset.one_hot(labels, num_classes: 11)
      assert Nx.shape(one_hot) == {3, 11}
    end

    test "infers num_classes from max label" do
      labels = Nx.tensor([0, 1, 2, 5])
      one_hot = ExBurn.Dataset.one_hot(labels)
      assert Nx.shape(one_hot) == {4, 6}
    end

    test "handles single label" do
      labels = Nx.tensor([3])
      one_hot = ExBurn.Dataset.one_hot(labels, num_classes: 5)
      assert Nx.to_list(one_hot) == [[0.0, 0.0, 0.0, 1.0, 0.0]]
    end
  end

  describe "stats/1 edge cases" do
    test "handles 1D targets" do
      x = Nx.iota({50, 10})
      y = Nx.iota({50})

      stats = ExBurn.Dataset.stats({x, y})
      assert stats.num_samples == 50
      assert stats.input_shape == {50, 10}
      assert stats.target_shape == {50}
    end

    test "handles multi-dimensional targets" do
      x = Nx.iota({20, 5})
      y = Nx.iota({20, 3, 2})

      stats = ExBurn.Dataset.stats({x, y})
      assert stats.num_samples == 20
      assert stats.target_shape == {20, 3, 2}
    end

    test "preserves type information" do
      x = Nx.iota({10, 5}, type: {:f, 64})
      y = Nx.iota({10, 1}, type: {:s, 32})

      stats = ExBurn.Dataset.stats({x, y})
      assert stats.input_type == {:f, 64}
      assert stats.target_type == {:s, 32}
    end
  end
end
