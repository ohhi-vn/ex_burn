# Benchmarks

ExBurn includes benchmark scripts in `bench/` that compare performance against plain Nx (BinaryBackend) across a range of tensor sizes. All scripts use `Mix.install/1` and can be run standalone.

## Running Benchmarks

```bash
# Tensor creation (zeros, ones, rand)
mix run bench/tensor_creation_bench.exs

# Element-wise arithmetic (add, mul, exp)
mix run bench/arithmetic_bench.exs

# Linear algebra (matmul, transpose)
mix run bench/linear_algebra_bench.exs

# Nx <-> Burn tensor conversion overhead
mix run bench/conversion_bench.exs

# End-to-end training (small/medium MLPs, optimizer comparison)
mix run bench/training_bench.exs

# Inference latency and throughput (single + batched + Nx.Serving)
mix run bench/serving_bench.exs
```

## Benchmark Design

- **Warmup**: Each benchmark runs a warmup pass before timing to account for NIF loading, GPU kernel compilation, and JIT caching.
- **Averaged measurements**: Results are averaged over multiple runs (20-50 depending on operation cost).
- **Formatted output**: Results are printed as aligned tables for easy comparison.

## Tensor Creation (`tensor_creation_bench.exs`)

Compares `Nx.broadcast/2` (for zeros/ones) and `Nx.Random.uniform/2` against `BurnBridge.zeros/1`, `BurnBridge.ones/1`, and `BurnBridge.rand/4`.

Tested shapes: `{10,10}`, `{100,100}`, `{500,500}`, `{1000,1000}`, `{100,1000}`, `{1000,100}`.

## Arithmetic (`arithmetic_bench.exs`)

Compares element-wise operations: `Nx.add/2`, `Nx.multiply/2`, `Nx.exp/1` against `BurnBridge.add/2`, `BurnBridge.mul/2`, `BurnBridge.exp/1`.

Tested sizes: 100×100, 500×500, 1000×1000, 2000×2000.

## Linear Algebra (`linear_algebra_bench.exs`)

Compares `Nx.dot/2` (matrix multiplication) and `Nx.transpose/1` against `BurnBridge.matmul/2` and `BurnBridge.transpose/1`.

Tested sizes: 50×50, 100×100, 250×250, 500×500, 1000×1000.

## Conversion Overhead (`conversion_bench.exs`)

Measures the cost of converting between Nx and Burn tensor formats. Reports one-way and round-trip times, plus overhead (round-trip minus sum of one-way times).

Tested shapes: `{10,10}`, `{100,100}`, `{500,500}`, `{1000,1000}`, `{2000,2000}`, `{100,5000}`, `{5000,100}`.

## Training (`training_bench.exs`)

Benchmarks end-to-end training performance:

1. **Small MLP** (10→32→16→3): 500 samples, 20 epochs, batch_size=32
2. **Medium MLP** (50→128→64→10): 1000 samples, 15 epochs, batch_size=64
3. **Optimizer comparison**: Adam vs SGD vs RMSprop on a 10→64→32→5 model
4. **Batch size scaling**: batch sizes 16, 32, 64, 128 on the same model

Reports forward pass latency, total training time, per-epoch time, and final validation loss.

## Serving (`serving_bench.exs`)

Benchmarks inference throughput:

1. **Single inference latency**: 100 runs with warmup
2. **Batched throughput**: batch sizes 1, 4, 16, 32, 64 — reports total time, per-sample time, and samples/sec
3. **Nx.Serving**: 10 concurrent requests through `ExBurn.Serving`

## Interpreting Results

- **Small tensors** (10×10): NIF call overhead dominates. Nx may be faster.
- **Medium tensors** (100-500): GPU acceleration begins to show advantage for compute-bound ops.
- **Large tensors** (1000+): GPU parallelism provides significant speedup for matmul and reductions.
- **Conversion overhead**: One-time cost per tensor. Amortized over many operations in training loops.
- **Training**: Per-epoch time includes forward pass, backward pass (numerical gradients), and optimizer step. Numerical gradients are O(n) in parameter count.
