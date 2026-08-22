# Contributing to ExBurn

Thank you for your interest in contributing! This guide walks you through the
full development workflow — from a fresh clone to a merged PR.

## Getting Started

### 1. Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Erlang/OTP | 27+ | |
| Elixir | ~> 1.18 | |
| Rust | stable | Required — the Burn NIF compiles from source |
| GPU drivers | optional | CUDA / Metal / Vulkan, depending on platform |

Install the Rust toolchain if you don't have it:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

For mobile targets (optional):

```bash
rustup target add aarch64-apple-ios      # iOS
rustup target add aarch64-linux-android  # Android
```

### 2. Clone and bootstrap

```bash
git clone https://github.com/YOUR_USERNAME/ex_burn.git
cd ex_burn
mix deps.get
mix compile          # first build compiles the Rust NIF (debug mode) — expect a few minutes
```

> The NIF builds **without a GPU backend by default** (CPU NdArray). To develop
> against real GPU acceleration, pick a backend before compiling:
>
> ```bash
> ./build.sh metal       # or cuda / vulkan / cpu — auto-detect with no args
> # or set the cargo feature manually:
> RUSTLER_NIF_CARGO_FEATURES=cuda mix compile
> ```

### 3. Verify your setup

```bash
mix run -e 'IO.puts(ExBurn.summary())'
# ExBurn v0.5.0
# Device: Metal (Apple M...)   ← or "NdArray (CPU)" on machines without GPUs
# GPU: available               ← or "not available"
# Backends: metal

mix run -e 'IO.inspect(ExBurn.smoke_test())'
# :ok
```

If `smoke_test/0` returns `:ok`, the whole pipeline (Nx → Backend → NIF → Burn)
is functional.

## Running Tests

```bash
mix test                          # full suite (~600 tests, <30s)
mix test test/backend_test.exs    # single file
mix test test/backend_test.exs:42 # single test by line
mix test --only nif               # only tests that need the NIF
mix test --exclude nif            # everything except NIF tests
```

### Test tags

| Tag | Meaning | Auto-excluded when |
|---|---|---|
| `:nif` | Requires the compiled Rust NIF | NIF failed to load |
| `:cuda` / `:metal` / `:vulkan` | Requires specific GPU hardware | No matching GPU detected |

`test/test_helper.exs` handles the exclusions automatically — you don't need
to pass them manually.

### Coverage

```bash
mix test --cover
```

Coverage is enforced with a threshold (see `test_coverage` in `mix.exs`).
Notes on the current configuration:

- `ExBurn.Nif` is **ignored**: its uncovered lines are load-bearing Rustler
  declarations (`:erlang.nif_error/1` bodies) that only execute if the NIF
  fails to load.
- The remaining ceiling comes from lines that are unreachable without fault
  injection into the native layer (documented inline in `mix.exs`). If you
  close one of those gaps, feel free to raise the threshold.

## Linting & Static Analysis

```bash
mix lint        # format check + Credo
mix lint.all    # the above + Dialyzer
mix format      # auto-fix formatting
```

Rust side:

```bash
cd native/ex_burn_nif
cargo fmt -- --check
cargo clippy -- -D warnings
```

Run these before opening a PR — CI runs all of them.

## Project Layout

```
lib/ex_burn/
  ex_burn.ex           Main API: version, configure!, device info, smoke_test
  backend.ex           Nx.Backend implementation (delegates to Burn via NIF)
  defn_compiler.ex     Nx.Defn.Compiler (traces exprs → backend callbacks)
  nif.ex               Raw Rustler stubs + dtype validation at the boundary
  nif_helper.ex        Safe wrappers converting raises into {:error, msg}
  tensor.ex            Single source of truth for Nx ↔ Burn dtype mapping
  error.ex             Structured ExBurn.Error exception
  burn_bridge.ex       Direct tensor API bypassing Nx
  cubecl_bridge.ex     ExCubecl buffers/kernels/pipelines
  dataset.ex           Split / normalize / one-hot / loaders
  model.ex             Compile, params, save/load, quantize, export
  training.ex          fit/3 loop: optimizers, schedules, clipping, callbacks
  serving.ex           Nx.Serving integration
  serving/server.ex    Nx.Serving callback implementation

native/ex_burn_nif/src/lib.rs   All Rust NIF entry points (Burn/CubeCL ops)

test/                     Unit + integration tests (mirrors lib layout)
examples/*.exs            Runnable end-to-end scripts (mix run examples/…)
bench/*.exs               Benchmark scripts (Nx vs ExBurn)
guides/*.md               User-facing guides (also rendered in hexdocs)
```

## Adding a New Operation (Walkthrough)

Say you want to add `ExBurn.Nif.neg_tensor/1`-style support for a new op,
`my_op`. Touch each layer, top-down:

1. **Rust** — implement and register the NIF:

   ```rust
   // native/ex_burn_nif/src/lib.rs
   #[rustler::nif]
   fn nif_my_op(a: ResourceArc<TensorResource>) -> ResourceArc<TensorResource> {
       // validate ranks/dtypes explicitly — panics surface as
       // {:error, "Erlang error: :nif_panicked"} on the Elixir side
       ...
   }
   ```

   Add `nif_my_op/1` to the `rustler::init!` list at the bottom of the file.

2. **Stub** — declare it in `ExBurn.Nif`:

   ```elixir
   def my_op(a), do: nif_my_op(a)
   def nif_my_op(_a), do: :erlang.nif_error(:nif_not_loaded)
   ```

3. **Safe wrapper** — add a guarded version in `ExBurn.NifHelper`:

   ```elixir
   def my_op(a), do: guard(:my_op, [a])   # returns {:ok, ref} | {:error, msg}
   ```

4. **Backend callback** (if the op should be reachable from Nx) — implement the
   matching `Nx.Backend` callback in `ExBurn.Backend`. Callbacks accept both
   `%Nx.Tensor{}` operands and bare `%ExBurn.Backend{}` structs (the defn
   compiler passes the latter); see the existing dual-head clauses for the
   pattern.

5. **Dtype discipline** — the NIF stores `f32` only. Convert inputs through
   `ExBurn.Tensor.nx_to_burn/1` / `burn_to_nx/1`; never reinterpret bytes.

6. **Tests** — add:
   - a value-checking test in `test/nif_api_test.exs`,
   - a callback test in `test/backend_callbacks_test.exs` (if applicable),
   - keep in mind NIF constraints: `matmul` needs rank ≥ 2, `pow` takes a
     scalar exponent, `transpose_tensor` is 2-D only, `broadcast` only expands
     existing dimensions.

7. **Docs** — flip the feature row in the README status table if user-facing,
   and add an entry under `[Unreleased]` in `CHANGELOG.md`.

## Debugging Tips

- **NIF panics**: set `RUST_BACKTRACE=1` before running tests to get a Rust
  stack trace instead of a bare `:nif_panicked`.
- **Stale NIF after changing Rust code**: `mix compile` recompiles the crate,
  but if the VM holds the old `.so`, restart IEx/the test run.
- **GPU not detected**: check `ExBurn.device_info()` — if `gpu_available` is
  `false`, the NIF was likely built CPU-only (see `./build.sh` above).
- **Dtype errors**: `ExBurn.Error` with `dtype ... is not supported` means a
  non-f32 dtype crossed the NIF boundary directly; convert via `Nx.as_type/2`
  or go through the backend, which value-converts automatically.

## How to Contribute

- **Bug reports**: open an issue with steps to reproduce, expected behavior,
  and actual behavior. Include `ExBurn.summary()` output.
- **Feature requests**: open an issue describing the feature and its use case.
- **Pull requests**: open a PR against `main`.

### PR Checklist

- [ ] `mix test` green
- [ ] `mix test --cover` meets the configured threshold
- [ ] `mix lint` clean (`mix format` applied)
- [ ] `cargo fmt -- --check && cargo clippy -- -D warnings` clean
- [ ] New public functions have `@doc`, `@spec`, and tests
- [ ] `CHANGELOG.md` updated under `[Unreleased]`

## Code Style

- Elixir: follow the Elixir Style Guide, enforced via `mix format`
- Rust: follow `rustfmt` and `clippy` defaults
- Prefer multi-clause functions with pattern matching over `if/else` chains
- Every NIF call crosses a trust boundary — wrap raised errors into
  `{:error, reason}` tuples (see `NifHelper.guard/2`) or raise structured
  `ExBurn.Error`s; never let raw Rust panics escape public APIs

## License

By contributing, you agree that your contributions will be licensed under
Apache 2.0.
