# Contributing to ExBurn

Thank you for your interest in contributing!

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/ex_burn.git`
3. Install dependencies: `mix deps.get`
4. Run tests: `mix test`
5. Run the Rust checks: `cd native/ex_burn_nif && cargo fmt -- --check && cargo clippy -- -D warnings`

## Prerequisites

- Elixir ~> 1.18
- OTP 27+
- Rust (stable toolchain) — required for NIF compilation
- For iOS development: Xcode, `aarch64-apple-ios` target (`rustup target add aarch64-apple-ios`)
- For Android development: Android NDK, `aarch64-linux-android` target

## How to Contribute

- **Bug reports**: Open an issue with steps to reproduce, expected behavior, and actual behavior
- **Feature requests**: Open an issue describing the feature and its use case
- **Pull requests**: Open a PR against `main`. Ensure tests pass and code is formatted (`mix format` + `cargo fmt`)

## Code Style

- Elixir: Follow the Elixir Style Guide, enforced via `mix format`
- Rust: Follow `rustfmt` and `clippy` defaults
- Add `@moduledoc` and `@spec` to all public functions

## License

By contributing, you agree that your contributions will be licensed under Apache 2.0.
