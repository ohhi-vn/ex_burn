defmodule ExBurn.MixProject do
  use Mix.Project

  @app :ex_burn
  @version "0.4.2"
  @github_url "https://github.com/ohhi-vn/ex_burn"

  def project do
    [
      app: @app,
      version: @version,
      elixir: "~> 1.18",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      description: "Middle layer between Nx and Burn for ML training",
      package: package(),
      source_url: @github_url,
      docs: docs(),
      rustler_crates: rustler_crates()
    ]
  end

  def application do
    [
      extra_applications: [:logger],
      mod: {ExBurn.Application, []}
    ]
  end

  defp deps do
    [
      # Nx tensor computation
      {:nx, ">= 0.12.0 and < 2.0.0"},
      # Rust NIF integration
      {:rustler, "~> 0.37.0", runtime: false},
      # Neural network library
      {:axon, "~> 0.8", optional: true},
      # Classical ML algorithms
      {:scholar, "~> 0.4", optional: true},
      # CubeCL GPU backend
      {:ex_cubecl, ">= 0.5.0 and < 2.0.0", optional: true},
      # JSON encoding (used by ExCubecl for kernel params)
      {:jason, "~> 1.4"},
      # Documentation
      {:ex_doc, "~> 0.40", only: :dev, runtime: false}
    ]
  end

  defp rustler_crates do
    [
      ex_burn_nif: [
        path: "native/ex_burn_nif",
        mode: if(Mix.env() == :prod, do: :release, else: :debug)
      ]
    ]
  end

  defp package do
    [
      licenses: ["Apache-2.0"],
      links: %{
        "GitHub" => @github_url,
        "Changelog" => "#{@github_url}/blob/main/CHANGELOG.md"
      },
      files: ~w(
        lib
        priv
        native/ex_burn_nif/Cargo.toml
        native/ex_burn_nif/Cargo.lock
        native/ex_burn_nif/src
        guides
        examples
        .formatter.exs
        mix.exs
        mix.lock
        README.md
        CHANGELOG.md
        CONTRIBUTING.md
        LICENSE
      ),
      maintainers: ["Manh Vu"]
    ]
  end

  defp docs do
    [
      main: "readme",
      extras: [
        "README.md",
        "guides/01_getting_started.md",
        "guides/02_training.md",
        "guides/03_mobile_deployment.md",
        "guides/04_architecture.md",
        "guides/05_training_optimization.md",
        "guides/06_deep_learning_guide.md",
        "guides/07_benchmarks.md"
      ],
      groups_for_extras: [
        Guides: ~r/guides\//
      ]
    ]
  end
end
