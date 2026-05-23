defmodule ExBurn.MixProject do
  use Mix.Project

  @app :ex_burn
  @version "0.1.0"
  @github_url "https://github.com/ohhi-vn/ex_burn"

  def project do
    [
      app: @app,
      version: @version,
      elixir: "~> 1.18",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      description: "Middle layer between Nx and Burn for mobile ML training",
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
      {:nx, "~> 0.7"},
      # Rust NIF integration
      {:rustler, "~> 0.37.0", runtime: false},
      # Neural network library
      {:axon, "~> 0.7", optional: true},
      # Classical ML algorithms
      {:scholar, "~> 0.4", optional: true},
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
        native/ex_burn_nif/Cargo.toml
        native/ex_burn_nif/Cargo.lock
        native/ex_burn_nif/src
        mix.exs
        mix.lock
        README.md
        CHANGELOG.md
        README.md
        LICENSE
      ),
      maintainers: ["Manh Vu"]
    ]
  end

  defp docs do
    [
      main: "readme",
      extras: ["README.md"]
    ]
  end
end
