defmodule ExBurn.Application do
  @moduledoc """
  Application callback for ExBurn.

  Starts the NIF loading supervision tree. The Rust NIF shared library
  is loaded on startup via `ExBurn.Nif.load_nif/0`.
  """
  use Application

  @impl true
  def start(_type, _args) do
    children = [
      # NIF loader — loads the Rust shared library at runtime
      ExBurn.Nif
    ]

    opts = [strategy: :one_for_one, name: ExBurn.Supervisor]
    Supervisor.start_link(children, opts)
  end
end
