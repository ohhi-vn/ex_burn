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
      # No NIF child needed — Rustler loads the NIF automatically
      # when ExBurn.Nif functions are first called
    ]

    opts = [strategy: :one_for_one, name: ExBurn.Supervisor]
    Supervisor.start_link(children, opts)
  end
end
