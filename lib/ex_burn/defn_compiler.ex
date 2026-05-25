defmodule ExBurn.Defn.Compiler do
  @moduledoc """
  Custom `Nx.Defn.Compiler` that compiles defn expressions
  to run on the Burn GPU backend via NIF.

  This compiler traverses the `Nx.Defn.Expr` expression tree,
  converts it into Burn tensor operations, and executes them
  through the Rust NIF layer for GPU acceleration.

  ## Usage

      Nx.Defn.global_default_options(compiler: ExBurn.Defn.Compiler)

  Or per-function:

      defn my_fun(x, y, opts \\\\ []) do
        Nx.add(x, y)
      end
      compiler: ExBurn.Defn.Compiler

  ## How it works

  1. `Nx.Defn` traces the function body into an expression tree
     of `Nx.Defn.Expr` nodes.
  2. The compiler receives the expression tree via `__compile__/4`
     or `__jit__/5`.
  3. Each expression node is evaluated by converting parameters
     to Burn tensors, then dispatching the operation to the NIF.
  4. The result is converted back to `Nx.Tensor` for the caller.

  ## Options

    * `:garbage_collect` - when `true`, garbage collects after
      evaluating each node (default: `false`)
  """

  @behaviour Nx.Defn.Compiler

  alias Nx.Defn.{Composite, Expr, Tree}
  alias ExBurn.NifHelper, as: Nif

  @creation_ops [:eye, :iota, :from_binary]
  @list_ops [:concatenate, :stack]
  @indices_ops [:slice, :put_slice]

  @impl true
  def __partitions_options__(opts) do
    List.duplicate(opts, Keyword.get(opts, :max_concurrency, 1))
  end

  @impl true
  def __to_backend__(_opts) do
    {ExBurn.Backend, []}
  end

  @impl true
  def __jit__(_key, vars, fun, args_list, opts) do
    compile_and_run(vars, fun, args_list, opts)
  end

  @impl true
  def __compile__(_key, vars, fun, opts) do
    fn [params] ->
      {result, _} = run_eval(vars, fun, params, opts)
      [result]
    end
  end

  @impl true
  def __shard_jit__(_key, _mesh, _vars, _fun, _args_list, _opts) do
    raise "sharding is not supported by ExBurn.Defn.Compiler"
  end

  # ── Compilation / Execution ───────────────────────────────────────

  defp compile_and_run(vars, fun, args_list, opts) do
    for args <- args_list do
      {result, _} = run_eval(vars, fun, args, opts)
      result
    end
  end

  defp run_eval(vars, fun, params, opts) do
    hooks = Keyword.get(opts, :hooks, %{})
    gc? = Keyword.get(opts, :garbage_collect, false)

    # Build the expression tree by calling the fun with parameter vars
    {expr, output} =
      vars
      |> fun.()
      |> Composite.traverse([], &{Nx.devectorize(&1), [Nx.to_template(&1) | &2]})

    state = %{
      params: params,
      gc: gc?,
      hooks: hooks,
      cache: %{}
    }

    {result, _cache} = composite_eval(expr, state)
    result = apply_output(result, Enum.reverse(output))
    {result, %{}}
  end

  defp apply_output(result, output) do
    {result, []} =
      Composite.traverse(result, output, fn result, [out | acc] ->
        {%{out | data: result.data}, acc}
      end)

    result
  end

  # ── Expression Evaluation ─────────────────────────────────────────

  defp eval(%Nx.Tensor{data: %Expr{op: :tensor, args: [t]}}, state) do
    {t, state}
  end

  defp eval(%Nx.Tensor{data: %Expr{op: :constant, args: [constant]}} = ans, state) do
    burn_tensor = constant_to_burn(ans, constant)
    {burn_tensor, state}
  end

  defp eval(%Nx.Tensor{data: %Expr{op: :metadata, args: [expr, _meta]}}, state) do
    eval(expr, state)
  end

  defp eval(%Nx.Tensor{} = ans, state) do
    burn_tensor = nx_tensor_to_burn(ans)
    {burn_tensor, state}
  end

  defp eval(%Nx.Tensor{data: %Expr{id: id, op: op}} = ans, state) do
    state = maybe_gc(state)

    case state.cache do
      %{^id => cached} ->
        {cached, state}

      %{} ->
        {args, state} = Tree.apply_args(ans, state, &eval/2)
        {result, state} = eval_apply(op, args, ans, state)
        state = put_in(state.cache[id], result)
        {result, state}
    end
  end

  defp maybe_gc(%{gc: true} = state) do
    :erlang.garbage_collect(self())
    state
  end

  defp maybe_gc(state), do: state

  # ── Operation Dispatch ────────────────────────────────────────────

  defp eval_apply(:parameter, [i], _ans, state) do
    case Enum.fetch!(state.params, i).() do
      %Nx.Tensor{data: %Nx.Defn.Expr{}} = tensor ->
        raise ArgumentError,
              "cannot pass a tensor expression as argument to defn, got: #{inspect(tensor)}"

      %Nx.Tensor{} = tensor ->
        burn_tensor = nx_tensor_to_burn(tensor)
        {burn_tensor, state}
    end
  end

  defp eval_apply(:elem, [tuple, i], _ans, state) do
    {tuple, state} = composite_eval(tuple, state)
    {elem(tuple, i), state}
  end

  defp eval_apply(:attach_token, [token, expr], _ans, state) do
    {_, state} = eval(token, state)
    eval(expr, state)
  end

  defp eval_apply(:fun, [length, expr, _mfa], _ans, state) do
    fun =
      case length do
        1 ->
          fn arg1 ->
            params = [fn -> Nx.to_tensor(arg1) end]
            {result, _} = run_eval([], fn _ -> expr end, params, [])
            result
          end

        2 ->
          fn arg1, arg2 ->
            params = [fn -> Nx.to_tensor(arg1) end, fn -> Nx.to_tensor(arg2) end]
            {result, _} = run_eval([], fn _ -> expr end, params, [])
            result
          end
      end

    {fun, state}
  end

  defp eval_apply(:cond, [clauses_cache, last_cache, _parent_ids], _ans, state) do
    {chosen, state} = cond_clause(clauses_cache, last_cache, state)
    composite_eval(chosen, state)
  end

  defp eval_apply(:while, [initial, pred, block, _while_cache], _ans, state) do
    {initial, state} = composite_eval(initial, state)
    {result, state} = while_loop(initial, pred, block, state)
    {result, state}
  end

  defp eval_apply(:token, [exprs_hooks], _ans, state) do
    state =
      List.foldr(exprs_hooks, state, fn {expr, hook_fun}, state ->
        {res, state} = composite_eval(expr, state)
        hook_fun && hook_fun.(res)
        state
      end)

    {{}, state}
  end

  defp eval_apply(:block, [struct, in_args, _expr, callback], ans, state) do
    {in_args, state} = eval_block_args(in_args, state)

    out =
      case ans do
        %{type: {:tuple, _}} -> ans
        _ -> ans
      end

    {ExBurn.Backend.block(struct, out, in_args, callback), state}
  end

  defp eval_apply(:runtime_call, [expr, fun, out_template, opts], _ans, state) do
    {tensor_value, state} = composite_eval(expr, state)
    result = fun.(tensor_value, opts)

    case out_template do
      %Nx.Tensor{} -> {result, state}
      _ -> {[result] |> Composite.flatten_list() |> List.to_tuple(), state}
    end
  end

  defp eval_apply(op, args, ans, state) do
    {args, state} = prepare_op_args(ans, args, state)
    {result, state} = execute_op(op, args, ans, state)
    {result, state}
  end

  # ── Block Args ────────────────────────────────────────────────────

  defp eval_block_args(in_args, state) do
    {args, state} =
      Enum.reduce(in_args, {[], state}, fn
        arg, {acc, state} when is_list(arg) ->
          # keyword options, pass through
          {acc ++ [arg], state}

        arg, {acc, state} ->
          {result, state} = eval(arg, state)
          {[result | acc], state}
      end)

    {Enum.reverse(args), state}
  end

  # ── Op Argument Preparation ───────────────────────────────────────

  defp prepare_op_args(ans, args, state) do
    Tree.apply_args(put_in(ans.data.args, args), state, &eval/2)
  end

  # ── Op Execution ──────────────────────────────────────────────────

  defp execute_op(op, args, ans, state) do
    {mod, call_args} =
      cond do
        op in @creation_ops ->
          {ExBurn.Backend, [ans | args] ++ [[]]}

        op in @list_ops ->
          {ExBurn.Backend, [ans | args]}

        op in @indices_ops ->
          {ExBurn.Backend, [ans | args]}

        match?({:tuple, _}, ans.type) ->
          {ExBurn.Backend, args}

        true ->
          {ExBurn.Backend, [ans | args]}
      end

    result = apply(mod, op, call_args)
    {result, state}
  end

  # ── Control Flow ──────────────────────────────────────────────────

  defp while_loop(acc, condition, block, state) do
    state = %{state | params: composite_to_params(acc)}
    {pred, state} = eval(condition, state)

    if burn_to_number(pred) != 0 do
      {acc, state} = composite_eval(block, state)
      while_loop(acc, condition, block, state)
    else
      {acc, state}
    end
  end

  defp cond_clause([{{pred, body}, cache} | clauses], last_cache, state) do
    {pred, state} = eval(pred, %{state | cache: cache})

    if burn_to_number(pred) != 0 do
      {body, state}
    else
      cond_clause(clauses, last_cache, state)
    end
  end

  defp cond_clause([], {last, last_cache}, state) do
    {last, %{state | cache: last_cache}}
  end

  # ── Composite Evaluation ──────────────────────────────────────────

  defp composite_eval(composite, state) do
    Composite.traverse(composite, state, &eval/2)
  end

  defp composite_to_params(composite) do
    composite |> composite_to_params([]) |> Enum.reverse()
  end

  defp composite_to_params(tuple, acc) when is_tuple(tuple) do
    Enum.reduce(Tuple.to_list(tuple), acc, &composite_to_params/2)
  end

  defp composite_to_params(other, acc) do
    [fn -> other end | acc]
  end

  # ── Tensor Conversion ─────────────────────────────────────────────

  defp constant_to_burn(ans, constant) do
    shape = Tuple.to_list(Nx.shape(ans))
    type = nx_type_to_burn_type(Nx.type(ans))
    data = <<constant::float-32-native>>

    case Nif.new_tensor(data, shape, Atom.to_string(type)) do
      {:ok, ref} -> %ExBurn.Backend{ref: ref, shape: shape, type: type}
      {:error, _} -> %ExBurn.Backend{ref: make_ref(), shape: shape, type: type}
    end
  end

  defp nx_tensor_to_burn(%Nx.Tensor{} = tensor) do
    data = Nx.to_binary(tensor)
    shape = Tuple.to_list(Nx.shape(tensor))
    type = nx_type_to_burn_type(Nx.type(tensor))

    case Nif.new_tensor(data, shape, Atom.to_string(type)) do
      {:ok, ref} -> %ExBurn.Backend{ref: ref, shape: shape, type: type}
      {:error, _} -> %ExBurn.Backend{ref: make_ref(), shape: shape, type: type}
    end
  end

  defp burn_to_number(%ExBurn.Backend{ref: ref}) do
    case Nif.tensor_to_binary(ref) do
      {:ok, <<val::float-32-native>>} -> val
      _ -> 0
    end
  end

  defp burn_to_number(val) when is_number(val), do: val

  defp nx_type_to_burn_type({:f, 32}), do: :f32
  defp nx_type_to_burn_type({:f, 64}), do: :f64
  defp nx_type_to_burn_type({:f, 16}), do: :f16
  defp nx_type_to_burn_type({:bf, 16}), do: :bf16
  defp nx_type_to_burn_type({:s, 32}), do: :i32
  defp nx_type_to_burn_type({:s, 64}), do: :i64
  defp nx_type_to_burn_type({:s, 16}), do: :i16
  defp nx_type_to_burn_type({:s, 8}), do: :i8
  defp nx_type_to_burn_type({:u, 8}), do: :u8
  defp nx_type_to_burn_type(_), do: :f32
end
