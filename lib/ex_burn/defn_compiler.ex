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
  alias ExBurn.Error
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

  # Zips evaluated results with their output templates. Bare `%ExBurn.Backend{}`
  # leaves are wrapped into the corresponding template tensor. We cannot use
  # `Composite.traverse` on the raw result here because bare backend structs are
  # not Nx tensors, so we walk composites ourselves.
  defp apply_output(result, output) do
    {result, []} = zip_output(result, output)
    result
  end

  defp zip_output(%ExBurn.Backend{} = data, [%Nx.Tensor{} = out | acc]) do
    {%{out | data: data}, acc}
  end

  defp zip_output(tuple, output) when is_tuple(tuple) do
    {list, acc} = tuple |> Tuple.to_list() |> Enum.map_reduce(output, &zip_output/2)
    {List.to_tuple(list), acc}
  end

  defp zip_output(map, output) when is_map(map) and not is_struct(map) do
    {pairs, acc} =
      map
      |> Enum.sort_by(&elem(&1, 0))
      |> Enum.map_reduce(output, fn {key, value}, acc ->
        {value, acc} = zip_output(value, acc)
        {{key, value}, acc}
      end)

    {Map.new(pairs), acc}
  end

  defp zip_output(other, [_out | acc]) do
    {other, acc}
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

  # Generic expression nodes must be matched BEFORE the bare-tensor
  # fallback below — %Nx.Tensor{} also matches Expr-backed tensors.
  defp eval(%Nx.Tensor{data: %Expr{id: id, op: op}} = ans, state) do
    state = maybe_gc(state)

    case state.cache do
      %{^id => cached} ->
        {cached, state}

      %{} ->
        {result, state} =
          case op do
            # Closure-bearing nodes must be dispatched with their RAW args —
            # recursing into closure bodies would evaluate their internal
            # parameters against the outer scope's params list.
            op when op in [:fun, :while] ->
              eval_apply(op, ans.data.args, ans, state)

            _ ->
              {args, state} = Tree.apply_args(ans, state, &eval/2)
              eval_apply(op, args, ans, state)
          end

        state = put_in(state.cache[id], result)
        {result, state}
    end
  end

  defp eval(%Nx.Tensor{} = ans, state) do
    burn_tensor = nx_tensor_to_burn(ans)
    {burn_tensor, state}
  end

  # Already-evaluated leaves — bare backend structs from an earlier pass,
  # plain numbers, or traced closures — pass through unchanged.
  defp eval(%ExBurn.Backend{} = data, state), do: {data, state}

  defp eval(other, state) when not is_struct(other, Nx.Tensor), do: {other, state}

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

      # Already-evaluated loop-carried values pass straight through
      %ExBurn.Backend{} = data ->
        {data, state}

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

  defp eval_apply(:cond, [clauses, last], _ans, state) when is_list(clauses) do
    # nx >= 0.13 hands us already-evaluated {pred, body} pairs plus the
    # final fallback value — pick the first branch whose pred is truthy.
    chosen =
      Enum.find_value(clauses, fn {pred, body} ->
        if burn_to_number(pred) != 0, do: body
      end) || last

    composite_eval(chosen, state)
  end

  defp eval_apply(:while, [initial, arg, condition, body], _ans, state) do
    # nx >= 0.13 node layout: [flatten_initial, flatten_arg, condition, body].
    # The loop variable arrives as a parameter expression whose *global*
    # index must be honoured when binding per-iteration params.
    loop_idx =
      case arg do
        %Nx.Tensor{data: %Nx.Defn.Expr{op: :parameter, args: [i]}} -> i
        _ -> 0
      end

    {initial_val, state} = composite_eval(initial, state)
    {result, state} = while_loop(initial_val, condition, body, state, loop_idx)
    {result, state}
  end

  defp eval_apply(:hook, [expr, callback_spec, _user_template, _ref], _ans, state) do
    # nx >= 0.13 node layout: [tensor_expr, callback_spec, template, ref].
    # The traced spec carries only the hook name — resolve the runtime
    # callback from state.hooks.
    {value, state} = composite_eval(expr, state)

    case callback_spec do
      {:named, name, nil} when is_atom(name) ->
        case Map.get(state.hooks, name) do
          fun when is_function(fun, 1) -> fun.(to_hook_value(value))
          _ -> :ok
        end

      {:named, _name, fun} when is_function(fun, 1) ->
        fun.(to_hook_value(value))

      fun when is_function(fun, 1) ->
        fun.(to_hook_value(value))

      _ ->
        :ok
    end

    {value, state}
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

  defp eval_apply(:fun, [param_exprs, expr, _mfa], _ans, state) do
    # nx >= 0.13 node layout: [param_exprs, body_expr, mfa]. Closure
    # parameters carry *global* indices — pad the local thunk list so
    # Enum.fetch!/2 inside :parameter lands on the right thunk.
    params_list = List.wrap(param_exprs)
    arity = length(params_list)

    offset =
      case params_list do
        [%Nx.Tensor{data: %Nx.Defn.Expr{op: :parameter, args: [i]}} | _] -> i
        _ -> 0
      end

    fun =
      case arity do
        1 ->
          fn arg1 ->
            thunks = [fn -> to_eval_input(arg1) end]
            {result, _} = run_eval([], fn _ -> expr end, pad_params(offset, thunks), [])
            to_plain_tensor(result)
          end

        2 ->
          fn arg1, arg2 ->
            thunks = [fn -> to_eval_input(arg1) end, fn -> to_eval_input(arg2) end]

            {result, _} = run_eval([], fn _ -> expr end, pad_params(offset, thunks), [])
            to_plain_tensor(result)
          end

        other ->
          raise "closures with arity #{other} are not supported by ExBurn.Defn.Compiler"
      end

    {fun, state}
  end

  # Host-side callbacks (e.g. Nx.reduce on BinaryBackend) must stay in
  # plain-tensor land — convert evaluator outputs back before returning.
  # Note: our backend_transfer is an identity op, so the round-trip has to
  # go through the stored bytes.
  defp eval_apply(op, args, ans, state) do
    {result, state} = execute_op(op, args, ans, state)
    {result, state}
  end

  defp to_plain_tensor(%Nx.Tensor{data: %ExBurn.Backend{}} = t) do
    t
    |> Nx.to_binary()
    |> Nx.from_binary(t.type)
    |> Nx.reshape(Nx.shape(t))
  end

  defp to_plain_tensor(%Nx.Tensor{} = t), do: t

  defp to_plain_tensor(%ExBurn.Backend{shape: shape, type: type} = data) do
    wrapped = %Nx.Tensor{
      data: data,
      type: ExBurn.Tensor.burn_to_nx(type),
      shape: List.to_tuple(shape),
      names: List.duplicate(nil, length(shape))
    }

    Nx.backend_transfer(wrapped, Nx.BinaryBackend)
  end

  defp to_plain_tensor(other), do: other

  defp to_eval_input(%Nx.Tensor{} = t), do: t
  defp to_eval_input(%ExBurn.Backend{} = data), do: data
  defp to_eval_input(other), do: Nx.to_tensor(other)

  defp pad_params(0, thunks), do: thunks
  defp pad_params(n, thunks) when n > 0, do: List.duplicate(fn -> Nx.tensor(0.0) end, n) ++ thunks

  defp to_hook_value(%Nx.Tensor{} = t), do: t

  defp to_hook_value(%ExBurn.Backend{shape: shape, type: type} = data) do
    %Nx.Tensor{
      data: data,
      type: ExBurn.Tensor.burn_to_nx(type),
      shape: List.to_tuple(shape),
      names: List.duplicate(nil, length(shape))
    }
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

    # Backend callbacks written for Nx dispatch expect %Nx.Tensor{} operands;
    # the evaluator produces bare %ExBurn.Backend{} structs — wrap them.
    result = apply(mod, op, normalize_backend_args(call_args))
    {result, state}
  end

  defp normalize_backend_args(args) do
    Enum.map(args, fn
      %ExBurn.Backend{} = data -> wrap_bare_arg(data)
      other -> other
    end)
  end

  defp wrap_bare_arg(%ExBurn.Backend{shape: shape, type: type} = data) do
    %Nx.Tensor{
      data: data,
      type: ExBurn.Tensor.burn_to_nx(type),
      shape: List.to_tuple(shape),
      names: List.duplicate(nil, length(shape))
    }
  end

  # ── Control Flow ──────────────────────────────────────────────────

  defp while_loop(acc, condition, block, state, idx) do
    # Fresh cache every iteration — cached subexpression results belong to
    # the previous accumulator and would otherwise freeze the loop.
    thunks = composite_to_params(acc)
    params = List.duplicate(fn -> Nx.tensor(0.0) end, idx) ++ thunks
    state = %{state | params: params, cache: %{}}
    {pred, state} = eval(condition, state)

    if burn_to_number(pred) != 0 do
      {acc, state} = composite_eval(block, state)
      while_loop(acc, condition, block, state, idx)
    else
      {acc, state}
    end
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

    # Replicate the scalar to fill the shape — the Rust side validates
    # byte size against the declared shape.
    data =
      <<constant * 1.0::float-32-native>>
      |> :binary.copy(max(Enum.product(shape), 1))

    case Nif.new_tensor(data, shape, Atom.to_string(type)) do
      {:ok, ref} -> %ExBurn.Backend{ref: ref, shape: shape, type: type}
      {:error, reason} -> raise Error, op: :constant, reason: reason
    end
  end

  defp nx_tensor_to_burn(%Nx.Tensor{} = tensor) do
    shape = Tuple.to_list(Nx.shape(tensor))
    type = nx_type_to_burn_type(Nx.type(tensor))

    # Cast to f32 (value-preserving) since the NIF stores f32 only.
    data =
      tensor
      |> Nx.as_type({:f, 32})
      |> Nx.to_binary()

    case Nif.new_tensor(data, shape, Atom.to_string(type)) do
      {:ok, ref} -> %ExBurn.Backend{ref: ref, shape: shape, type: type}
      {:error, reason} -> raise Error, op: :parameter, reason: reason
    end
  end

  defp burn_to_number(%ExBurn.Backend{ref: ref}) do
    case Nif.tensor_to_binary(ref) do
      {:ok, <<val::float-32-native>>} -> val
      _ -> 0
    end
  end

  defp burn_to_number(val) when is_number(val), do: val

  # Delegates to the canonical mapping in ExBurn.Tensor.
  defp nx_type_to_burn_type(type), do: ExBurn.Tensor.nx_to_burn(type)
end
