//! A native calls a Lua function with `CallbackAction::CallThen`, either
//! consuming its results in a continuation or handing them straight to its
//! own caller.

use tcvm::env::{ActionFn, Error, Function, LuaString, NativeClosure, Stack, Value};
use tcvm::lua::Context;
use tcvm::vm::native::{CallbackAction, OnOk, Protect};
use tcvm::{Executor, LoadError, Lua};

/// Native callback `bumper(f)` calls `f()` then adds 1 to the result.
fn bumper<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    stack: Stack<'gc, '_>,
) -> Result<CallbackAction, Error<'gc>> {
    if stack.get(0).get_function().is_none() {
        return Err(Error::from_str(ctx, "bumper expects a function"));
    }
    // The callee at stack[0] with no arguments is already call layout.
    Ok(CallbackAction::call_then(0, add_one))
}

/// Takes the result at slot 0 and adds 1.
fn add_one<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
    _status: Result<(), Error<'gc>>,
) -> Result<CallbackAction, Error<'gc>> {
    let v = stack.get(0).get_integer().unwrap_or(0);
    stack.replace(&[Value::integer(ctx.mutation(), v + 1)]);
    Ok(CallbackAction::Return)
}

/// Native callback `forward(f, ...)`: `f(...)`'s results are the caller's.
fn forward<'gc>(
    _ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    _stack: Stack<'gc, '_>,
) -> Result<CallbackAction, Error<'gc>> {
    Ok(CallbackAction::CallThen {
        at: 0,
        protect: Protect::No,
        ok: OnOk::Return,
        cont: returned,
    })
}

fn returned<'gc>(
    _ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    _stack: Stack<'gc, '_>,
    _status: Result<(), Error<'gc>>,
) -> Result<CallbackAction, Error<'gc>> {
    Ok(CallbackAction::Return)
}

#[test]
fn native_calls_lua_then_post_processes() {
    let mut lua = Lua::new();

    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let bumper_fn = Function::new_action(ctx.mutation(), bumper, &[]);
            let key = Value::string(LuaString::new(ctx, b"bumper"));
            ctx.globals().raw_set(ctx, key, Value::function(bumper_fn));
            let chunk = ctx.load(
                "local function f() return 41 end\n\
                 return bumper(f)",
                Some("native_calls_lua"),
            )?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    let result: i64 = lua.execute(&ex).expect("run");
    assert_eq!(result, 42);
}

#[test]
fn native_call_without_then_returns_to_caller() {
    let mut lua = Lua::new();

    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let f = Function::new_action(ctx.mutation(), forward, &[]);
            let key = Value::string(LuaString::new(ctx, b"forward"));
            ctx.globals().raw_set(ctx, key, Value::function(f));
            let chunk = ctx.load(
                "local a, b, c = forward(function(x) return x, x + 1 end, 1)\n\
                 return (a == 1 and b == 2 and c == nil) and 1 or 0",
                Some("native_calls_lua"),
            )?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    let result: i64 = lua.execute(&ex).expect("run");
    assert_eq!(result, 1);
}

fn run_with_natives(src: &str) -> Result<i64, String> {
    let mut lua = Lua::new();
    lua.load_all();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            for (name, f) in [
                ("bumper", bumper as ActionFn),
                ("forward", forward as ActionFn),
            ] {
                let f = Function::new_action(ctx.mutation(), f, &[]);
                let key = Value::string(LuaString::new(ctx, name.as_bytes()));
                ctx.globals().raw_set(ctx, key, Value::function(f));
            }
            let chunk = ctx.load(src, Some("=t"))?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    lua.execute(&ex).map_err(|e| match e {
        tcvm::RuntimeError::Lua(stashed) => lua.enter(|ctx| {
            let s = ctx.fetch(&stashed).value().get_string().unwrap();
            String::from_utf8_lossy(s.as_bytes()).into_owned()
        }),
        other => panic!("{other:?}"),
    })
}

/// Tail-called from a metamethod, the native's follow-up sequence lands the
/// result through the metamethod's continuation.
#[test]
fn metamethod_tail_calls_a_native_that_calls_lua() {
    let src = "local t = setmetatable({}, { __index = function(t, k) \
               return bumper(function() return 41 end) end }) \
               return t.x";
    assert_eq!(run_with_natives(src), Ok(42));
}

/// Without a follow-up sequence the callee returns through the metamethod's
/// continuation itself.
#[test]
fn metamethod_tail_calls_a_native_that_forwards_to_lua() {
    let src = "local t = setmetatable({}, { __index = function(t, k) \
               return forward(function() return 1 end) end }) \
               return t.x";
    assert_eq!(run_with_natives(src), Ok(1));
}
