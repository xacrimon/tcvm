//! `coroutine.wrap(f)` returns a callable that resumes the underlying
//! coroutine on each call: yields propagate as return values, errors
//! rethrow (in contrast to `resume`'s catch-and-wrap behavior).

use tcvm::env::{Error, Function, LuaString, NativeContext, NativeFn, Stack, Value};
use tcvm::vm::sequence::CallbackAction;
use tcvm::{Executor, LoadError, Lua, RuntimeError};

/// Wrap a generator that yields ascending values then returns. Each
/// invocation of the wrapper drives one resume cycle; values surface
/// without the `(ok, ...)` tuple.
#[test]
fn wrap_happy_path_yields_then_returns() {
    let mut lua = Lua::new();
    lua.load_all();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load(
                "local f = coroutine.wrap(function(x)\n\
                   local y = coroutine.yield(x + 1)\n\
                   return y + 100\n\
                 end)\n\
                 local a = f(5)        -- yielded 6\n\
                 local b = f(20)       -- returned 120\n\
                 return a + b",
                Some("wrap_happy"),
            )?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    let result: i64 = lua.execute(&ex).expect("run");
    assert_eq!(result, 126, "a=6 (yielded), b=120 (returned)");
}

fn boomer<'gc>(
    nctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    Err(Error::from_str(nctx.ctx, "boom"))
}

/// A wrapped function that errors propagates the error to the wrap-caller
/// (no `(false, msg)` wrapping). With no Lua `pcall` yet, the error
/// surfaces all the way to the host as `RuntimeError::Lua`.
#[test]
fn wrap_rethrows_error_to_caller() {
    let mut lua = Lua::new();
    lua.load_all();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let boom = Function::new_native(ctx.mutation(), boomer as NativeFn, Box::new([]));
            let key = Value::string(LuaString::new(ctx, b"boom"));
            ctx.globals().raw_set(ctx, key, Value::function(boom));

            let chunk = ctx.load(
                "local f = coroutine.wrap(function() boom() end)\n\
                 f()",
                Some("wrap_rethrow"),
            )?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    let err = lua.execute::<()>(&ex).expect_err("wrap should rethrow");
    let stashed = match err {
        RuntimeError::Lua(s) => s,
        other => panic!("expected RuntimeError::Lua, got {other:?}"),
    };
    let msg = lua.enter(|ctx| {
        let e = ctx.fetch(&stashed);
        let s = e.value().get_string().expect("error payload is a string");
        std::str::from_utf8(s.as_bytes()).unwrap().to_owned()
    });
    assert_eq!(msg, "boom");
}
