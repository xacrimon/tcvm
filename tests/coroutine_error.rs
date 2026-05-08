//! Errors thrown inside a coroutine surface as `(false, msg)` from
//! `coroutine.resume`, courtesy of `PCallSequence`'s error handler.

use tcvm::env::{Error, Function, LuaString, NativeContext, NativeFn, Stack, Value};
use tcvm::vm::sequence::CallbackAction;
use tcvm::{Executor, LoadError, Lua};

/// A native callback that always errors.
fn boomer<'gc>(
    nctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    Err(Error::from_str(nctx.ctx, "boom"))
}

/// Coroutine calls `boomer()` (a native that errors); resume sees `(false, "boom")`.
#[test]
fn native_error_inside_coroutine() {
    let mut lua = Lua::new();
    lua.load_all();

    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let boom = Function::new_native(ctx.mutation(), boomer as NativeFn, Box::new([]));
            let key = Value::string(LuaString::new(ctx, b"boom"));
            ctx.globals().raw_set(ctx, key, Value::function(boom));

            let chunk = ctx.load(
                "local co = coroutine.create(function() boom() end)\n\
                 local ok, msg = coroutine.resume(co)\n\
                 if ok then return -1\n\
                 elseif msg == 'boom' then return 1\n\
                 else return 0 end",
                Some("native_error_in_coro"),
            )?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    let result: i64 = lua.execute(&ex).expect("run");
    assert_eq!(result, 1, "ok=false and msg=='boom'");
}
