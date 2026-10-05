//! When the main thread yields to the host, `Lua::finish` / `execute`
//! must surface that as `RuntimeError::MainYielded` rather than
//! silently producing a `BadMode` from `take_result` (which is what
//! happens if `finish` returns `Ok` on a yielded executor).

use tcvm::env::{Error, Function, LuaString, NativeClosure, Stack, Value};
use tcvm::vm::native::CallbackAction;
use tcvm::{Context, Executor, LoadError, Lua, RuntimeError};

/// A native callback that yields to its resumer (the host, when called
/// from the main thread).
fn yielder<'gc>(
    _ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    _stack: Stack<'gc, '_>,
) -> Result<CallbackAction, Error<'gc>> {
    Ok(CallbackAction::Yield)
}

#[test]
fn main_thread_yield_surfaces_as_main_yielded() {
    let mut lua = Lua::new();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let y = Function::new_action(ctx.mutation(), yielder, &[]);
            let key = Value::string(LuaString::new(ctx, b"yielder"));
            ctx.globals().raw_set(ctx, key, Value::function(y));
            let chunk = ctx.load("return yielder()", Some("main_yield"))?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    let err = lua.finish(&ex).expect_err("main yield should error");
    assert!(
        matches!(err, RuntimeError::MainYielded),
        "expected MainYielded, got {err:?}"
    );
}
