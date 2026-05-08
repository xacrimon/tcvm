//! Errors propagate correctly across two coroutine layers: A resumes B,
//! B resumes C, C errors, B re-raises (once it sees `(false, _)`),
//! A's `coroutine.resume` catches the error as `(false, msg)`.
//!
//! Without Lua-source `error()` (still `todo!()`), B's re-raise is
//! delegated to a small native helper.
//!
//! Workaround for #37 (MultRet propagation in nested call chains): the
//! natural form would be `reraise(coroutine.resume(c))`, propagating the
//! `(false, msg)` pair. Today TCVM emits `ret=2` for the inner call
//! instead of `MULTRET`, so only the first value reaches `reraise` and
//! the message is lost. Once #37 is fixed, this test should switch to:
//!
//!   * `reraise` taking `(ok, msg)` and erroring with `msg` when `ok`
//!     is false (instead of the no-arg always-erroring helper),
//!   * the chunk doing `reraise(coroutine.resume(c))` directly,
//!   * the assertion verifying `msg == "deep boom"` end-to-end (the
//!     real two-level test, not the indirected "two-level" sentinel).

use tcvm::env::{Error, Function, LuaString, NativeContext, NativeFn, Stack, Value};
use tcvm::vm::sequence::CallbackAction;
use tcvm::{Executor, LoadError, Lua};

fn deep_boomer<'gc>(
    nctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    Err(Error::from_str(nctx.ctx, "deep boom"))
}

/// Always errors with a fixed payload. Stand-in for Lua's `error()`,
/// used to re-raise from B once it sees its inner resume failed.
fn always_err<'gc>(
    nctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    Err(Error::from_str(nctx.ctx, "two-level"))
}

#[test]
fn two_level_error_propagation() {
    let mut lua = Lua::new();
    lua.load_all();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let boom = Function::new_native(ctx.mutation(), deep_boomer as NativeFn, Box::new([]));
            ctx.globals().raw_set(
                ctx,
                Value::string(LuaString::new(ctx, b"boom")),
                Value::function(boom),
            );
            let err = Function::new_native(ctx.mutation(), always_err as NativeFn, Box::new([]));
            ctx.globals().raw_set(
                ctx,
                Value::string(LuaString::new(ctx, b"reraise")),
                Value::function(err),
            );

            // Chain: chunk → resume(b) → resume(c) → boom() raises.
            // c yields no catcher, propagates to b's resume → (false, msg).
            // b inspects ok; if false, calls reraise() — which raises a
            // fresh error in b. b yields no catcher; propagates to
            // chunk's resume → (false, "two-level").
            let chunk = ctx.load(
                "local c = coroutine.create(function() boom() end)\n\
                 local b = coroutine.create(function()\n\
                   local ok = coroutine.resume(c)\n\
                   if ok == false then reraise() end\n\
                 end)\n\
                 local ok = coroutine.resume(b)\n\
                 if ok == false then return 1 else return 0 end",
                Some("two_level_err"),
            )?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    let result: i64 = lua.execute(&ex).expect("run");
    assert_eq!(result, 1);
}
