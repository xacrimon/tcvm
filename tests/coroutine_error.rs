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

/// An errored coroutine transitions to 'dead' (issue #86): status is 'dead'
/// and a second resume reports 'cannot resume dead coroutine'. Also locks in
/// the no-regression cases: a normally-returned coroutine is 'dead' and a
/// yielded one is 'suspended'.
#[test]
fn errored_coroutine_is_dead() {
    let mut lua = Lua::new();
    lua.load_all();

    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load(
                "local co = coroutine.create(function() error('x') end)\n\
                 coroutine.resume(co)\n\
                 if coroutine.status(co) ~= 'dead' then return 10 end\n\
                 local ok2, e2 = coroutine.resume(co)\n\
                 if ok2 ~= false then return 11 end\n\
                 if e2 ~= 'cannot resume dead coroutine' then return 12 end\n\
                 local done = coroutine.create(function() return 1 end)\n\
                 coroutine.resume(done)\n\
                 if coroutine.status(done) ~= 'dead' then return 13 end\n\
                 local yld = coroutine.create(function() coroutine.yield() end)\n\
                 coroutine.resume(yld)\n\
                 if coroutine.status(yld) ~= 'suspended' then return 14 end\n\
                 return 1",
                Some("errored_coroutine_is_dead"),
            )?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    let result: i64 = lua.execute(&ex).expect("run");
    assert_eq!(
        result, 1,
        "errored coroutine dead + 2nd-resume msg + no-regress"
    );
}

/// A multi-level unwind (outer resumes a wrapped inner that errors; the
/// re-raise unwinds outer with no catcher) kills the outer coroutine too.
#[test]
fn nested_error_kills_outer_coroutine() {
    let mut lua = Lua::new();
    lua.load_all();

    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load(
                "local outer = coroutine.create(function()\n\
                 \x20 local w = coroutine.wrap(function() error('inner') end)\n\
                 \x20 w()\n\
                 end)\n\
                 local ok = coroutine.resume(outer)\n\
                 if ok ~= false then return 20 end\n\
                 if coroutine.status(outer) ~= 'dead' then return 21 end\n\
                 return 1",
                Some("nested_error_kills_outer_coroutine"),
            )?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    let result: i64 = lua.execute(&ex).expect("run");
    assert_eq!(
        result, 1,
        "outer coroutine is 'dead' after inner error unwinds it"
    );
}

/// `coroutine.close` on a coroutine that died via error re-surfaces the
/// killing error as `(false, err)` (and only once — a second close is
/// `true`). A normally-completed coroutine still closes with `true`.
#[test]
fn close_error_dead_returns_false_and_error() {
    let mut lua = Lua::new();
    lua.load_all();

    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load(
                "local c = coroutine.create(function() error('x') end)\n\
                 coroutine.resume(c)\n\
                 local ok, e = coroutine.close(c)\n\
                 if ok ~= false then return 30 end\n\
                 if e ~= 'x' then return 31 end\n\
                 if coroutine.close(c) ~= true then return 32 end\n\
                 local d = coroutine.create(function() return 1 end)\n\
                 coroutine.resume(d)\n\
                 if coroutine.close(d) ~= true then return 33 end\n\
                 local t = coroutine.create(function() error({code=7}) end)\n\
                 coroutine.resume(t)\n\
                 local ok2, v = coroutine.close(t)\n\
                 if ok2 ~= false then return 34 end\n\
                 if type(v) ~= 'table' or v.code ~= 7 then return 35 end\n\
                 return 1",
                Some("close_error_dead_returns_false_and_error"),
            )?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    let result: i64 = lua.execute(&ex).expect("run");
    assert_eq!(result, 1, "close surfaces death error as (false, err)");
}

/// Re-invoking a `coroutine.wrap` closure after its body errored re-raises
/// 'cannot resume dead coroutine' (gated like `resume`) instead of aborting
/// the executor. The error is observed via a guard coroutine so `resume`
/// catches it.
#[test]
fn wrap_recall_after_error_reports_dead() {
    let mut lua = Lua::new();
    lua.load_all();

    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load(
                "local w = coroutine.wrap(function() error('boom') end)\n\
                 local g1 = coroutine.create(w)\n\
                 local ok1 = coroutine.resume(g1)\n\
                 if ok1 ~= false then return 40 end\n\
                 local g2 = coroutine.create(w)\n\
                 local ok2, e2 = coroutine.resume(g2)\n\
                 if ok2 ~= false then return 41 end\n\
                 if e2 ~= 'cannot resume dead coroutine' then return 42 end\n\
                 return 1",
                Some("wrap_recall_after_error_reports_dead"),
            )?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    let result: i64 = lua.execute(&ex).expect("run");
    assert_eq!(result, 1, "wrap re-call after error reports dead");
}
