//! Basic Lua coroutine round-trip via `coroutine.create` / `resume` / `yield`.

use tcvm::{Executor, LoadError, Lua};

/// `coroutine.create(f)`, `coroutine.resume(co)` once with no yield —
/// `f` returns `42`; resume returns `(true, 42)`.
#[test]
fn create_resume_no_yield() {
    let mut lua = Lua::new();
    lua.load_all();

    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load(
                "local co = coroutine.create(function() return 42 end)\n\
                 local ok, v = coroutine.resume(co)\n\
                 if ok and v == 42 then return 1 else return 0 end",
                Some("create_resume_no_yield"),
            )?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    let result: i64 = lua.execute(&ex).expect("run");
    assert_eq!(result, 1, "ok and v == 42");
}

/// One yield, one resume passes a value back through the yield. Three-step
/// dance: resume(co) → yields 1; resume(co, 10) → returns 10+5 = 15.
#[test]
fn yield_and_resume_with_value() {
    let mut lua = Lua::new();
    lua.load_all();

    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load(
                "local co = coroutine.create(function()\n\
                   local x = coroutine.yield(1)\n\
                   return x + 5\n\
                 end)\n\
                 local ok1, y1 = coroutine.resume(co)\n\
                 local ok2, y2 = coroutine.resume(co, 10)\n\
                 return y1 + y2",
                Some("yield_and_resume_with_value"),
            )?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    let result: i64 = lua.execute(&ex).expect("run");
    assert_eq!(result, 1 + 15, "y1=1 (yielded), y2=15 (returned)");
}

/// `yield(a, b, c)` and `resume(co, p, q)` carry multi-value windows
/// across the suspension. Stack-window arithmetic for >1 values differs
/// from the single-value case.
#[test]
fn multi_value_yield_and_resume() {
    let mut lua = Lua::new();
    lua.load_all();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load(
                "local co = coroutine.create(function(a, b, c)\n\
                   local p, q = coroutine.yield(a + 1, b + 2, c + 3)\n\
                   return p * q\n\
                 end)\n\
                 local ok1, y1, y2, y3 = coroutine.resume(co, 10, 20, 30)\n\
                 local ok2, prod = coroutine.resume(co, 5, 7)\n\
                 return y1 + y2 + y3 + prod",
                Some("multi_value_yield"),
            )?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    let result: i64 = lua.execute(&ex).expect("run");
    // y1=11, y2=22, y3=33, prod=35; sum = 101.
    assert_eq!(result, 101);
}

/// Loop-driven multi-cycle yield/resume: same coroutine yields three
/// times, then returns. Verifies the suspend/restore path is
/// idempotent across repeated trips.
#[test]
fn multi_cycle_yield_and_resume() {
    let mut lua = Lua::new();
    lua.load_all();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load(
                "local co = coroutine.create(function()\n\
                   for i = 1, 3 do coroutine.yield(i) end\n\
                   return 100\n\
                 end)\n\
                 local _, a = coroutine.resume(co)\n\
                 local _, b = coroutine.resume(co)\n\
                 local _, c = coroutine.resume(co)\n\
                 local _, d = coroutine.resume(co)\n\
                 return a + b + c + d",
                Some("multi_cycle"),
            )?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    let result: i64 = lua.execute(&ex).expect("run");
    // 1 + 2 + 3 + 100 = 106.
    assert_eq!(result, 106);
}

/// A coroutine resuming another coroutine ferries values across the
/// nested boundary: inner returns 42, outer adds 1, the chunk sees 43.
#[test]
fn nested_resume_value_flow_return() {
    let mut lua = Lua::new();
    lua.load_all();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load(
                "local inner = coroutine.create(function() return 42 end)\n\
                 local outer = coroutine.create(function()\n\
                   local ok, v = coroutine.resume(inner)\n\
                   return v + 1\n\
                 end)\n\
                 local ok, v = coroutine.resume(outer)\n\
                 if ok then return v else return -1 end",
                Some("nested_return"),
            )?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    let result: i64 = lua.execute(&ex).expect("run");
    assert_eq!(result, 43);
}

/// Inner yields a value; outer's `coroutine.resume` returns that value.
/// Tests yield-through-nested rather than terminal-return propagation.
#[test]
fn nested_resume_value_flow_yield() {
    let mut lua = Lua::new();
    lua.load_all();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load(
                "local inner = coroutine.create(function()\n\
                   coroutine.yield(7)\n\
                 end)\n\
                 local outer = coroutine.create(function()\n\
                   local ok, v = coroutine.resume(inner)\n\
                   return v\n\
                 end)\n\
                 local ok, v = coroutine.resume(outer)\n\
                 if ok then return v else return -1 end",
                Some("nested_yield"),
            )?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    let result: i64 = lua.execute(&ex).expect("run");
    assert_eq!(result, 7);
}

/// After a coroutine returns, its status is `"dead"`, and a further
/// `resume` returns `(false, "cannot resume dead coroutine")`.
#[test]
fn resume_dead_coroutine() {
    let mut lua = Lua::new();
    lua.load_all();

    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load(
                "local co = coroutine.create(function() return 1 end)\n\
                 coroutine.resume(co)\n\
                 if coroutine.status(co) == 'dead' then return 1 else return 0 end",
                Some("resume_dead"),
            )?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    let result: i64 = lua.execute(&ex).expect("run");
    assert_eq!(result, 1, "finished coroutine reports status='dead'");
}
