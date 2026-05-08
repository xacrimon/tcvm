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
