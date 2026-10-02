//! `coroutine.running` / `isyieldable` / `status` honor the actual
//! running thread via `Execution::current_thread`.

use tcvm::{Executor, LoadError, Lua};

/// On the main thread, `running()` reports `is_main = true` and
/// `isyieldable()` is false.
#[test]
fn main_thread_introspection() {
    let mut lua = Lua::new();
    lua.load_all();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load(
                "local _, is_main = coroutine.running()\n\
                 local yieldable = coroutine.isyieldable()\n\
                 if is_main and not yieldable then return 1 else return 0 end",
                Some("main_introspection"),
            )?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    let result: i64 = lua.execute(&ex).expect("run");
    assert_eq!(
        result, 1,
        "main: is_main should be true and isyieldable false"
    );
}

/// Inside a coroutine body, `running()` reports `is_main = false`,
/// `isyieldable()` is true, and `status(co)` is `"running"` when the
/// argument is the running coroutine itself (or its handle).
#[test]
fn coroutine_introspection() {
    let mut lua = Lua::new();
    lua.load_all();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load(
                "local co\n\
                 co = coroutine.create(function()\n\
                   local self_thread, is_main = coroutine.running()\n\
                   local y = coroutine.isyieldable()\n\
                   local s = coroutine.status(co)\n\
                   if (not is_main) and y and s == 'running' then\n\
                     return 1\n\
                   else\n\
                     return 0\n\
                   end\n\
                 end)\n\
                 local ok, v = coroutine.resume(co)\n\
                 if ok then return v else return -1 end",
                Some("coro_introspection"),
            )?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    let result: i64 = lua.execute(&ex).expect("run");
    assert_eq!(result, 1, "coro: not main, yieldable, status=running");
}

/// Yielded and freshly-created coroutines report `"suspended"`; finished
/// ones report `"dead"`.
#[test]
fn status_lifecycle() {
    let mut lua = Lua::new();
    lua.load_all();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load(
                "local co = coroutine.create(function() coroutine.yield() end)\n\
                 local s_pre = coroutine.status(co)  -- suspended (fresh)\n\
                 coroutine.resume(co)\n\
                 local s_yielded = coroutine.status(co)  -- suspended (yielded)\n\
                 coroutine.resume(co)\n\
                 local s_dead = coroutine.status(co)  -- dead\n\
                 if s_pre == 'suspended' and s_yielded == 'suspended' and s_dead == 'dead' then\n\
                   return 1\n\
                 else\n\
                   return 0\n\
                 end",
                Some("status_lifecycle"),
            )?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    let result: i64 = lua.execute(&ex).expect("run");
    assert_eq!(result, 1, "status should walk suspended → suspended → dead");
}

/// A coroutine that has resumed another sees the inner as `"running"`
/// and itself (when checked from the inner) as `"normal"`.
#[test]
fn nested_status() {
    let mut lua = Lua::new();
    lua.load_all();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load(
                "local outer\n\
                 local inner_status\n\
                 outer = coroutine.create(function()\n\
                   local inner = coroutine.create(function()\n\
                     inner_status = coroutine.status(outer)\n\
                   end)\n\
                   coroutine.resume(inner)\n\
                 end)\n\
                 coroutine.resume(outer)\n\
                 if inner_status == 'normal' then return 1 else return 0 end",
                Some("nested_status"),
            )?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    let result: i64 = lua.execute(&ex).expect("run");
    assert_eq!(result, 1, "outer should be reported 'normal' from inner");
}

/// `coroutine.isyieldable(co)` with an explicit thread arg: false for the
/// main thread, true for any non-main coroutine handle.
#[test]
fn isyieldable_explicit_arg() {
    let mut lua = Lua::new();
    lua.load_all();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load(
                "local main = coroutine.running()\n\
                 local co = coroutine.create(function() end)\n\
                 if coroutine.isyieldable(main) == false\n\
                    and coroutine.isyieldable(co) == true then\n\
                   return 1\n\
                 else\n\
                   return 0\n\
                 end",
                Some("isyieldable_explicit"),
            )?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    let result: i64 = lua.execute(&ex).expect("run");
    assert_eq!(result, 1);
}

/// `yield()` and `resume(co)` with no values: yield returns nil, the
/// resumed function continues with no resume-args.
#[test]
fn zero_value_yield_resume_round_trip() {
    let mut lua = Lua::new();
    lua.load_all();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load(
                "local co = coroutine.create(function()\n\
                   local x = coroutine.yield()\n\
                   if x == nil then return 7 else return -1 end\n\
                 end)\n\
                 coroutine.resume(co)\n\
                 local ok, v = coroutine.resume(co)\n\
                 if ok then return v else return -2 end",
                Some("zero_value_yield"),
            )?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    let result: i64 = lua.execute(&ex).expect("run");
    assert_eq!(result, 7);
}
