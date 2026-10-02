//! `coroutine.resume(co, ...)` returns `(false, msg)` (catchable in Lua)
//! when `co` isn't resumable: dead, currently running, an active resumer,
//! or the main thread.

use tcvm::{Executor, LoadError, Lua};

#[test]
fn resume_dead_coroutine_returns_false() {
    let mut lua = Lua::new();
    lua.load_all();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load(
                "local co = coroutine.create(function() return 1 end)\n\
                 local ok1 = coroutine.resume(co)\n\
                 local ok2, msg = coroutine.resume(co)\n\
                 if ok1 == true and ok2 == false and msg == 'cannot resume dead coroutine' then\n\
                   return 1\n\
                 else\n\
                   return 0\n\
                 end",
                Some("resume_dead"),
            )?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    let result: i64 = lua.execute(&ex).expect("run");
    assert_eq!(result, 1);
}

#[test]
fn resume_running_self_returns_false() {
    let mut lua = Lua::new();
    lua.load_all();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load(
                "local co\n\
                 co = coroutine.create(function() return coroutine.resume(co) end)\n\
                 local ok, inner_ok = coroutine.resume(co)\n\
                 if ok and inner_ok == false then return 1 else return 0 end",
                Some("self_resume"),
            )?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    let result: i64 = lua.execute(&ex).expect("run");
    assert_eq!(result, 1);
}

#[test]
fn resume_main_thread_returns_false() {
    let mut lua = Lua::new();
    lua.load_all();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load(
                "local main = coroutine.running()\n\
                 local ok, msg = coroutine.resume(main)\n\
                 if ok == false and msg == 'cannot resume non-suspended coroutine' then\n\
                   return 1\n\
                 else\n\
                   return 0\n\
                 end",
                Some("resume_main"),
            )?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    let result: i64 = lua.execute(&ex).expect("run");
    assert_eq!(result, 1);
}

#[test]
fn resume_normal_resumer_returns_false() {
    // Inner resumes outer (which is mid-resume on the executor's thread
    // stack with status Normal). Should fail with non-suspended.
    let mut lua = Lua::new();
    lua.load_all();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load(
                "local outer\n\
                 local saw_msg\n\
                 outer = coroutine.create(function()\n\
                   local inner = coroutine.create(function()\n\
                     local ok, msg = coroutine.resume(outer)\n\
                     saw_msg = (not ok) and msg\n\
                   end)\n\
                   coroutine.resume(inner)\n\
                 end)\n\
                 coroutine.resume(outer)\n\
                 if saw_msg == 'cannot resume non-suspended coroutine' then return 1 else return 0 end",
                Some("resume_normal"),
            )?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    let result: i64 = lua.execute(&ex).expect("run");
    assert_eq!(result, 1);
}
