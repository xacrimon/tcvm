//! `coroutine.close(co)` validation: succeeds for dead/suspended threads,
//! returns `(nil, msg)` for non-suspended (Normal / currently-running).

use tcvm::{Executor, LoadError, Lua};

#[test]
fn close_suspended_returns_true() {
    let mut lua = Lua::new();
    lua.load_all();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load(
                "local co = coroutine.create(function() coroutine.yield() end)\n\
                 coroutine.resume(co)\n\
                 local ok = coroutine.close(co)\n\
                 if ok == true and coroutine.status(co) == 'dead' then return 1 else return 0 end",
                Some("close_suspended"),
            )?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    let result: i64 = lua.execute(&ex).expect("run");
    assert_eq!(result, 1);
}

#[test]
fn close_dead_returns_true() {
    let mut lua = Lua::new();
    lua.load_all();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load(
                "local co = coroutine.create(function() return 1 end)\n\
                 coroutine.resume(co)\n\
                 local ok = coroutine.close(co)\n\
                 if ok == true then return 1 else return 0 end",
                Some("close_dead"),
            )?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    let result: i64 = lua.execute(&ex).expect("run");
    assert_eq!(result, 1);
}

#[test]
fn close_normal_resumer_returns_nil_msg() {
    // Inner closes outer (which is on the stack as a resumer, status Normal).
    let mut lua = Lua::new();
    lua.load_all();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load(
                "local outer\n\
                 local saw_msg\n\
                 outer = coroutine.create(function()\n\
                   local inner = coroutine.create(function()\n\
                     local ok, msg = coroutine.close(outer)\n\
                     saw_msg = (ok == nil) and msg\n\
                   end)\n\
                   coroutine.resume(inner)\n\
                 end)\n\
                 coroutine.resume(outer)\n\
                 if saw_msg == 'cannot close a non-suspended coroutine' then return 1 else return 0 end",
                Some("close_normal"),
            )?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    let result: i64 = lua.execute(&ex).expect("run");
    assert_eq!(result, 1);
}

#[test]
fn close_running_self_returns_nil_msg() {
    let mut lua = Lua::new();
    lua.load_all();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load(
                "local co\n\
                 local saw_msg\n\
                 co = coroutine.create(function()\n\
                   local ok, msg = coroutine.close(co)\n\
                   saw_msg = (ok == nil) and msg\n\
                 end)\n\
                 coroutine.resume(co)\n\
                 if saw_msg == 'cannot close a non-suspended coroutine' then return 1 else return 0 end",
                Some("close_self"),
            )?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    let result: i64 = lua.execute(&ex).expect("run");
    assert_eq!(result, 1);
}
