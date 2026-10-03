//! `coroutine.close(co)` validation: succeeds for dead/suspended threads,
//! returns `(nil, msg)` for non-suspended (Normal / currently-running).

use tcvm::env::LuaString;
use tcvm::{Executor, LoadError, Lua, RuntimeError};

use crate::common::{start_on, yielding_lua};

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
fn close_normal_resumer_raises() {
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
                     local ok, msg = pcall(coroutine.close, outer)\n\
                     saw_msg = (ok == false) and msg\n\
                   end)\n\
                   coroutine.resume(inner)\n\
                 end)\n\
                 coroutine.resume(outer)\n\
                 if saw_msg == 'cannot close a normal coroutine' then return 1 else return 0 end",
                Some("close_normal"),
            )?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    let result: i64 = lua.execute(&ex).expect("run");
    assert_eq!(result, 1);
}

/// After explicit `close`, the thread is dead — a subsequent `resume`
/// should land in the standard dead-coroutine path (false-prefixed),
/// not BadMode and not panic on the executor invariant.
///
/// Workaround for #64 (`free_reg out of order` in the compiler): the
/// natural form is `local ok, msg = coroutine.resume(co); if not ok
/// and msg == 'cannot resume dead coroutine' then ...`, but that
/// chunk shape (multiple preceding discarded calls + multi-local +
/// `if not x`) trips a register-allocator assertion in the compiler,
/// independent of coroutines. We bind only `ok` here. Once #64 is
/// fixed, this test should switch to checking both `ok == false` and
/// the dead-coroutine message inline. The message itself is also
/// covered by `coroutine_resume_misuse::resume_dead_coroutine_returns_false`,
/// so behaviour coverage is intact in the meantime.
#[test]
fn resume_after_close_is_dead() {
    let mut lua = Lua::new();
    lua.load_all();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load(
                "local co = coroutine.create(function() coroutine.yield() end)\n\
                 coroutine.resume(co)\n\
                 coroutine.close(co)\n\
                 local ok = coroutine.resume(co)\n\
                 if ok == false then return 1 end\n\
                 return 0",
                Some("resume_after_close"),
            )?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    let result: i64 = lua.execute(&ex).expect("run");
    assert_eq!(result, 1);
}

/// Closing itself ends the coroutine as if it returned nothing.
#[test]
fn close_running_self_ends_it() {
    let mut lua = Lua::new();
    lua.load_all();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load(
                "local co\n\
                 local after\n\
                 co = coroutine.create(function()\n\
                   coroutine.close(co)\n\
                   after = true\n\
                 end)\n\
                 local r = table.pack(coroutine.resume(co))\n\
                 if r.n == 1 and r[1] == true and after == nil and coroutine.status(co) == 'dead' then return 1 else return 0 end",
                Some("close_self"),
            )?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    let result: i64 = lua.execute(&ex).expect("run");
    assert_eq!(result, 1);
}

#[test]
fn close_closes_open_upvalues() {
    // Closures that escaped keep working on their own copy of `x`, and a
    // value stored through one survives a full collection (#189). Expected
    // output from lua 5.5.1 with `collectgarbage()` for `yielder()`.
    let mut lua = yielding_lua();
    let ex = start_on(
        &mut lua,
        "local co = coroutine.create(function()\n\
           local x = 42\n\
           coroutine.yield(function() return x end, function(v) x = v end)\n\
         end)\n\
         local _, get, set = coroutine.resume(co)\n\
         local closed = coroutine.close(co)\n\
         local before = get()\n\
         set({tag = 'kept'})\n\
         yielder()\n\
         return tostring(closed) .. ' ' .. before .. ' ' .. get().tag",
    );
    let mut step = lua.finish(&ex);
    while let Err(RuntimeError::MainYielded) = step {
        lua.collect_all();
        step = lua.resume(&ex, ());
    }
    step.expect("run");
    let result = lua.enter(|ctx| {
        let s = ctx
            .fetch(&ex)
            .take_result::<LuaString>(ctx)
            .expect("result");
        String::from_utf8_lossy(s.as_bytes()).into_owned()
    });
    assert_eq!(result, "true 42 kept");
}
