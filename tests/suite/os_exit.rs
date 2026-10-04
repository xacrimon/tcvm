//! `os.exit` hands its status to the host instead of ending the process.

use tcvm::{Executor, ExecutorMode, Lua, RuntimeError};

fn run(src: &str) -> (Lua, tcvm::StashedExecutor, Result<(), RuntimeError>) {
    let mut lua = Lua::new();
    lua.load_all();
    let ex = lua.enter(|ctx| {
        let chunk = ctx.load(src, Some("exit_test")).expect("load");
        ctx.stash(Executor::start(ctx, chunk, ()))
    });
    let res = lua.finish(&ex);
    (lua, ex, res)
}

#[test]
fn exit_status_reaches_host_past_catchers() {
    for (src, code) in [
        ("os.exit(3)", 3),
        ("os.exit(true)", 0),
        ("os.exit(false)", 1),
        ("pcall(os.exit, 4)", 4),
        ("xpcall(os.exit, print, 5)", 5),
        ("coroutine.wrap(function() pcall(os.exit, 6) end)()", 6),
    ] {
        let (mut lua, ex, res) = run(src);
        assert!(
            matches!(res, Err(RuntimeError::Exit(c)) if c == code),
            "{src}: {res:?}"
        );
        assert!(lua.enter(|ctx| ctx.fetch(&ex).mode() == ExecutorMode::Stopped));
    }
}

#[test]
fn exit_skips_pending_close() {
    // A coroutine that exits is reset: its `__close` never runs, even on a
    // later `coroutine.close`, and a closure over its locals keeps their values.
    let (mut lua, _ex, res) = run("_G.closed = 0\n\
         local x <close> = setmetatable({}, {__close = function() _G.closed = 1 end})\n\
         _G.co = coroutine.create(function()\n\
           local y <close> = setmetatable({}, {__close = function() _G.closed = 2 end})\n\
           local v = 42\n\
           _G.get = function() return v end\n\
           os.exit(0)\n\
         end)\n\
         coroutine.resume(co)");
    assert!(matches!(res, Err(RuntimeError::Exit(0))));
    let ex = lua.enter(|ctx| {
        let chunk = ctx
            .load(
                "return closed, coroutine.status(co) == 'dead', coroutine.close(co), closed, get()",
                Some("check"),
            )
            .expect("load");
        ctx.stash(Executor::start(ctx, chunk, ()))
    });
    let (before, dead, close_ok, after, v) = lua
        .execute::<(i64, bool, bool, i64, i64)>(&ex)
        .expect("check runs");
    assert_eq!(before, 0);
    assert!(dead);
    assert!(close_ok);
    assert_eq!(after, 0);
    assert_eq!(v, 42);
}
