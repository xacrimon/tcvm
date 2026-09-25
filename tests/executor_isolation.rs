//! Each `Executor` runs on its own thread, so starting one must not disturb
//! another that is suspended or holding results (#6).

use tcvm::env::{Error, Function, LuaString, NativeClosure, NativeFn, Stack, Value};
use tcvm::vm::sequence::CallbackAction;
use tcvm::{Context, Executor, Lua, RuntimeError};

fn yielder<'gc>(
    _ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    _stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    Ok(CallbackAction::yield_(None))
}

#[test]
fn start_preserves_unread_results() {
    let mut lua = Lua::new();
    let a = lua.enter(|ctx| {
        let chunk = ctx.load("return 1", Some("a")).unwrap();
        ctx.stash(Executor::start(ctx, chunk, ()))
    });
    lua.finish(&a).unwrap();
    let b = lua.enter(|ctx| {
        let chunk = ctx.load("return 2", Some("b")).unwrap();
        ctx.stash(Executor::start(ctx, chunk, ()))
    });
    assert_eq!(lua.execute::<i64>(&b).unwrap(), 2);
    assert_eq!(
        lua.try_enter(|ctx| ctx.fetch(&a).take_result::<i64>(ctx))
            .unwrap(),
        1
    );
}

#[test]
fn start_preserves_suspended_executor() {
    let mut lua = Lua::new();
    let a = lua.enter(|ctx| {
        let y = Function::new_native(ctx.mutation(), yielder as NativeFn, Box::new([]));
        let key = Value::string(LuaString::new(ctx, b"yielder"));
        ctx.globals().raw_set(ctx, key, Value::function(y));
        let chunk = ctx
            .load("local x = 10\nyielder()\nreturn x + 1", Some("a"))
            .unwrap();
        ctx.stash(Executor::start(ctx, chunk, ()))
    });
    assert!(matches!(lua.finish(&a), Err(RuntimeError::MainYielded)));
    let b = lua.enter(|ctx| {
        let chunk = ctx.load("return 2", Some("b")).unwrap();
        ctx.stash(Executor::start(ctx, chunk, ()))
    });
    assert_eq!(lua.execute::<i64>(&b).unwrap(), 2);
    lua.resume(&a, ()).unwrap();
    assert_eq!(
        lua.try_enter(|ctx| ctx.fetch(&a).take_result::<i64>(ctx))
            .unwrap(),
        11
    );
}

/// Another executor's main thread looks like PUC's main thread does from a
/// coroutine, so it can't be resumed or closed out from under its executor.
#[test]
fn other_executors_main_thread_is_normal() {
    let mut lua = Lua::new();
    lua.load_all();
    let a = lua.enter(|ctx| {
        let y = Function::new_native(ctx.mutation(), yielder as NativeFn, Box::new([]));
        let key = Value::string(LuaString::new(ctx, b"yielder"));
        ctx.globals().raw_set(ctx, key, Value::function(y));
        let chunk = ctx
            .load(
                "other = coroutine.running()\nyielder()\nreturn 1",
                Some("a"),
            )
            .unwrap();
        ctx.stash(Executor::start(ctx, chunk, ()))
    });
    assert!(matches!(lua.finish(&a), Err(RuntimeError::MainYielded)));
    let b = lua.enter(|ctx| {
        let chunk = ctx
            .load(
                "local r = {coroutine.resume(other)}\n\
                 r[#r + 1] = coroutine.status(other)\n\
                 for _, v in ipairs({pcall(coroutine.close, other)}) do r[#r + 1] = v end\n\
                 r[#r + 1] = coroutine.isyieldable(other)\n\
                 for i, v in ipairs(r) do r[i] = tostring(v) end\n\
                 return table.concat(r, '|')",
                Some("b"),
            )
            .unwrap();
        ctx.stash(Executor::start(ctx, chunk, ()))
    });
    lua.finish(&b).unwrap();
    let got = lua.enter(|ctx| {
        let s = ctx.fetch(&b).take_result::<LuaString>(ctx).unwrap();
        String::from_utf8(s.as_bytes().to_vec()).unwrap()
    });
    assert_eq!(
        got,
        "false|cannot resume non-suspended coroutine|normal|false|cannot close a normal coroutine|false"
    );
    lua.resume(&a, ()).unwrap();
    assert_eq!(
        lua.try_enter(|ctx| ctx.fetch(&a).take_result::<i64>(ctx))
            .unwrap(),
        1
    );
}
