//! Continuation natives (`call_then`, `resume`) and async natives
//! that call, yield and wait: where their results land and how errors reach
//! them.

use crate::env::{Error, Function, LuaString, NativeClosure, Stack, Value};
use crate::vm::async_native::{AsyncFn, Spawned};
use crate::vm::native::{ContFn, NativeOut, OnOk, Protect, cont};
use crate::{Context, Executor, LoadError, Lua, RuntimeError, StashedExecutor, StepResult};

fn start(
    lua: &mut Lua,
    src: &str,
    natives: &[(&str, ContFn)],
    asyncs: &[(&str, AsyncFn)],
) -> StashedExecutor {
    lua.try_enter(|ctx| -> Result<_, LoadError> {
        let fns = natives
            .iter()
            .map(|&(name, f)| (name, Function::new_cont(ctx.mutation(), f, &[])))
            .chain(
                asyncs
                    .iter()
                    .map(|&(name, f)| (name, Function::new_async(ctx.mutation(), f, &[]))),
            );
        for (name, f) in fns {
            let key = Value::string(LuaString::new(ctx, name.as_bytes()));
            ctx.globals().raw_set(ctx, key, Value::function(f));
        }
        let chunk = ctx.load(src, Some("=t"))?;
        Ok(ctx.stash(Executor::start(ctx, chunk, ())))
    })
    .expect("load")
}

fn run(src: &str, natives: &[(&str, ContFn)], asyncs: &[(&str, AsyncFn)]) -> i64 {
    let mut lua = Lua::new();
    lua.load_all();
    let ex = start(&mut lua, src, natives, asyncs);
    lua.execute(&ex).expect("run")
}

pub(crate) fn returned<'gc>(
    _ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    _stack: Stack<'gc, '_>,
    _status: Result<(), Error<'gc>>,
) -> NativeOut {
    NativeOut::RETURN
}

/// `tail_resume(co)`: whatever `co` returns goes straight to the caller.
fn tail_resume<'gc>(
    _ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    _stack: Stack<'gc, '_>,
) -> NativeOut {
    NativeOut::resume(0, cont::TEST_RETURNED, OnOk::Return)
}

#[test]
fn resumed_results_pass_through() {
    let src = "local co = coroutine.create(function() return 99 end)\n\
               return tail_resume(co)";
    assert_eq!(run(src, &[("tail_resume", tail_resume)], &[]), 99);
}

/// `forward(f)`: `f(41)`'s results are the caller's.
fn forward<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> NativeOut {
    let f = stack.get(0);
    stack.replace(&[f, Value::integer(ctx.mutation(), 41)]);
    NativeOut::call_then(0, cont::TEST_RETURNED, Protect::No, OnOk::Return)
}

#[test]
fn call_results_land_at_the_original_call() {
    let natives: &[(&str, ContFn)] = &[("forward", forward)];
    let src = "local function addone(x) return x + 1 end\n\
               return forward(addone)";
    assert_eq!(run(src, natives, &[]), 42);
    // Through `__call`.
    let src = "local addone = setmetatable({}, {__call = function(_, x) return x + 1 end})\n\
               return forward(addone)";
    assert_eq!(run(src, natives, &[]), 42);
}

/// `bumpr(co)`: resumes `co` and adds 1 to what it returns.
fn bumpr<'gc>(
    _ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    _stack: Stack<'gc, '_>,
) -> NativeOut {
    NativeOut::resume(0, cont::TEST_ADD_ONE, OnOk::Cont)
}

pub(crate) fn add_one<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
    status: Result<(), Error<'gc>>,
) -> NativeOut {
    status?;
    let v = stack.get(0).get_integer().expect("an integer");
    stack.replace(&[Value::integer(ctx.mutation(), v + 1)]);
    NativeOut::RETURN
}

#[test]
fn resume_then_post_processes() {
    let src = "local co = coroutine.create(function() return 41 end)\n\
               return bumpr(co)";
    assert_eq!(run(src, &[("bumpr", bumpr)], &[]), 42);
}

/// `call_boomer()`: calls `boomer`, a native that raises.
fn call_boomer<'gc>(
    _ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<Spawned, Error<'gc>> {
    Ok(stack.spawn(|cx| async move {
        cx.enter(|ctx, mut stack| {
            let boomer = Function::new_native(ctx.mutation(), boomer, &[]);
            stack.replace(&[Value::function(boomer)]);
        });
        cx.call(0).await;
        Ok(())
    }))
}

fn boomer<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    _stack: Stack<'gc, '_>,
) -> Result<(), Error<'gc>> {
    Err(Error::from_str(ctx, "boom"))
}

#[test]
fn a_native_error_in_an_async_call_reaches_resume() {
    let src = "local co = coroutine.create(function() call_boomer() end)\n\
               local ok, msg = coroutine.resume(co)\n\
               return ((not ok) and msg == 'boom') and 1 or 0";
    assert_eq!(run(src, &[], &[("call_boomer", call_boomer)]), 1);
}

/// `pend3()`: lets the host run three times, then returns 7.
fn pend3<'gc>(
    _ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<Spawned, Error<'gc>> {
    Ok(stack.spawn(|cx| async move {
        for _ in 0..3 {
            cx.pending().await;
        }
        cx.enter(|ctx, mut stack| stack.replace(&[Value::integer(ctx.mutation(), 7)]));
        Ok(())
    }))
}

#[test]
fn pending_surfaces_then_completes() {
    let mut lua = Lua::new();
    let ex = start(&mut lua, "return pend3()", &[], &[("pend3", pend3)]);
    let mut pendings = 0;
    loop {
        let done = lua
            .try_enter(|ctx| -> Result<_, RuntimeError> {
                Ok(match ctx.fetch(&ex).step(ctx)? {
                    StepResult::Done => true,
                    StepResult::Yielded(_) => panic!("unexpected Yielded"),
                    StepResult::Pending => false,
                })
            })
            .expect("step");
        if done {
            break;
        }
        pendings += 1;
        assert!(pendings < 100, "guard against runaway loop");
    }
    assert_eq!(pendings, 3);
    let result: i64 = lua
        .try_enter(|ctx| ctx.fetch(&ex).take_result::<i64>(ctx))
        .expect("take_result");
    assert_eq!(result, 7);
}

/// `yseq()`: yields 42, then returns what it was resumed with plus 1.
fn yseq<'gc>(
    _ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<Spawned, Error<'gc>> {
    Ok(stack.spawn(|cx| async move {
        cx.enter(|ctx, mut stack| stack.replace(&[Value::integer(ctx.mutation(), 42)]));
        cx.yield_(0).await;
        cx.enter(|ctx, mut stack| {
            let n = stack.get(0).get_integer().unwrap_or(0);
            stack.replace(&[Value::integer(ctx.mutation(), n + 1)]);
        });
        Ok(())
    }))
}

#[test]
fn yields_then_resumes() {
    let src = "local co = coroutine.create(function() return yseq() end)\n\
               local ok1, y1 = coroutine.resume(co)\n\
               local ok2, y2 = coroutine.resume(co, 100)\n\
               return y1 + y2";
    assert_eq!(run(src, &[], &[("yseq", yseq)]), 42 + 101);
}

/// `guard(f)`: whether `f()` raised, caught by `pcall`.
fn guard<'gc>(
    _ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<Spawned, Error<'gc>> {
    Ok(stack.spawn(|cx| async move {
        let caught = cx.pcall(0).await.is_err();
        cx.enter(|_, mut stack| stack.replace(&[Value::boolean(caught)]));
        Ok(())
    }))
}

#[test]
fn pcall_receives_the_callee_error() {
    let src = "local caught = guard(function() error('x') end)\n\
               local fine = guard(function() return 1 end)\n\
               return (caught == true and fine == false) and 1 or 0";
    assert_eq!(run(src, &[], &[("guard", guard)]), 1);
}
