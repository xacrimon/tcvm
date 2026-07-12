//! A *native metamethod that suspends* — the `apply_native_continuation` path.
//!
//! A native metamethod runs inline with no Lua frame of its own, so it has
//! nowhere to park the `Continuation` that says where its result goes. If it
//! suspends, the executor stashes the continuation on the `CallSite` and
//! replays the payload by hand once the value arrives (`land_call_results` →
//! `apply_native_continuation`) against the *caller's* frame, whose registers
//! sit below the staging window the metamethod's args were placed in.
//!
//! Each test drives one continuation payload through a suspension:
//! `StoreResult` (`__index`), `CondJump` (`__lt`, where the resumed value must
//! steer a branch rather than land in a register), and `TForCall`.

use tcvm::env::{Error, Function, LuaString, NativeContext, NativeFn, Stack, Table, Value};
use tcvm::vm::sequence::CallbackAction;
use tcvm::{Executor, IntoMultiValue, LoadError, Lua, RuntimeError, StepResult};

/// A metamethod that refuses to answer inline: it yields to the host, which
/// supplies the result on resume.
fn suspending_mm<'gc>(
    _nctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    Ok(CallbackAction::Yield { then: None })
}

/// Loads `src` with a global `t` whose metatable maps `event` to the suspending
/// native, then runs to the first yield.
fn setup(event: &[u8], src: &str) -> (Lua, tcvm::StashedExecutor) {
    let mut lua = Lua::new();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let mm = Function::new_native(ctx.mutation(), suspending_mm as NativeFn, Box::new([]));
            let meta = Table::new(ctx);
            let ev = Value::string(LuaString::new(ctx, event));
            meta.raw_set(ctx, ev, Value::function(mm));

            let t = Table::new(ctx);
            t.set_metatable(ctx, Some(meta));
            let key = Value::string(LuaString::new(ctx, b"t"));
            ctx.globals().raw_set(ctx, key, Value::table(t));

            let chunk = ctx.load(src, Some("suspended_mm"))?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    (lua, ex)
}

/// Step to the yield, resume with `value` (which drives to completion), and
/// take the chunk's result.
fn yield_then_resume<A>(lua: &mut Lua, ex: &tcvm::StashedExecutor, value: A) -> i64
where
    A: for<'gc> IntoMultiValue<'gc>,
{
    lua.try_enter(|ctx| -> Result<(), RuntimeError> {
        match ctx.fetch(ex).step(ctx)? {
            StepResult::Yielded(_) => Ok(()),
            StepResult::Done => panic!("metamethod should have suspended, not completed"),
            StepResult::Pending => panic!("expected Yielded, got Pending"),
        }
    })
    .expect("step to the metamethod's yield");

    lua.resume(ex, value).expect("resume to completion");
    lua.try_enter(|ctx| ctx.fetch(ex).take_result::<i64>(ctx))
        .expect("take_result")
}

/// `StoreResult`: the resumed value must land in the register the GETFIELD
/// targeted, with the caller's other locals undisturbed by the staging window
/// that sat above them.
#[test]
fn suspended_index_stores_result_in_caller_register() {
    let (mut lua, ex) = setup(
        b"__index",
        "local a, b, c = 100, 200, 300\n\
         local v = t.missing\n\
         return v + a + b + c",
    );
    let got = yield_then_resume(&mut lua, &ex, (7i64,));
    assert_eq!(got, 7 + 100 + 200 + 300, "resumed __index value + locals");
}

/// `CondJump`: the resumed value decides a branch. A truthy answer can't be
/// "stored" anywhere — the executor has to bump the caller frame's saved `pc`
/// past the comparison's following JMP.
#[test]
fn suspended_lt_steers_the_branch() {
    let src = "if t < 5 then return 11 else return 22 end";
    let (mut lua, ex) = setup(b"__lt", src);
    assert_eq!(
        yield_then_resume(&mut lua, &ex, (true,)),
        11,
        "truthy __lt should take the then-branch"
    );

    let (mut lua, ex) = setup(b"__lt", src);
    assert_eq!(
        yield_then_resume(&mut lua, &ex, (false,)),
        22,
        "falsy __lt should take the else-branch"
    );
}

/// `TForCall`: a generic-for whose iterator is a native that suspends on every
/// call. This is the payload that lands *several* values at once, into the
/// loop's control registers. Two suspensions here — one per iterator call — so
/// it drives the executor directly rather than through `Lua::resume` (which
/// would surface the second yield as `MainYielded`).
#[test]
fn suspended_iterator_lands_loop_vars() {
    let mut lua = Lua::new();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let iter =
                Function::new_native(ctx.mutation(), suspending_mm as NativeFn, Box::new([]));
            let key = Value::string(LuaString::new(ctx, b"iter"));
            ctx.globals().raw_set(ctx, key, Value::function(iter));
            let chunk = ctx.load(
                "local sum, n = 0, 0\n\
                 for k, v in iter, nil, nil do\n\
                 \x20 sum = sum + k + v\n\
                 \x20 n = n + 1\n\
                 end\n\
                 return sum * 100 + n",
                Some("suspended_iter"),
            )?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");

    // First iterator call suspends; feed it (3, 4) → one loop pass.
    lua.try_enter(|ctx| -> Result<(), RuntimeError> {
        match ctx.fetch(&ex).step(ctx)? {
            StepResult::Yielded(_) => Ok(()),
            _ => panic!("expected the iterator to suspend"),
        }
    })
    .expect("first yield");

    // Second call also suspends; a nil control value ends the loop.
    lua.try_enter(|ctx| -> Result<(), RuntimeError> {
        let executor = ctx.fetch(&ex);
        executor.resume(ctx, (3i64, 4i64))?;
        match executor.step(ctx)? {
            StepResult::Yielded(_) => Ok(()),
            _ => panic!("expected the iterator to suspend a second time"),
        }
    })
    .expect("second yield");

    lua.try_enter(|ctx| -> Result<(), RuntimeError> {
        let executor = ctx.fetch(&ex);
        executor.resume(ctx, ())?;
        match executor.step(ctx)? {
            StepResult::Done => Ok(()),
            _ => panic!("nil control value should have ended the loop"),
        }
    })
    .expect("loop terminates");

    let got: i64 = lua
        .try_enter(|ctx| ctx.fetch(&ex).take_result::<i64>(ctx))
        .expect("take_result");
    // One pass with k=3, v=4: sum=7, n=1.
    assert_eq!(got, 7 * 100 + 1, "resumed (3,4) should land in k and v");
}

/// A suspended metamethod inside a *called* function: the continuation has to
/// resume into the callee's frame, not the main chunk's.
#[test]
fn suspended_index_inside_a_call() {
    let (mut lua, ex) = setup(
        b"__index",
        "local function f(n)\n\
         \x20 local pad1, pad2 = 1, 2\n\
         \x20 return t.missing + n + pad1 + pad2\n\
         end\n\
         return f(10)",
    );
    let got = yield_then_resume(&mut lua, &ex, (5i64,));
    assert_eq!(got, 5 + 10 + 1 + 2);
}
