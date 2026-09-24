//! Basic test for `async_sequence` — a native callback that returns a
//! `Sequence` built from an `async move` block, calls a Lua function via
//! `.await`, and returns the result.

use tcvm::env::{Function, LuaString, NativeClosure, NativeFn, Stack, Value};
use tcvm::vm::async_sequence::{SequenceReturn, async_sequence};
use tcvm::vm::sequence::CallbackAction;
use tcvm::{Context, Executor, LoadError, Lua};

/// `pending().await` once, then return the constant 7. Validates the
/// minimal poll-resume cycle.
#[test]
fn async_pending_then_return() {
    let mut lua = Lua::new();

    fn make<'gc>(
        ctx: Context<'gc>,
        _closure: &NativeClosure<'gc>,
        mut stack: Stack<'gc, '_>,
    ) -> Result<CallbackAction<'gc>, tcvm::env::Error<'gc>> {
        let _ = &mut stack;
        let seq = async_sequence(ctx.mutation(), |_locals, mut seq| async move {
            seq.pending().await;
            seq.enter(|ctx, _locals, _exec, mut stack| {
                stack.replace(&[Value::integer(ctx.mutation(), 7)]);
            });
            Ok(SequenceReturn::Return)
        });
        Ok(CallbackAction::sequence(seq))
    }

    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let make_fn = Function::new_native(ctx.mutation(), make as NativeFn, Box::new([]));
            let key = Value::string(LuaString::new(ctx, b"makeseq"));
            ctx.globals().raw_set(ctx, key, Value::function(make_fn));
            let chunk = ctx.load("return makeseq()", Some("async_pending"))?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    let result: i64 = lua.execute(&ex).expect("run");
    assert_eq!(result, 7);
}

/// `guard(f)`: `call(f).await` and report whether it returned an error.
fn guard<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, tcvm::env::Error<'gc>> {
    let f = stack.get(0).get_function().expect("function");
    let mc = ctx.mutation();
    let seq = async_sequence(mc, move |locals, mut seq| {
        let f = locals.stash(mc, f);
        async move {
            let caught = seq.call(&f, 0).await.is_err();
            seq.enter(|_ctx, _locals, _exec, mut stack| {
                stack.replace(&[Value::boolean(caught)]);
            });
            Ok(SequenceReturn::Return)
        }
    });
    Ok(CallbackAction::sequence(seq))
}

/// An error raised by the awaited call is delivered to the future rather
/// than unwinding past the sequence.
#[test]
fn async_call_receives_callee_error() {
    let mut lua = Lua::new();
    lua.load_all();

    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let guard_fn = Function::new_native(ctx.mutation(), guard as NativeFn, Box::new([]));
            let key = Value::string(LuaString::new(ctx, b"guard"));
            ctx.globals().raw_set(ctx, key, Value::function(guard_fn));
            let chunk = ctx.load(
                "local caught = guard(function() error('x') end)\n\
                 local fine = guard(function() return 1 end)\n\
                 return (caught == true and fine == false) and 1 or 0",
                Some("async_call_error"),
            )?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    let result: i64 = lua.execute(&ex).expect("run");
    assert_eq!(result, 1);
}
