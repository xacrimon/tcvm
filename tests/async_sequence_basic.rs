//! Basic test for `async_sequence` — a native callback that returns a
//! `Sequence` built from an `async move` block, calls a Lua function via
//! `.await`, and returns the result.

use tcvm::env::{Function, LuaString, NativeContext, NativeFn, Stack, Value};
use tcvm::vm::async_sequence::{SequenceReturn, async_sequence};
use tcvm::vm::sequence::CallbackAction;
use tcvm::{Executor, LoadError, Lua};

/// `pending().await` once, then return the constant 7. Validates the
/// minimal poll-resume cycle.
#[test]
fn async_pending_then_return() {
    let mut lua = Lua::new();

    fn make<'gc>(
        nctx: NativeContext<'gc, '_>,
        mut stack: Stack<'gc, '_>,
    ) -> Result<CallbackAction<'gc>, tcvm::env::Error<'gc>> {
        let _ = &mut stack;
        let seq = async_sequence(nctx.ctx.mutation(), |_locals, mut seq| async move {
            seq.pending().await;
            seq.enter(|_ctx, _locals, _exec, mut stack| {
                stack.replace(&[Value::integer(7)]);
            });
            Ok(SequenceReturn::Return)
        });
        Ok(CallbackAction::Sequence(seq))
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
