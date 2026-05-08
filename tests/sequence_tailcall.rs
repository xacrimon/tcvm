//! A `Sequence` returning `SequencePoll::TailCall` lands the call's
//! results at the original CALL's `func_idx` slot — not at `bottom`,
//! and not at `bottom + 1`. This guards a slot-offset regression in
//! `pump_sequence`.

use std::pin::Pin;

use tcvm::dmm::{Collect, Trace};
use tcvm::env::{Error, Function, LuaString, NativeContext, NativeFn, Stack, Value};
use tcvm::lua::Context;
use tcvm::vm::sequence::{BoxSequence, CallbackAction, Execution, Sequence, SequencePoll};
use tcvm::{Executor, LoadError, Lua};

/// Reads `target` from stack[0], replaces the stack window with the
/// integer 41, then `TailCall`s `target` (a Lua function `f(x) = x+1`).
struct TailCallSeq;

unsafe impl<'gc> Collect<'gc> for TailCallSeq {
    const NEEDS_TRACE: bool = false;
}

impl<'gc> Sequence<'gc> for TailCallSeq {
    fn trace_pointers(&self, _cc: &mut dyn Trace<'gc>) {}

    fn poll(
        self: Pin<&mut Self>,
        _ctx: Context<'gc>,
        _exec: Execution<'gc, '_>,
        mut stack: Stack<'gc, '_>,
    ) -> Result<SequencePoll<'gc>, Error<'gc>> {
        let target = stack
            .get(0)
            .get_function()
            .expect("stack[0] should be a function");
        stack.replace(&[Value::integer(41)]);
        Ok(SequencePoll::TailCall(target))
    }
}

/// `forward(f)` becomes a `Sequence` that tail-calls `f(41)`.
fn forward<'gc>(
    nctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let seq = BoxSequence::new(nctx.ctx.mutation(), TailCallSeq);
    Ok(CallbackAction::Sequence(seq))
}

#[test]
fn sequence_tailcall_lands_at_original_func_idx() {
    let mut lua = Lua::new();

    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let forward_fn =
                Function::new_native(ctx.mutation(), forward as NativeFn, Box::new([]));
            let key = Value::string(LuaString::new(ctx, b"forward"));
            ctx.globals().raw_set(ctx, key, Value::function(forward_fn));
            let chunk = ctx.load(
                "local function addone(x) return x + 1 end\n\
                 return forward(addone)",
                Some("seq_tailcall"),
            )?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    let result: i64 = lua.execute(&ex).expect("run");
    assert_eq!(
        result, 42,
        "TailCall result should land where the caller expected"
    );
}
