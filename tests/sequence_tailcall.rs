//! A `Sequence` returning `SequencePoll::TailCall` lands the call's
//! results at the original CALL's `func_idx` slot — not at `bottom`,
//! and not at `bottom + 1`. This guards a slot-offset regression in
//! `pump_sequence`.

use std::pin::Pin;

use tcvm::dmm::{Collect, Trace};
use tcvm::env::{Error, Function, LuaString, NativeClosure, NativeFn, Stack, Value};
use tcvm::lua::Context;
use tcvm::vm::sequence::{BoxSequence, CallbackAction, Execution, Sequence, SequencePoll};
use tcvm::{Executor, LoadError, Lua};

/// Reads `target` from stack[0], replaces the stack window with the
/// integer 41, then `TailCall`s `target`.
struct TailCallSeq;

unsafe impl<'gc> Collect<'gc> for TailCallSeq {
    const NEEDS_TRACE: bool = false;
}

impl<'gc> Sequence<'gc> for TailCallSeq {
    fn trace_pointers(&self, _cc: &mut dyn Trace<'gc>) {}

    fn poll(
        self: Pin<&mut Self>,
        ctx: Context<'gc>,
        _exec: Execution<'gc>,
        mut stack: Stack<'gc, '_>,
    ) -> Result<SequencePoll<'gc>, Error<'gc>> {
        let target = stack.get(0);
        stack.replace(&[Value::integer(ctx.mutation(), 41)]);
        Ok(SequencePoll::TailCall(target))
    }
}

/// `forward(f)` becomes a `Sequence` that tail-calls `f(41)`.
fn forward<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    _stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let seq = BoxSequence::new(ctx.mutation(), TailCallSeq);
    Ok(CallbackAction::sequence(seq))
}

/// Runs `src` with `forward` installed as a global.
fn run(src: &str) -> i64 {
    let mut lua = Lua::new();
    lua.load_all();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let forward_fn =
                Function::new_native(ctx.mutation(), forward as NativeFn, Box::new([]));
            let key = Value::string(LuaString::new(ctx, b"forward"));
            ctx.globals().raw_set(ctx, key, Value::function(forward_fn));
            let chunk = ctx.load(src, Some("seq_tailcall"))?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    lua.execute(&ex).expect("run")
}

#[test]
fn sequence_tailcall_lands_at_original_func_idx() {
    let result = run("local function addone(x) return x + 1 end\n\
                      return forward(addone)");
    assert_eq!(
        result, 42,
        "TailCall result should land where the caller expected"
    );
}

#[test]
fn sequence_tailcall_goes_through_call_metamethod() {
    let result = run(
        "local addone = setmetatable({}, {__call = function(_, x) return x + 1 end})\n\
                      return forward(addone)",
    );
    assert_eq!(result, 42);
}
