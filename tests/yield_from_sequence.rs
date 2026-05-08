//! Regression test: when a coroutine yields from inside a Sequence,
//! `coroutine.resume(co, x)` must NOT call `land_call_results` on the
//! resumer because the Sequence sits on top and reads values from
//! `stack[seq.bottom..]` directly. The previous unconditional landing
//! corrupted the sequence's window.

use std::pin::Pin;

use tcvm::dmm::{Collect, Trace};
use tcvm::env::{Error, Function, LuaString, NativeContext, NativeFn, Stack, Value};
use tcvm::lua::Context;
use tcvm::vm::sequence::{BoxSequence, CallbackAction, Execution, Sequence, SequencePoll};
use tcvm::{Executor, LoadError, Lua};

/// On poll, yields `42`. On re-poll (after resume with `n`), returns
/// `n + 1`. Exercises a Sequence that survives a yield.
struct YieldThenAddOne {
    yielded: bool,
}

unsafe impl<'gc> Collect<'gc> for YieldThenAddOne {
    const NEEDS_TRACE: bool = false;
}

impl<'gc> Sequence<'gc> for YieldThenAddOne {
    fn trace_pointers(&self, _cc: &mut dyn Trace<'gc>) {}

    fn poll(
        mut self: Pin<&mut Self>,
        _ctx: Context<'gc>,
        _exec: Execution<'gc, '_>,
        mut stack: Stack<'gc, '_>,
    ) -> Result<SequencePoll<'gc>, Error<'gc>> {
        if !self.yielded {
            self.yielded = true;
            stack.replace(&[Value::integer(42)]);
            // Yield with the bottom relative to the sequence's window.
            // bottom = 0 means stack[seq.bottom..] (the 42) is what
            // gets yielded.
            Ok(SequencePoll::Yield { bottom: 0 })
        } else {
            // After resume: stack[0..] holds the resume-args.
            let v = stack
                .get(0)
                .get_integer()
                .expect("resume-arg should be integer");
            stack.replace(&[Value::integer(v + 1)]);
            Ok(SequencePoll::Return)
        }
    }
}

/// Native callback that becomes a Sequence yielding 42 once, then
/// returning resume-arg + 1.
fn yielding_seq<'gc>(
    nctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let seq = BoxSequence::new(nctx.ctx.mutation(), YieldThenAddOne { yielded: false });
    Ok(CallbackAction::Sequence(seq))
}

#[test]
fn coroutine_yields_from_sequence_then_resumes() {
    let mut lua = Lua::new();
    lua.load_all();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let f = Function::new_native(ctx.mutation(), yielding_seq as NativeFn, Box::new([]));
            let key = Value::string(LuaString::new(ctx, b"yseq"));
            ctx.globals().raw_set(ctx, key, Value::function(f));
            // Inside a coroutine, call yseq() which yields 42; the
            // resumer feeds 100 back, expecting 101 as the eventual
            // result. The coroutine returns it, the chunk sums 42 + 101.
            let chunk = ctx.load(
                "local co = coroutine.create(function()\n\
                   return yseq()\n\
                 end)\n\
                 local ok1, y1 = coroutine.resume(co)\n\
                 local ok2, y2 = coroutine.resume(co, 100)\n\
                 return y1 + y2",
                Some("yield_from_seq"),
            )?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    let result: i64 = lua.execute(&ex).expect("run");
    assert_eq!(result, 42 + 101, "y1 = 42 (yielded), y2 = 101 (resume+1)");
}
