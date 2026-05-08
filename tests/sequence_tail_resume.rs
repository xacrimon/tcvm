//! A `Sequence` returning `SequencePoll::TailResume` is consumed; the
//! resumed thread's eventual return values land directly at the
//! original CALL site.

use std::pin::Pin;

use tcvm::dmm::{Collect, Trace};
use tcvm::env::{Error, Function, LuaString, NativeContext, NativeFn, Stack, Thread, Value};
use tcvm::lua::Context;
use tcvm::vm::sequence::{BoxSequence, CallbackAction, Execution, Sequence, SequencePoll};
use tcvm::{Executor, LoadError, Lua};

struct TailResumeSeq<'gc> {
    target: Thread<'gc>,
}

unsafe impl<'gc> Collect<'gc> for TailResumeSeq<'gc> {
    const NEEDS_TRACE: bool = true;

    fn trace<T: tcvm::dmm::Trace<'gc>>(&self, cc: &mut T) {
        cc.trace(&self.target);
    }
}

impl<'gc> Sequence<'gc> for TailResumeSeq<'gc> {
    fn trace_pointers(&self, cc: &mut dyn Trace<'gc>) {
        tcvm::seq_trace_pointers!(self, cc);
    }

    fn poll(
        self: Pin<&mut Self>,
        _ctx: Context<'gc>,
        _exec: Execution<'gc, '_>,
        mut stack: Stack<'gc, '_>,
    ) -> Result<SequencePoll<'gc>, Error<'gc>> {
        // No args to forward.
        stack.replace(&[]);
        Ok(SequencePoll::TailResume(self.target))
    }
}

/// `forward(co)` tail-resumes `co`; whatever co returns goes straight
/// to forward's caller (no post-processing).
fn forward<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let target = stack.get(0).get_thread().expect("arg #1 is a coroutine");
    stack.replace(&[]);
    let seq = BoxSequence::new(nctx.ctx.mutation(), TailResumeSeq { target });
    Ok(CallbackAction::Sequence(seq))
}

#[test]
fn sequence_tail_resume_propagates_return() {
    let mut lua = Lua::new();
    lua.load_all();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let f = Function::new_native(ctx.mutation(), forward as NativeFn, Box::new([]));
            let key = Value::string(LuaString::new(ctx, b"forward"));
            ctx.globals().raw_set(ctx, key, Value::function(f));
            let chunk = ctx.load(
                "local co = coroutine.create(function() return 99 end)\n\
                 return forward(co)",
                Some("seq_tail_resume"),
            )?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    let result: i64 = lua.execute(&ex).expect("run");
    assert_eq!(result, 99, "co's return should reach the chunk verbatim");
}
