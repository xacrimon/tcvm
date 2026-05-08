//! A `Sequence` returning `SequencePoll::Resume` resumes a coroutine,
//! and is then re-polled with the coroutine's eventual return values
//! visible in its window.

use std::pin::Pin;

use tcvm::dmm::{Collect, Trace};
use tcvm::env::{Error, Function, LuaString, NativeContext, NativeFn, Stack, Thread, Value};
use tcvm::lua::Context;
use tcvm::vm::sequence::{BoxSequence, CallbackAction, Execution, Sequence, SequencePoll};
use tcvm::{Executor, LoadError, Lua};

/// Resumes the captured thread once, then on the re-poll reads the
/// thread's return at stack[0] and adds 1.
struct ResumeAndAddOne<'gc> {
    target: Thread<'gc>,
    resumed: bool,
}

unsafe impl<'gc> Collect<'gc> for ResumeAndAddOne<'gc> {
    const NEEDS_TRACE: bool = true;

    fn trace<T: tcvm::dmm::Trace<'gc>>(&self, cc: &mut T) {
        cc.trace(&self.target);
    }
}

impl<'gc> Sequence<'gc> for ResumeAndAddOne<'gc> {
    fn trace_pointers(&self, cc: &mut dyn Trace<'gc>) {
        tcvm::seq_trace_pointers!(self, cc);
    }

    fn poll(
        mut self: Pin<&mut Self>,
        _ctx: Context<'gc>,
        _exec: Execution<'gc, '_>,
        mut stack: Stack<'gc, '_>,
    ) -> Result<SequencePoll<'gc>, Error<'gc>> {
        if !self.resumed {
            self.resumed = true;
            // No args to the resume.
            stack.replace(&[]);
            Ok(SequencePoll::Resume {
                thread: self.target,
                bottom: 0,
            })
        } else {
            let v = stack
                .get(0)
                .get_integer()
                .expect("target should return an integer");
            stack.replace(&[Value::integer(v + 1)]);
            Ok(SequencePoll::Return)
        }
    }
}

/// Native `bumpr(co)` becomes a Sequence that resumes `co` and adds 1
/// to its return.
fn bumpr<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let target = stack.get(0).get_thread().expect("arg #1 is a coroutine");
    stack.replace(&[]);
    let seq = BoxSequence::new(
        nctx.ctx.mutation(),
        ResumeAndAddOne {
            target,
            resumed: false,
        },
    );
    Ok(CallbackAction::Sequence(seq))
}

#[test]
fn sequence_resume_then_post_processes() {
    let mut lua = Lua::new();
    lua.load_all();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let f = Function::new_native(ctx.mutation(), bumpr as NativeFn, Box::new([]));
            let key = Value::string(LuaString::new(ctx, b"bumpr"));
            ctx.globals().raw_set(ctx, key, Value::function(f));
            let chunk = ctx.load(
                "local co = coroutine.create(function() return 41 end)\n\
                 return bumpr(co)",
                Some("seq_resume"),
            )?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    let result: i64 = lua.execute(&ex).expect("run");
    assert_eq!(result, 42, "co returns 41; sequence post-adds 1");
}
