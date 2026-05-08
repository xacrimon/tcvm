//! `Sequence::poll` returning `Pending` must surface as
//! `StepResult::Pending` from a single `Executor::step`, not deadlock the
//! driver loop. Driving the executor `n` times yields `n` Pendings before
//! the sequence advances to Return.

use std::pin::Pin;

use tcvm::dmm::{Collect, Trace};
use tcvm::env::{Error, Function, LuaString, NativeContext, NativeFn, Stack, Value};
use tcvm::lua::Context;
use tcvm::vm::sequence::{BoxSequence, CallbackAction, Execution, Sequence, SequencePoll};
use tcvm::{Executor, LoadError, Lua, RuntimeError, StepResult};

struct PendNTimes {
    remaining: u32,
}

unsafe impl<'gc> Collect<'gc> for PendNTimes {
    const NEEDS_TRACE: bool = false;
}

impl<'gc> Sequence<'gc> for PendNTimes {
    fn trace_pointers(&self, _cc: &mut dyn Trace<'gc>) {}
    fn poll(
        mut self: Pin<&mut Self>,
        _ctx: Context<'gc>,
        _exec: Execution<'gc, '_>,
        mut stack: Stack<'gc, '_>,
    ) -> Result<SequencePoll<'gc>, Error<'gc>> {
        if self.remaining > 0 {
            self.remaining -= 1;
            Ok(SequencePoll::Pending)
        } else {
            stack.replace(&[Value::integer(7)]);
            Ok(SequencePoll::Return)
        }
    }
}

fn factory<'gc>(
    nctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let seq = BoxSequence::new(nctx.ctx.mutation(), PendNTimes { remaining: 3 });
    Ok(CallbackAction::Sequence(seq))
}

#[test]
fn pending_surfaces_then_completes() {
    let mut lua = Lua::new();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let f = Function::new_native(ctx.mutation(), factory as NativeFn, Box::new([]));
            let key = Value::string(LuaString::new(ctx, b"factory"));
            ctx.globals().raw_set(ctx, key, Value::function(f));
            let chunk = ctx.load("return factory()", Some("pending"))?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");

    let mut pendings = 0;
    loop {
        let outcome = lua
            .try_enter(|ctx| -> Result<_, RuntimeError> {
                let executor = ctx.fetch(&ex);
                Ok(match executor.step(ctx)? {
                    StepResult::Done => 0,
                    StepResult::Yielded(_) => panic!("unexpected Yielded"),
                    StepResult::Pending => 1,
                })
            })
            .expect("step");
        if outcome == 0 {
            break;
        }
        pendings += 1;
        assert!(pendings < 100, "guard against runaway loop");
    }
    assert_eq!(pendings, 3, "PendNTimes(3) should produce 3 Pendings");
    let result: i64 = lua
        .try_enter(|ctx| ctx.fetch(&ex).take_result::<i64>(ctx))
        .expect("take_result");
    assert_eq!(result, 7);
}
