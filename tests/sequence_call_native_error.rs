//! When a `Sequence::Call` invokes a native that errors, the error must
//! propagate to the nearest catcher (e.g. the `PCallSequence` under
//! `coroutine.resume`) instead of short-circuiting to the host.

use std::pin::Pin;

use tcvm::dmm::{Collect, Trace};
use tcvm::env::{Error, Function, LuaString, NativeContext, NativeFn, Stack, Value};
use tcvm::lua::Context;
use tcvm::vm::sequence::{BoxSequence, CallbackAction, Execution, Sequence, SequencePoll};
use tcvm::{Executor, LoadError, Lua};

fn boomer<'gc>(
    nctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    Err(Error::from_str(nctx.ctx, "boom"))
}

/// Sequence: on first poll, calls `boomer` (with no `then` follow-up). The
/// error from boomer should bubble up to the `PCallSequence` installed by
/// `coroutine.resume`.
struct CallBoomerSeq<'gc> {
    boomer: Function<'gc>,
    called: bool,
}

unsafe impl<'gc> Collect<'gc> for CallBoomerSeq<'gc> {
    const NEEDS_TRACE: bool = true;
    fn trace<T: tcvm::dmm::Trace<'gc>>(&self, cc: &mut T) {
        cc.trace(&self.boomer);
    }
}

impl<'gc> Sequence<'gc> for CallBoomerSeq<'gc> {
    fn trace_pointers(&self, cc: &mut dyn Trace<'gc>) {
        tcvm::seq_trace_pointers!(self, cc);
    }
    fn poll(
        mut self: Pin<&mut Self>,
        _ctx: Context<'gc>,
        _exec: Execution<'gc, '_>,
        mut stack: Stack<'gc, '_>,
    ) -> Result<SequencePoll<'gc>, Error<'gc>> {
        if !self.called {
            self.called = true;
            stack.replace(&[]);
            Ok(SequencePoll::Call {
                function: self.boomer,
                bottom: 0,
            })
        } else {
            Ok(SequencePoll::Return)
        }
    }
}

fn factory<'gc>(
    nctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let boomer_fn = Function::new_native(nctx.ctx.mutation(), boomer as NativeFn, Box::new([]));
    let seq = BoxSequence::new(
        nctx.ctx.mutation(),
        CallBoomerSeq {
            boomer: boomer_fn,
            called: false,
        },
    );
    Ok(CallbackAction::Sequence(seq))
}

#[test]
fn native_error_in_sequence_call_caught_by_pcallseq() {
    let mut lua = Lua::new();
    lua.load_all();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let f = Function::new_native(ctx.mutation(), factory as NativeFn, Box::new([]));
            let key = Value::string(LuaString::new(ctx, b"factory"));
            ctx.globals().raw_set(ctx, key, Value::function(f));
            let chunk = ctx.load(
                "local co = coroutine.create(function() factory() end)\n\
                 local ok, msg = coroutine.resume(co)\n\
                 if (not ok) and msg == 'boom' then return 1 else return 0 end",
                Some("seq_call_native_err"),
            )?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    let result: i64 = lua.execute(&ex).expect("run");
    assert_eq!(result, 1);
}
