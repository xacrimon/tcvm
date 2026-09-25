//! Calling `__close` on to-be-closed variables (`luaF_close`).

use std::pin::Pin;

use crate::dmm::{Collect, Trace};
use crate::env::error::Error;
use crate::env::function::Stack;
use crate::env::thread::{ExecKind, TbcEntry, ThreadState, ThreadStatus};
use crate::env::{Function, NativeClosure, Value};
use crate::lua::Context;
use crate::vm::sequence::{
    BoxSequence, CallbackAction, Catch, Execution, Sequence, SequencePoll, seq_trace_pointers,
};

/// Closes the running frame's variables at or above `level` in turn, taking
/// each off `tbc_list` before its call, so a failing `__close` leaves the rest
/// registered for the unwinder to close with the error. `redo` steps the frame
/// back onto its RETURN, which then finds nothing left to close.
pub(crate) struct CloseSequence {
    pub(crate) level: usize,
    pub(crate) redo: bool,
}

unsafe impl<'gc> Collect<'gc> for CloseSequence {
    const NEEDS_TRACE: bool = false;
}

impl<'gc> Sequence<'gc> for CloseSequence {
    fn trace_pointers(&self, _cc: &mut dyn Trace<'gc>) {}

    fn poll(
        self: Pin<&mut Self>,
        ctx: Context<'gc>,
        _exec: Execution<'gc>,
        mut stack: Stack<'gc, '_>,
    ) -> Result<SequencePoll<'gc>, Error<'gc>> {
        stack.clear();
        let ts = stack.thread_mut();
        let Some(entry) = ts.tbc_list.pop_if(|e| e.pos() >= self.level) else {
            if self.redo {
                let frame = ts.top_lua_mut().expect("close sequence without its frame");
                frame.pc = unsafe { frame.pc.sub(1) };
            }
            return Ok(SequencePoll::Return);
        };
        let v = entry.value(&ts.stack);
        let tm = ctx.metamethod_of(v, ctx.symbols().close);
        if tm.get_function().is_none() && ctx.metamethod_of(tm, ctx.symbols().mm_call).is_nil() {
            let tn = crate::vm::debug::object_type_name(ctx, tm);
            let msg = format!("attempt to call a {tn} value (metamethod 'close')");
            return Err(Error::from_str(ctx, &msg));
        }
        stack.push(v);
        Ok(SequencePoll::Call {
            function: tm,
            bottom: 0,
        })
    }
}

/// Closes the variables an error unwound past, detached at `level`, once the
/// unwinder has reached the catcher (`luaD_closeprotected`). An error in a
/// `__close` replaces `err` for the rest. Ends by re-raising `err` to the
/// catcher, whose message handler has already seen it.
#[derive(Collect)]
#[collect(internal, no_drop)]
pub(crate) struct ErrorCloseSequence<'gc> {
    #[collect(require_static)]
    pub(crate) level: usize,
    pub(crate) err: Error<'gc>,
    /// The catcher's message handler, for errors raised by `__close` itself.
    pub(crate) handler: Option<Function<'gc>>,
}

impl<'gc> ErrorCloseSequence<'gc> {
    fn next(
        &mut self,
        ctx: Context<'gc>,
        mut stack: Stack<'gc, '_>,
    ) -> Result<SequencePoll<'gc>, Error<'gc>> {
        stack.clear();
        let ts = stack.thread_mut();
        let Some(entry) = ts.tbc_list.pop_if(|e| e.pos() >= self.level) else {
            return Err(self.err.mark_handled());
        };
        let v = entry.value(&ts.stack);
        stack.extend([v, self.err.value()]);
        Ok(SequencePoll::Call {
            function: ctx.metamethod_of(v, ctx.symbols().close),
            bottom: 0,
        })
    }
}

impl<'gc> Sequence<'gc> for ErrorCloseSequence<'gc> {
    fn trace_pointers(&self, cc: &mut dyn Trace<'gc>) {
        seq_trace_pointers!(self, cc);
    }

    fn poll(
        self: Pin<&mut Self>,
        ctx: Context<'gc>,
        _exec: Execution<'gc>,
        stack: Stack<'gc, '_>,
    ) -> Result<SequencePoll<'gc>, Error<'gc>> {
        self.get_mut().next(ctx, stack)
    }

    fn error(
        self: Pin<&mut Self>,
        ctx: Context<'gc>,
        _exec: Execution<'gc>,
        err: Error<'gc>,
        stack: Stack<'gc, '_>,
    ) -> Result<SequencePoll<'gc>, Error<'gc>> {
        let this = self.get_mut();
        this.err = err;
        this.next(ctx, stack)
    }

    fn catch(&self) -> Catch<'gc> {
        Catch::Here(self.handler)
    }
}

/// Closes a coroutine's variables on its own thread for `coroutine.close` and
/// `coroutine.wrap` (`luaE_resetthread`), which cannot yield meanwhile. An
/// error in a `__close` replaces `err` for the rest. Returns `true`, or
/// `false` and the last error.
#[derive(Collect)]
#[collect(internal, no_drop)]
pub(crate) struct ThreadCloseSequence<'gc> {
    pub(crate) err: Option<Error<'gc>>,
}

impl<'gc> ThreadCloseSequence<'gc> {
    fn next(&mut self, ctx: Context<'gc>, mut stack: Stack<'gc, '_>) -> SequencePoll<'gc> {
        stack.clear();
        let ts = stack.thread_mut();
        let Some(entry) = ts.tbc_list.pop() else {
            ts.no_yield = false;
            match self.err {
                Some(err) => stack.extend([Value::boolean(false), err.value()]),
                None => stack.push(Value::boolean(true)),
            }
            return SequencePoll::Return;
        };
        ts.no_yield = true;
        let v = entry.value(&ts.stack);
        stack.push(v);
        if let Some(err) = self.err {
            stack.push(err.value());
        }
        SequencePoll::Call {
            function: ctx.metamethod_of(v, ctx.symbols().close),
            bottom: 0,
        }
    }
}

impl<'gc> Sequence<'gc> for ThreadCloseSequence<'gc> {
    fn trace_pointers(&self, cc: &mut dyn Trace<'gc>) {
        seq_trace_pointers!(self, cc);
    }

    fn poll(
        self: Pin<&mut Self>,
        ctx: Context<'gc>,
        _exec: Execution<'gc>,
        stack: Stack<'gc, '_>,
    ) -> Result<SequencePoll<'gc>, Error<'gc>> {
        Ok(self.get_mut().next(ctx, stack))
    }

    fn error(
        self: Pin<&mut Self>,
        ctx: Context<'gc>,
        _exec: Execution<'gc>,
        err: Error<'gc>,
        stack: Stack<'gc, '_>,
    ) -> Result<SequencePoll<'gc>, Error<'gc>> {
        let this = self.get_mut();
        this.err = Some(err);
        Ok(this.next(ctx, stack))
    }

    fn catch(&self) -> Catch<'gc> {
        Catch::Here(None)
    }
}

/// Entry of a coroutine seeded by [`seed_thread_close`].
fn thread_close_entry<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let err = stack
        .thread_mut()
        .death_error
        .take()
        .map(|v| Error::new(ctx, v));
    let seq = ThreadCloseSequence { err };
    Ok(CallbackAction::sequence(BoxSequence::new(
        ctx.mutation(),
        seq,
    )))
}

/// Reset the suspended or dead `ts`, leaving it to close its open variables
/// when next resumed; `false` if it had none, leaving it merely reset. Its
/// death error, if any, goes to the first `__close` and is kept on reset.
pub(crate) fn seed_thread_close<'gc>(ctx: Context<'gc>, ts: &mut ThreadState<'gc>) -> bool {
    crate::vm::interp::close_upvalues(ctx.mutation(), ts, 0);
    let tbc_list: Vec<_> = ts
        .tbc_list
        .iter()
        .map(|e| TbcEntry::Detached {
            level: 0,
            value: e.value(&ts.stack),
        })
        .collect();
    let death_error = ts.death_error;
    ts.reset();
    ts.death_error = death_error;
    ts.status = ThreadStatus::Stopped;
    if tbc_list.is_empty() {
        return false;
    }
    ts.tbc_list = tbc_list;
    let entry = Function::new_native(ctx.mutation(), thread_close_entry, Box::new([]));
    ts.push_exec(ExecKind::Start(Value::function(entry)));
    ts.status = ThreadStatus::Suspended;
    true
}
