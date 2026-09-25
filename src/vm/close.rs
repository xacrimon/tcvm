//! Calling `__close` on to-be-closed variables (`luaF_close`).

use std::pin::Pin;

use crate::dmm::{Collect, Trace};
use crate::env::Function;
use crate::env::error::Error;
use crate::env::function::Stack;
use crate::lua::Context;
use crate::vm::sequence::{Catch, Execution, Sequence, SequencePoll, seq_trace_pointers};

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
