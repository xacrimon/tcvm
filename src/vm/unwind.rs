//! Error unwinding (`luaD_throw` and `luaD_pcall`'s recovery): from the raise
//! to the nearest frame that catches, inside dispatch. The `xpcall` message
//! handler and the `__close` calls of the variables an error unwound past run
//! from native frames of their own, whose continuations re-raise once done.

use crate::env::error::{Error, Exit};
use crate::env::function::{Function, NativeClosure, Stack};
use crate::env::thread::{
    ExecKind, FrameRef, LuaFrame, MAX_STACK, TbcEntry, ThreadState, ThreadStatus, frame_flags,
};
use crate::env::{LuaString, MetamethodBits, Value};
use crate::lua::Context;
use crate::vm::interp::{close_upvalues, native_frame, protected_frame, ret_exit};
use crate::vm::native::{CallbackAction, NativeCont, OnOk, Protect};

/// Where [`unwind`] left an error.
pub(crate) enum Unwound<'gc> {
    /// The native frame now on top catches it in its continuation.
    Catch(Error<'gc>),
    /// The native frame now on top calls the message handler or a `__close`
    /// waiting at this slot.
    Call(usize),
    /// The executor takes over: nothing on the thread catches the error, as
    /// the `ExecKind::Error` left on top says, or it is a process exit.
    Exit,
}

/// Unwind `thread` from the raise of the located `err` to the frame that
/// catches it. With `switch`, an error that kills a coroutine its resumer
/// waits on in dispatch goes on in the resumer, which becomes `thread`.
pub(crate) fn unwind<'gc>(
    ctx: Context<'gc>,
    thread: &mut &mut ThreadState<'gc>,
    mut err: Error<'gc>,
    switch: bool,
) -> Unwound<'gc> {
    loop {
        let ts = &mut **thread;
        if let Exit::Process(_) = err.exit_kind() {
            ts.push_exec(ExecKind::Error(err));
            return Unwound::Exit;
        }
        let exit = err.exit_kind() != Exit::No;
        if !err.is_handled()
            && let Some(handler) = message_handler(ts)
        {
            return Unwound::Call(push_handler(ctx, ts, handler, err));
        }
        // `luaD_seterrorobj`: only once the error is being caught (a handler
        // still sees the raw nil).
        if err.value().is_nil() && !exit {
            let msg = LuaString::new(ctx, b"<no error object>");
            err = err.with_value(ctx, Value::string(msg));
        }
        // The popped frames' to-be-closed variables stay listed, detached at
        // the lowest popped base, which ends up `ts.top`. They close once the
        // catcher is reached, after the frames are gone (`luaD_pcall`). An
        // exit never has any: the coroutine closed them before raising it.
        let mut detached = false;
        loop {
            if let Some(lf) = ts.top_lua() {
                if lf.is_native() {
                    let handler = if exit && lf.flags & frame_flags::BASE == 0 {
                        None
                    } else {
                        native_catch(ts, lf)
                    };
                    let at = lf.base() + lf.num_extras as usize;
                    let Some(handler) = handler else {
                        let task = lf.pc == crate::vm::async_native::async_cont as *const _;
                        ts.pop_lua();
                        if task {
                            ts.drop_task();
                        }
                        continue;
                    };
                    if detached && !exit {
                        return Unwound::Call(push_close(ctx, ts, err, handler));
                    }
                    // Nothing of the failed call is left above its slot.
                    ts.set_top_unchecked(at);
                    return Unwound::Catch(err);
                }
                let (base, func_slot, protect) = (lf.base(), lf.func_slot(), lf.elided_protect());
                ts.pop_lua();
                close_upvalues(ctx.mutation(), ts, base);
                for entry in ts.tbc_list.iter_mut().rev() {
                    if entry.pos() < base {
                        break;
                    }
                    let value = entry.value(&ts.stack);
                    *entry = TbcEntry::Detached { level: base, value };
                    detached = true;
                }
                // Only the logical top drops: an outer frame's register window
                // can extend past this frame's base, so the vec must not shrink.
                ts.set_top_unchecked(base);
                if let Some(k) = protect {
                    // The `pcall` that called it without a frame catches: put
                    // its frame back, for the native catch above.
                    let nf = protected_frame(ts, func_slot - k, k);
                    ts.push_lua(nf);
                }
                continue;
            }
            match ts.top_exec() {
                Some(ExecKind::WaitThread { .. }) => {
                    ts.pop_exec();
                }
                Some(ExecKind::Start(_) | ExecKind::Error(_)) => {
                    unreachable!("Start / Error frame mid-unwind violates the executor invariant");
                }
                None => break,
            }
        }
        // The host's call closes like a `pcall`; a dead coroutine keeps its
        // variables for `coroutine.close` (`lua_resume` leaves them open).
        if detached && ts.main {
            return Unwound::Call(push_close(ctx, ts, err, None));
        }
        // A dead coroutine whose resumer waits in dispatch: its native frame
        // catches the error. Anything else is the executor's.
        let resumer = ts
            .resumer
            .filter(|_| switch && !ts.main && err.exit_kind() != Exit::Clean)
            .map(|r| unsafe { r.state_mut(ctx.mutation()) })
            .filter(|rs| rs.top_lua().is_some_and(LuaFrame::is_native));
        let Some(rs) = resumer else {
            ts.push_exec(ExecKind::Error(err));
            return Unwound::Exit;
        };
        // Stash the killing error so `coroutine.close` can surface it as
        // `(false, err)`.
        ts.status = ThreadStatus::Stopped;
        ts.death_error = Some(err.value());
        ts.resumer = None;
        rs.status = ThreadStatus::Normal;
        // Already located on the coroutine. An exit ends only the coroutine
        // that closed itself.
        if exit {
            err = Error::new(ctx, err.value());
        }
        *thread = rs;
    }
}

/// The message handler of the nearest catch point (`L->errfunc`): a plain
/// `pcall` in between shadows an outer `xpcall`.
fn message_handler<'gc>(ts: &ThreadState<'gc>) -> Option<Function<'gc>> {
    ts.frames_rev()
        .find_map(|f| match f {
            FrameRef::Native(nf) => native_catch(ts, nf),
            FrameRef::Elided(lf) => Some(match lf.elided_protect() {
                // `xpcall`'s handler sits in the slot below the callee's.
                Some(2) => ts.stack[lf.func_slot() - 1].get_function(),
                _ => None,
            }),
            _ => None,
        })
        .flatten()
}

/// Whether the native frame `nf` catches errors: `Some` with its message
/// handler, if any.
fn native_catch<'gc>(ts: &ThreadState<'gc>, nf: &LuaFrame<'gc>) -> Option<Option<Function<'gc>>> {
    if nf.flags & frame_flags::PROTECTED == 0 {
        return None;
    }
    Some(if nf.flags & frame_flags::HANDLER != 0 {
        ts.stack[nf.base()].get_function()
    } else {
        None
    })
}

/// Push a native frame for `cont` at `slot`, its function slot, with the
/// `window` above, and the call it waits for at `window[at]`. Returns the
/// call's slot.
fn push_runner<'gc>(
    ctx: Context<'gc>,
    ts: &mut ThreadState<'gc>,
    slot: usize,
    window: &[Value<'gc>],
    at: usize,
    protect: Protect,
    cont: NativeCont,
) -> usize {
    let f = ctx.unwind_fn();
    let base = slot + 1;
    ts.ensure_slots(base + window.len());
    ts.stack[slot] = Value::function(f);
    ts.stack[base..base + window.len()].copy_from_slice(window);
    ts.set_top_unchecked(base + window.len());
    // Never returns: its continuation always raises.
    let nf = native_frame(f, base, at as u32, protect, OnOk::Cont, cont, ret_exit);
    ts.push_lua(nf);
    base + at
}

/// The nested handler call at this depth gets "stack overflow in message
/// handler" instead of the error (PUC: `LUAI_MAXCCALLS`, "C stack overflow").
const MAX_HANDLER_DEPTH: i64 = 200;

/// The depth at which the loop gives up (`luaE_checkcstack`).
const HANDLER_GIVE_UP_DEPTH: i64 = MAX_HANDLER_DEPTH / 10 * 11;

/// Call an `xpcall` message handler with `err` above the failing frames
/// (`luaG_errormsg`), in the headroom past the stack limit, so that it can
/// run even after a stack overflow. Its window: the handler, the depth of
/// nested handler calls, the stack limit to restore, then the call.
fn push_handler<'gc>(
    ctx: Context<'gc>,
    ts: &mut ThreadState<'gc>,
    handler: Function<'gc>,
    err: Error<'gc>,
) -> usize {
    // Above every live register: `top` alone can sit inside the innermost
    // window when the catcher's callee failed at once.
    let slot = ts.live_top();
    let limit = std::mem::replace(&mut ts.stack_limit, MAX_STACK);
    let h = Value::function(handler);
    let window = [
        h,
        Value::small(0),
        Value::small(limit as i32),
        h,
        err.value(),
    ];
    push_runner(ctx, ts, slot, &window, 3, Protect::Errors, handler_cont)
}

/// The message handler returned: its first result, marked handled, goes on
/// to the catcher without consulting the handler again. An error inside the
/// handler calls the handler again with it (manual §2.3), on top of the
/// still-intact failing frames, until the depth limits cut the loop. The
/// handler may yield: the reference forbids that only because it runs the
/// handler on the C stack, and we keep every call resumable, as LuaJIT does.
fn handler_cont<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
    status: Result<(), Error<'gc>>,
) -> Result<CallbackAction, Error<'gc>> {
    let depth = stack.get(1).get_integer().unwrap_or(0);
    let limit = stack.get(2).get_integer().unwrap_or(0) as usize;
    let err = match status {
        Ok(()) => Error::new(ctx, stack.get(3)).mark_handled(),
        // Already handled, it is "error in error handling" from a stack
        // overflow in the handler, and goes to the catcher as is.
        Err(e) if e.is_handled() => e,
        Err(_) if depth == HANDLER_GIVE_UP_DEPTH => crate::vm::debug::error_in_error_handling(ctx),
        Err(e) => {
            let depth = depth + 1;
            let value = if depth == MAX_HANDLER_DEPTH {
                Value::string(LuaString::new(ctx, b"stack overflow in message handler"))
            } else {
                e.value()
            };
            let handler = stack.get(0);
            stack.truncate(2);
            stack.extend([Value::small(limit as i32), handler, value]);
            stack.as_mut_slice()[1] = Value::small(depth as i32);
            return Ok(CallbackAction::CallThen {
                at: 3,
                protect: Protect::Errors,
                ok: OnOk::Cont,
                cont: handler_cont,
            });
        }
    };
    stack.thread_mut().stack_limit = limit;
    Err(err)
}

/// Close the variables the walk detached at `ts.top`, with `err` above the
/// catcher, whose message handler (if any) also sees errors the `__close`
/// calls raise (`luaD_closeprotected`). Its window: that handler, the level
/// detached variables close from, the error, then the call.
fn push_close<'gc>(
    ctx: Context<'gc>,
    ts: &mut ThreadState<'gc>,
    err: Error<'gc>,
    handler: Option<Function<'gc>>,
) -> usize {
    let slot = ts.top;
    let protect = close_protect(handler.is_some());
    let handler = handler.map_or(Value::nil(), Value::function);
    let window = [handler, Value::small(slot as i32), err.value()];
    let at = push_runner(ctx, ts, slot, &window, 3, protect, close_cont);
    let more = close_next(ctx, &mut Stack::new(ts, slot + 1), err.value());
    debug_assert!(more, "no detached variable to close");
    at
}

fn close_protect(handler: bool) -> Protect {
    if handler {
        Protect::Handler
    } else {
        Protect::Errors
    }
}

/// Put the call of the next detached variable's `__close` at window slot 3,
/// with `errv`; `false` with all closed.
fn close_next<'gc>(ctx: Context<'gc>, stack: &mut Stack<'gc, '_>, errv: Value<'gc>) -> bool {
    let level = stack.get(1).get_integer().unwrap_or(0) as usize;
    let ts = stack.thread_mut();
    let Some(entry) = ts.tbc_list.pop_if(|e| e.pos() >= level) else {
        return false;
    };
    let v = entry.value(&ts.stack);
    let mm = ctx.mm_of(v, MetamethodBits::CLOSE);
    stack.truncate(2);
    stack.extend([errv, mm, v, errv]);
    true
}

/// A `__close` call returned; an error in it replaces the error for the rest
/// (`luaF_close`). With all closed, the error goes on to the catcher, whose
/// message handler has already seen it.
fn close_cont<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
    status: Result<(), Error<'gc>>,
) -> Result<CallbackAction, Error<'gc>> {
    let errv = match status {
        Ok(()) => stack.get(2),
        Err(e) => e.value(),
    };
    if !close_next(ctx, &mut stack, errv) {
        return Err(Error::new(ctx, errv).mark_handled());
    }
    Ok(CallbackAction::CallThen {
        at: 3,
        protect: close_protect(!stack.get(0).is_nil()),
        ok: OnOk::Cont,
        cont: close_cont,
    })
}

/// The function of the frames above: never called.
pub(crate) fn unwind_native<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    _stack: Stack<'gc, '_>,
) -> Result<(), Error<'gc>> {
    Err(Error::from_str(ctx, "internal function"))
}
