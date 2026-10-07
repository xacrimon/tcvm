//! Raising and unwinding: from the raise to the frame that catches,
//! inside dispatch. The `xpcall` message handler and the `__close` calls of
//! the variables an error unwound past run from native frames of their own,
//! whose continuations re-raise once done.

use crate::env::error::{Error, Exit as ExitKind};
use crate::env::function::{Function, NativeClosure, Stack};
use crate::env::thread::{MAX_STACK, TbcEntry, ThreadState, ThreadStatus};
use crate::env::{LuaString, MetamethodBits, Value};
use crate::lua::Context;
use crate::vm::abi::{Exit, Jump, handler, handler_bits};
use crate::vm::frame::{self, HDR, NativeHdr, flag};
use crate::vm::native::{ContIdx, NativeOut, OnOk, Protect, cont, native_catch, ok};
use crate::vm::ops::call::ret_exit;
use crate::vm::ops::control::close_upvalues;

/// Why an opcode faulted. `impl_error` renders the reference message for it
/// and raises it as a Lua error on the current frame.
pub(crate) enum OpError<'gc> {
    Index(Value<'gc>),
    Call(Value<'gc>),
    Arith(Value<'gc>, Value<'gc>),
    Bitwise(Value<'gc>, Value<'gc>),
    Concat(Value<'gc>, Value<'gc>),
    Compare(Value<'gc>, Value<'gc>),
    Len(Value<'gc>),
    DivByZero,
    ModByZero,
    IndexChainLoop,
    NewIndexChainLoop,
    CallChainTooLong,
    /// ERRNNIL: constant index of the global's name.
    GlobalRedefined(u16),
    ForStepZero,
    /// Which `for` control value (`"limit"`, `"step"`, `"initial value"`)
    /// failed to coerce, and the offending value.
    ForNotNumber(&'static str, Value<'gc>),
    NilIndex,
    NanIndex,
    /// A call's register window would cross `ThreadState::stack_limit`.
    StackOverflow,
    /// A named vararg table's `n` isn't an integer in `0..=i32::MAX / 2`.
    VarargN,
    /// TBC: the register holds a value without `__close`.
    NonClosable(u8),
    Internal(&'static str),
    /// Raised by a native or a metamethod, carrying its own value.
    Thrown(Error<'gc>),
}

handler! {
    bind(insn, pc, base, rt, closure, thread, nret, values);

    /// Cold tail of `raise!`/`throw!`: publish the faulting frame, render the
    /// message and unwind to the frame that catches it. A handler so it can
    /// be `become`d: a plain call here would put a frame on every raising
    /// handler's fast path.
    slow fn impl_error {
        // Null from an `enter` staged by a native frame, which is published.
        if !base.is_null() {
            sync!();
        }
        let mut tsr: &mut ThreadState<'gc> = thread!();
        let err = render(rt, tsr, rt.take_fault(), true);
        let j = unwind(rt, &mut tsr, err);
        jump!(j)
    }
}

/// The error for an opcode fault: `positioned` at the running Lua frame, or
/// bare when the running function counts as native (a frameless `pcall`
/// entering its callee).
pub(crate) fn render<'gc>(
    ctx: Context<'gc>,
    ts: &ThreadState<'gc>,
    kind: OpError<'gc>,
    positioned: bool,
) -> Error<'gc> {
    match kind {
        OpError::Thrown(err) => err,
        OpError::StackOverflow => crate::vm::debug::stack_overflow(ctx, ts, positioned),
        kind => {
            let msg = crate::vm::debug::op_error_message(ctx, ts, kind);
            if positioned {
                crate::vm::debug::error_at(ctx, ts, &msg, 0)
            } else {
                Error::new(ctx, Value::string(LuaString::new(ctx, msg.as_bytes())))
            }
        }
    }
}

/// Whether the continuation at `cont` catches exits too (`Protect::Base`):
/// the base-level runner of `coroutine.close`.
fn catches_exit(cont: u8) -> bool {
    cont == cont::CLOSE_ENTRY.0
}

/// Unwind `ts` from the raise of `err` to the frame that catches it. An error
/// that kills a coroutine goes on in its resumer, which becomes `ts`.
pub(crate) fn unwind<'gc>(
    ctx: Context<'gc>,
    ts: &mut &mut ThreadState<'gc>,
    err: Error<'gc>,
) -> Jump<'gc> {
    let mut err = crate::vm::debug::locate(ctx, ts, err);
    let t = &mut **ts;
    if let ExitKind::Process(_) = err.exit_kind() {
        t.uncaught = Some(err);
        t.status = ThreadStatus::Stopped;
        return Jump::Exit(Exit::End);
    }
    let exit = err.exit_kind() != ExitKind::No;
    if !err.is_handled()
        && let Some(handler) = message_handler(t)
    {
        return push_handler(ctx, t, handler, err);
    }
    // `luaD_seterrorobj`: only once the error is being caught (a handler
    // still sees the raw nil).
    if err.value().is_nil() && !exit {
        let msg = LuaString::new(ctx, b"<no error object>");
        err = err.with_value(ctx, Value::string(msg));
    }
    // The popped frames' to-be-closed variables stay listed, detached at
    // the lowest popped base, which ends up `top`. They close once the
    // catcher is reached, after the frames are gone (`luaD_pcall`). An
    // exit never has any: the coroutine closed them before raising it.
    let mut detached = false;
    loop {
        let base = t.top_base;
        if base.is_null() {
            break;
        }
        let bi = t.slot_index(base);
        let rw = unsafe { frame::ret_word(base) };
        let native = rw & flag::NATIVE != 0;
        let marker = rw & !flag::MASK;
        let xpcall = marker == handler_bits(crate::vm::native::ret_xpcall);
        // A native frame's own protection comes first: its continuation
        // decides what the error becomes.
        if native && rw & flag::PROTECTED != 0 {
            let nh = NativeHdr::unpack(unsafe { frame::func_word(base) });
            if !exit || catches_exit(nh.cont) {
                if detached && !exit {
                    let handler = (rw & flag::HANDLER != 0)
                        .then(|| t.stack[bi].get_function())
                        .flatten();
                    return push_close(ctx, t, err, handler);
                }
                return native_catch(ctx, ts, bi, err);
            }
        }
        if (marker == handler_bits(crate::vm::native::ret_pcall) || xpcall) && !exit {
            // A `pcall` that called this frame without a frame of its own
            // catches: land `(false, err)` where the callee was and
            // continue through the CALL's continuation. Variables
            // left to close run first, from a runner above this frame,
            // which the handled error then reaches again.
            if !native {
                close_upvalues(ctx.mutation(), t, bi);
                detach_tbc(t, bi, &mut detached);
            }
            if detached {
                t.set_top_unchecked(bi);
                let handler = xpcall.then(|| {
                    let nv = unsafe { frame::extras(base) };
                    t.stack[bi - HDR - nv - 1].get_function()
                });
                return push_close(ctx, t, err, handler.flatten());
            }
            let cpc = unsafe { frame::caller_pc(base) };
            pop_frame(ctx, t, base, bi, native, &mut detached);
            t.ensure_slots(bi + 2);
            // Popped before the growth, the frame's header was not
            // rebased; the continuation still reads its caller word.
            unsafe { frame::set_caller_base(t.slot_ptr(bi), t.top_base) };
            t.stack[bi] = Value::boolean(false);
            t.stack[bi + 1] = err.value();
            t.set_top_unchecked(bi + 2);
            let call = unsafe { *cpc.sub(1) };
            return Jump::Ret {
                ret: ctx.ret(call.c()),
                nret: 2,
                values: t.slot_ptr(bi),
                base: t.slot_ptr(bi),
            };
        }
        pop_frame(ctx, t, base, bi, native, &mut detached);
    }
    // The host's call closes like a `pcall`; a dead coroutine keeps its
    // variables for `coroutine.close` (`lua_resume` leaves them open).
    if detached && t.main {
        return push_close(ctx, t, err, None);
    }
    if !t.main && err.exit_kind() == ExitKind::Clean {
        // A coroutine closed itself: it returns nothing.
        t.set_top_unchecked(0);
        return match crate::vm::coro::end_into_resumer(ctx, ts, 0) {
            Some(j) => j,
            None => {
                let t = &mut **ts;
                t.discard_above(0);
                t.status = ThreadStatus::Result { bottom: 0 };
                Jump::Exit(Exit::End)
            }
        };
    }
    // A dead coroutine whose resumer waits in dispatch: its native frame
    // catches the error. Anything else is the host's.
    let resumer = t
        .resumer
        .filter(|_| !t.main)
        .map(|r| unsafe { r.state_mut(ctx.mutation()) })
        .filter(|rs| rs.top_is_native());
    let Some(rs) = resumer else {
        t.uncaught = Some(err);
        t.status = ThreadStatus::Stopped;
        return Jump::Exit(Exit::End);
    };
    // Stash the killing error so `coroutine.close` can surface it as
    // `(false, err)`.
    t.status = ThreadStatus::Stopped;
    t.death_error = Some(err.value());
    t.resumer = None;
    rs.status = ThreadStatus::Normal;
    // Already located on the coroutine. An exit ends only the coroutine
    // that closed itself.
    if exit {
        err = Error::new(ctx, err.value());
    }
    let rwin = rs.top_base_index();
    *ts = rs;
    ctx.set_thread(*ts);
    native_catch(ctx, ts, rwin, err)
}

/// Pop the frame at `base`: close its upvalues, detach its to-be-closed
/// variables, drop its task, and make its caller the top frame.
fn pop_frame<'gc>(
    ctx: Context<'gc>,
    t: &mut ThreadState<'gc>,
    base: *mut Value<'gc>,
    bi: usize,
    native: bool,
    detached: &mut bool,
) {
    if native {
        let nh = NativeHdr::unpack(unsafe { frame::func_word(base) });
        if nh.cont == cont::ASYNC.0 {
            t.drop_task();
        }
    } else {
        close_upvalues(ctx.mutation(), t, bi);
        detach_tbc(t, bi, detached);
    }
    // Only the logical top drops: an outer frame's register window can
    // extend past this frame's base, so the vec must not shrink.
    t.set_top_unchecked(bi);
    unsafe {
        t.top_pc = frame::caller_pc(base);
        t.top_base = frame::caller_base(base);
    }
}

/// Detach the to-be-closed variables of the frame at `bi`, which an error is
/// popping: they stay listed, at `bi`, until the catcher closes them.
fn detach_tbc<'gc>(t: &mut ThreadState<'gc>, bi: usize, detached: &mut bool) {
    for entry in t.tbc_list.iter_mut().rev() {
        if entry.pos() < bi {
            break;
        }
        let value = entry.value(&t.stack);
        *entry = TbcEntry::Detached { level: bi, value };
        *detached = true;
    }
}

/// The message handler of the nearest catch point (`L->errfunc`): a plain
/// `pcall` in between shadows an outer `xpcall`.
fn message_handler<'gc>(ts: &ThreadState<'gc>) -> Option<Function<'gc>> {
    frame::frames(ts)
        .find_map(|f| {
            let rw = f.ret;
            let marker = rw & !flag::MASK;
            if marker == handler_bits(crate::vm::native::ret_xpcall) {
                // `xpcall`'s handler sits in the slot below the callee's
                // header and extras.
                let nv = unsafe { frame::extras(f.base) };
                let bi = ts.slot_index(f.base);
                return Some(ts.stack[bi - HDR - nv - 1].get_function());
            }
            if marker == handler_bits(crate::vm::native::ret_pcall) {
                return Some(None);
            }
            if f.is_native() && rw & flag::PROTECTED != 0 {
                return Some(if rw & flag::HANDLER != 0 {
                    let bi = ts.slot_index(f.base);
                    ts.stack[bi].get_function()
                } else {
                    None
                });
            }
            None
        })
        .flatten()
}

/// Push a native frame for `cont` above `live_top`, its window `window`, and
/// enter the call staged at `window[at]` (the callee, then its arguments).
fn push_runner<'gc>(
    ctx: Context<'gc>,
    ts: &mut ThreadState<'gc>,
    window: &[Value<'gc>],
    at: usize,
    protect: Protect,
    cont: ContIdx,
) -> Jump<'gc> {
    let f = ctx.unwind_fn();
    let nc = f.as_native().expect("the unwind native");
    // Above every live register: `top` alone can sit inside the innermost
    // window when the catcher's callee failed at once.
    let hdr = ts.live_top();
    let win = hdr + HDR;
    ts.ensure_slots(win + window.len() + HDR);
    ts.stack[win..win + window.len()].copy_from_slice(window);
    ts.set_top_unchecked(win + window.len());
    // Never returns: its continuation always raises.
    let (below, below_pc) = (ts.top_base, ts.top_pc);
    let win_ptr = ts.slot_ptr(win);
    unsafe {
        frame::write_hdr(
            ts.slot_ptr(hdr),
            NativeHdr {
                at,
                cont: cont.0,
                ok: ok::CONT,
            }
            .pack(nc),
            handler_bits(ret_exit) | flag::NATIVE | protect.flags(),
            below,
            below_pc,
        );
    }
    ts.top_base = win_ptr;
    // The call it waits for, in call layout.
    let callee = win + at;
    let nargs = ts.top - (callee + 1);
    ts.stack.copy_within(callee + 1..ts.top, callee + HDR);
    ts.top += 3;
    unsafe {
        let c = ts.slot_ptr(callee).cast::<u64>();
        c.add(1).write(handler_bits(crate::vm::native::ret_native));
        c.add(2).write(win_ptr as usize as u64);
        c.add(3).write(0);
    }
    Jump::Enter {
        hdr: ts.slot_ptr(callee),
        nargs,
        base: std::ptr::null_mut(),
    }
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
) -> Jump<'gc> {
    let limit = std::mem::replace(&mut ts.stack_limit, MAX_STACK);
    let h = Value::function(handler);
    let window = [
        h,
        Value::small(0),
        Value::small(limit as i32),
        h,
        err.value(),
    ];
    push_runner(ctx, ts, &window, 3, Protect::Errors, cont::HANDLER)
}

/// The message handler returned: its first result, marked handled, goes on
/// to the catcher without consulting the handler again. An error inside the
/// handler calls the handler again with it (manual §2.3), on top of the
/// still-intact failing frames, until the depth limits cut the loop. The
/// handler may yield: the reference forbids that only because it runs the
/// handler on the C stack, and we keep every call resumable, as LuaJIT does.
pub(crate) fn handler_cont<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
    status: Result<(), Error<'gc>>,
) -> NativeOut {
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
            return NativeOut::call_then(3, cont::HANDLER, Protect::Errors, OnOk::Cont);
        }
    };
    stack.thread_mut().stack_limit = limit;
    NativeOut::error(err)
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
) -> Jump<'gc> {
    let level = ts.top;
    let protect = close_protect(handler.is_some());
    let handler = handler.map_or(Value::nil(), Value::function);
    let entry = ts
        .tbc_list
        .pop_if(|e| e.pos() >= level)
        .expect("no detached variable to close");
    let v = entry.value(&ts.stack);
    let mm = ctx.mm_of(v, MetamethodBits::CLOSE);
    let errv = err.value();
    let window = [handler, Value::small(level as i32), errv, mm, v, errv];
    push_runner(ctx, ts, &window, 3, protect, cont::CLOSE)
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
pub(crate) fn close_cont<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
    status: Result<(), Error<'gc>>,
) -> NativeOut {
    let errv = match status {
        Ok(()) => stack.get(2),
        Err(e) => e.value(),
    };
    if !close_next(ctx, &mut stack, errv) {
        return NativeOut::error(Error::new(ctx, errv).mark_handled());
    }
    NativeOut::call_then(
        3,
        cont::CLOSE,
        close_protect(!stack.get(0).is_nil()),
        OnOk::Cont,
    )
}

/// The function of the runner frames: never called.
pub(crate) fn unwind_native<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    _stack: Stack<'gc, '_>,
) -> Result<(), Error<'gc>> {
    Err(Error::from_str(ctx, "internal function"))
}
