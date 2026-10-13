//! Coroutine switches. Every thread runs on its own stack (`vm::stack`), so
//! a resume, a yield and a coroutine's end switch stacks: the side that
//! switches away waits inside the switch, and the other copies the values
//! it hands over to where it waits. Here are those routines, the `resume`,
//! `wrap` and `yield` fast entries, and the coroutine body's bottom
//! continuation.
//!
//! Code that switches works on raw `*mut ThreadState`s: the other side
//! writes both states while it waits, so no reference to one may be live
//! across a switch.

use crate::env::Thread;
use crate::env::error::Error;
use crate::env::function::NativeClosure;
use crate::env::thread::{ThreadState, ThreadStatus};
use crate::env::value::Value;
use crate::instruction::{Instruction, Op};
use crate::lua::Context;
use crate::vm::abi::{Slot, handler, handler_bits};
use crate::vm::frame::{self, HDR, NativeHdr, flag};
use crate::vm::native::{cont, native_call, native_catch, native_continue, ok};
use crate::vm::stack::{Park, park, switch_to};

/// `LUAI_MAXCCALLS`: threads resumed inside one another.
pub(crate) const MAX_RESUME_DEPTH: u16 = 200;

/// How a resume came back: with values at the resumer's wait slot, `top`
/// ending them, or with the error that killed the coroutine.
pub(crate) enum Back<'gc> {
    Values,
    Died(Error<'gc>),
}

/// Hand `co` the `n` values at `a.stack[args..]` and make it the resumed
/// coroutine of `a`, the running thread, whose values land at
/// `a.stack[wait..]`. Returns `co`'s state, to switch to.
///
/// # Safety
/// `a` is the running thread and no reference to its state is live; the
/// caller checked that `co` may be resumed with that many values.
unsafe fn begin_resume<'gc>(
    ctx: Context<'gc>,
    a: *mut ThreadState<'gc>,
    co: Thread<'gc>,
    args: usize,
    n: usize,
    wait: usize,
) -> *mut ThreadState<'gc> {
    // SAFETY: `co` is suspended, so nothing else uses its state.
    let b: *mut ThreadState<'gc> = unsafe { co.state_mut(ctx.mutation()) };
    unsafe {
        let (at, bt) = (&mut *a, &mut *b);
        let src = at.stack.as_ptr().add(args);
        if bt.started {
            let yb = bt
                .yield_bottom
                .take()
                .expect("resumed a coroutine that didn't yield");
            bt.copy_in(yb, src, n);
        } else {
            bt.ensure_stack(ctx.stack_pool());
            bt.copy_in(HDR, src, n);
        }
        link_resume(at, bt, wait);
    }
    b
}

/// `b`, which has its values, is the resumed coroutine of `a`, whose values
/// land at `a.stack[wait..]`.
#[inline(always)]
fn link_resume<'gc>(a: &mut ThreadState<'gc>, b: &mut ThreadState<'gc>, wait: usize) {
    a.status = ThreadStatus::Normal;
    a.resume_wait = wait;
    b.status = ThreadStatus::Normal;
    b.resumer = Some(a.handle());
    b.resume_depth = a.resume_depth + 1;
}

/// `b`, which handed its resumer `a` the values from `yb`, waits there.
#[inline(always)]
fn link_yield<'gc>(b: &mut ThreadState<'gc>, a: &mut ThreadState<'gc>, yb: usize) {
    a.status = ThreadStatus::Normal;
    b.resumer = None;
    b.top = yb;
}

/// Resume `co` from the running thread `a` (see [`begin_resume`]) and wait
/// for what it yields or returns, which `a.top` then ends.
///
/// # Safety
/// As [`begin_resume`].
pub(crate) unsafe fn resume<'gc>(
    ctx: Context<'gc>,
    a: *mut ThreadState<'gc>,
    co: Thread<'gc>,
    args: usize,
    n: usize,
    wait: usize,
) -> Back<'gc> {
    unsafe {
        let b = begin_resume(ctx, a, co, args, n, wait);
        switch_to(ctx, &raw mut (*a).sw, b, 0);
    }
    let at = unsafe { &mut *a };
    match at.transfer.take() {
        Some(err) => Back::Died(err),
        None => Back::Values,
    }
}

/// Hand the values at `b.stack[yb..top]` of the running coroutine `b` to its
/// resumer, which it then waits for at `yb`. Returns the resumer's state,
/// none for a main thread, which yields to the host.
///
/// # Safety
/// As [`begin_resume`], for `b`; the caller checked that `b` may yield.
unsafe fn begin_yield<'gc>(
    ctx: Context<'gc>,
    b: *mut ThreadState<'gc>,
    yb: usize,
) -> Option<*mut ThreadState<'gc>> {
    let bt = unsafe { &mut *b };
    bt.yield_bottom = Some(yb);
    bt.status = ThreadStatus::Suspended;
    let r = bt.resumer.filter(|_| !bt.main)?;
    let a: *mut ThreadState<'gc> = unsafe { r.state_mut(ctx.mutation()) };
    let at = unsafe { &mut *a };
    at.copy_in(
        at.resume_wait,
        unsafe { bt.stack.as_ptr().add(yb) },
        bt.top - yb,
    );
    link_yield(bt, at, yb);
    Some(a)
}

/// Yield from the running coroutine `b` (see [`begin_yield`]) and wait to be
/// resumed: returns how many values it was resumed with, now at
/// `b.stack[yb..]`.
///
/// # Safety
/// As [`begin_yield`].
pub(crate) unsafe fn yield_to<'gc>(
    ctx: Context<'gc>,
    b: *mut ThreadState<'gc>,
    yb: usize,
) -> usize {
    match unsafe { begin_yield(ctx, b, yb) } {
        // The host takes the values from `yield_bottom` and puts the ones it
        // resumes with there.
        None => park(ctx, Park::Yield),
        Some(a) => unsafe {
            switch_to(ctx, &raw mut (*b).sw, a, 0);
        },
    }
    resumed(b, yb)
}

/// The count of values the waiting coroutine `b` was resumed with at `yb`.
fn resumed<'gc>(b: *mut ThreadState<'gc>, yb: usize) -> usize {
    let bt = unsafe { &mut *b };
    bt.yield_bottom = None;
    bt.status = ThreadStatus::Normal;
    bt.top - yb
}

/// The running coroutine `b` is done: its resumer gets `err`, or else the
/// values at `b.stack[values..top]`, and `b` hands its stack over and
/// switches to it for good.
///
/// # Safety
/// As [`begin_resume`], for `b`, which has a resumer and no frames left.
unsafe fn finish<'gc>(
    ctx: Context<'gc>,
    b: *mut ThreadState<'gc>,
    values: usize,
    err: Option<Error<'gc>>,
) -> ! {
    unsafe {
        let bt = &mut *b;
        let r = bt
            .resumer
            .take()
            .expect("a coroutine ended without a resumer");
        let a: *mut ThreadState<'gc> = r.state_mut(ctx.mutation());
        let at = &mut *a;
        bt.top_base = std::ptr::null_mut();
        bt.top_pc = std::ptr::null();
        match err {
            Some(err) => {
                at.transfer = Some(err);
                bt.status = ThreadStatus::Stopped;
            }
            None => {
                at.copy_in(
                    at.resume_wait,
                    bt.stack.as_ptr().add(values),
                    bt.top - values,
                );
                bt.status = ThreadStatus::Result { bottom: 0 };
            }
        }
        at.status = ThreadStatus::Normal;
        // Released by `a`, off this stack.
        ctx.runtime().dead.set(bt.take_stack());
        let mut gone = crate::vm::stack::SwitchPoint::default();
        switch_to(ctx, &raw mut gone, a, 0);
    }
    std::process::abort()
}

/// The running coroutine `b` returned the values at `b.stack[values..top]`.
///
/// # Safety
/// As [`finish`].
pub(crate) unsafe fn end_to<'gc>(ctx: Context<'gc>, b: *mut ThreadState<'gc>, values: usize) -> ! {
    unsafe { finish(ctx, b, values, None) }
}

/// The running coroutine `b` died of `err`, which nothing on it caught.
///
/// # Safety
/// As [`finish`].
pub(crate) unsafe fn die_to<'gc>(
    ctx: Context<'gc>,
    b: *mut ThreadState<'gc>,
    err: Error<'gc>,
) -> ! {
    unsafe { finish(ctx, b, 0, Some(err)) }
}

/// Whether the running thread `ts` may resume `co` with `n` values without
/// the builtin: anything else is the builtin's to raise or refuse.
#[inline(always)]
fn may_resume<'gc>(ts: &ThreadState<'gc>, co: Thread<'gc>, n: usize) -> bool {
    if co.ptr_eq(ts.handle()) || ts.resume_depth + 1 >= MAX_RESUME_DEPTH {
        return false;
    }
    let Ok(cs) = co.inner().try_borrow() else {
        return false;
    };
    let bottom = cs.yield_bottom.unwrap_or(HDR);
    cs.status == ThreadStatus::Suspended && !cs.main && n <= cs.stack_limit.saturating_sub(bottom)
}

/// The coroutine builtins compiled code switches for itself
/// (`abi::coro_resume` and its siblings).
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Switch {
    Resume,
    Wrap,
    Yield,
}

/// Which of the switching builtins `nc` is, by its fast entry.
pub(crate) fn switch_kind(nc: &NativeClosure<'_>) -> Option<Switch> {
    let e = handler_bits(nc.entry);
    [
        (handler_bits(ff_resume), Switch::Resume),
        (handler_bits(ff_wrap), Switch::Wrap),
        (handler_bits(ff_yield), Switch::Yield),
    ]
    .into_iter()
    .find_map(|(h, k)| (h == e).then_some(k))
}

/// Make the native frame of compiled code's call of `nc` with the `n`
/// values at window `w` (index `win`) wait with `h` for what comes back,
/// returning through `ret_shim` (whose walks find the caller through
/// `shim`).
#[inline(always)]
unsafe fn compiled_wait<'gc>(
    ts: &mut ThreadState<'gc>,
    (w, win, n): (*mut Value<'gc>, usize, usize),
    nc: &NativeClosure<'gc>,
    h: NativeHdr,
    protect: u64,
) {
    unsafe {
        let p = w.sub(HDR).cast::<u64>();
        p.write(h.pack(nc));
        p.add(1)
            .write(crate::vm::stack::ret_shim_bits() | flag::NATIVE | protect);
    }
    ts.top_base = w;
    ts.top = win + n;
}

/// Start compiled code's call of `coroutine.resume` (`wrap` false) or of a
/// `wrap` function `nc` on the `n` values at `w`, as `ff_resume` and
/// `ff_wrap` do, when nothing on the way calls out: the coroutine yielded
/// before, its committed stack takes the values, and writing to it needs no
/// barrier. Returns the thread to switch to, now the running one, or `None`
/// with nothing changed for the builtin's own path.
///
/// # Safety
/// `a` is the running thread, `w` a window on its stack with room for a
/// header below it, and no reference to its state is live.
#[inline(always)]
pub(crate) unsafe fn compiled_resume<'gc>(
    ctx: Context<'gc>,
    a: *mut ThreadState<'gc>,
    w: *mut Value<'gc>,
    n: usize,
    nc: &NativeClosure<'gc>,
    wrap: bool,
) -> Option<*mut ThreadState<'gc>> {
    let ts = unsafe { &mut *a };
    let (co, at, h) = if wrap {
        let co = unsafe {
            nc.upvalues()
                .get_unchecked(0)
                .get_thread()
                .unwrap_unchecked()
        };
        (
            co,
            0,
            NativeHdr {
                at: 0,
                cont: cont::WRAP.0,
                ok: ok::RETURN,
            },
        )
    } else {
        // Without the thread argument the builtin raises.
        if n == 0 {
            return None;
        }
        let co = unsafe { w.read() }.get_thread()?;
        (
            co,
            1,
            NativeHdr {
                at: 1,
                cont: cont::RESUME.0,
                ok: ok::RETURN_TRUE,
            },
        )
    };
    let k = n - at;
    if !may_resume(ts, co, k) {
        return None;
    }
    // SAFETY: `co` is suspended, so nothing else uses its state.
    let b = unsafe { co.state_mut_if_gray() }?;
    let yb = b.yield_bottom?;
    if b.stack.len() < yb + k {
        return None;
    }
    let win = ts.slot_index(w);
    unsafe {
        compiled_wait(ts, (w, win, n), nc, h, flag::PROTECTED);
        frame::copy_values(b.slot_ptr(yb), w.add(at), k);
    }
    b.yield_bottom = None;
    b.top = yb + k;
    link_resume(ts, b, win + at);
    ctx.set_thread(b);
    Some(b)
}

/// Start compiled code's call of `coroutine.yield` `nc` on the `n` values
/// at `w`, as `ff_yield` does, when nothing on the way calls out; see
/// [`compiled_resume`].
///
/// # Safety
/// As [`compiled_resume`].
#[inline(always)]
pub(crate) unsafe fn compiled_yield<'gc>(
    ctx: Context<'gc>,
    b: *mut ThreadState<'gc>,
    w: *mut Value<'gc>,
    n: usize,
    nc: &NativeClosure<'gc>,
) -> Option<*mut ThreadState<'gc>> {
    let ts = unsafe { &mut *b };
    if ts.no_yield || ts.main {
        return None;
    }
    // SAFETY: the resumer waits, so nothing else uses its state.
    let a = unsafe { ts.resumer?.state_mut_if_gray() }?;
    let wait = a.resume_wait;
    if a.stack.len() < wait + n {
        return None;
    }
    let win = ts.slot_index(w);
    unsafe {
        compiled_wait(
            ts,
            (w, win, n),
            nc,
            NativeHdr {
                at: 0,
                cont: 0,
                ok: ok::RETURN,
            },
            0,
        );
        frame::copy_values(a.slot_ptr(wait), w, n);
    }
    a.top = wait + n;
    ts.yield_bottom = Some(win);
    ts.status = ThreadStatus::Suspended;
    link_yield(ts, a, win);
    ctx.set_thread(a);
    Some(a)
}

/// A yield while `coroutine.close`/`wrap` closes the thread's variables.
pub(crate) fn yield_across_close(ctx: Context<'_>) -> Error<'_> {
    let msg = crate::env::LuaString::new(ctx, b"attempt to yield across a C-call boundary");
    Error::new(ctx, Value::string(msg))
}

/// Resume `co` from the CALL running this entry (`ff_resume`, `ff_wrap`) as
/// its builtin would: the CALL becomes a native frame (window at `R[a+4]`,
/// the values from slot `$at`, `$ok` and `$cont` for what comes back), the
/// arguments from `R[a + 4 + $skip]` go to `co`, and the values land through
/// the CALL's continuation. Anything the switch doesn't cover goes to the
/// builtin.
macro_rules! resume_entry {
    ($pc:ident, $base:ident, $rt:ident, $closure:ident, $call:ident, $co:expr, $skip:literal, $at:literal, $ok:expr, $cont:expr) => {{
        let co: Thread<'gc> = $co;
        let (a, b, c) = $call.abc();
        let ts = thread!();
        let bi = ts.slot_index($base);
        let win = bi + a as usize + HDR;
        let args = win + $skip;
        let n = if b == 0 {
            ts.top - args
        } else {
            b as usize - 1 - $skip
        };
        if !may_resume(ts, co, n) {
            tail!(native_call)
        }
        let nc = unsafe { $closure.as_native() };
        let win_ptr = unsafe { $base.add(a as usize + HDR) };
        unsafe {
            frame::write_hdr(
                $base.add(a as usize),
                NativeHdr {
                    at: $at,
                    cont: $cont,
                    ok: $ok,
                }
                .pack(nc),
                handler_bits($rt.ret(c)) | flag::NATIVE | flag::PROTECTED,
                $base,
                $pc,
            );
        }
        ts.top_base = win_ptr;
        // The window is the published frame's, live while it waits.
        ts.top = args + n;
        let b = unsafe { begin_resume($rt, thread_ptr!(), co, args, n, win + $at) };
        switch_out!(b)
    }};
}

/// Wait at the running thread's switch point to continue in `switch_land`,
/// and switch to the thread `$to`.
macro_rules! switch_out {
    ($to:expr) => {{
        let to: *mut ThreadState<'gc> = $to;
        unsafe {
            if (*to).sw.cont == 0 {
                (*to).sw =
                    crate::vm::stack::fresh((*to).mem.as_ref().expect("a thread without a stack"));
            }
        }
        let from = thread_ptr!();
        switch!(to);
        tail!(
            crate::vm::stack::switch_handler,
            pc = switch_land as *const Instruction,
            insn = Slot::from_raw(from as u64),
            closure = Slot::from_raw(to as u64)
        )
    }};
}

handler! {
    bind(insn, pc, base, rt, closure, thread, nret, values);

    /// The entry of a `coroutine.wrap` function; see [`resume_entry`].
    entry fn ff_wrap {
        let call = insn.as_insn();
        if call.op() == Op::TAILCALL {
            tail!(native_call)
        }
        let nc = unsafe { closure.as_native() };
        let co = unsafe { nc.upvalues().get_unchecked(0).get_thread().unwrap_unchecked() };
        resume_entry!(pc, base, rt, closure, call, co, 0, 0, ok::RETURN, cont::WRAP.0)
    }

    /// The entry of `coroutine.resume`; see [`resume_entry`]. `true` goes in
    /// the window's first slot, before the values.
    entry fn ff_resume {
        let call = insn.as_insn();
        if call.op() == Op::TAILCALL {
            tail!(native_call)
        }
        let (a, b) = (call.a(), call.b());
        // Without the thread argument the builtin raises.
        let co = if b != 1 { reg![a + 4].get_thread() } else { None };
        let Some(co) = co else {
            tail!(native_call)
        };
        resume_entry!(pc, base, rt, closure, call, co, 1, 1, ok::RETURN_TRUE, cont::RESUME.0)
    }

    /// The entry of `coroutine.yield` from a coroutine: this CALL becomes the
    /// yield's native frame, its arguments go to the resumer, and the values
    /// it is resumed with land through the CALL's continuation. A main
    /// thread's yield and a yield where it is an error go to the builtin.
    entry fn ff_yield {
        let call = insn.as_insn();
        let ts = thread!();
        if call.op() == Op::TAILCALL || ts.resumer.is_none() || ts.no_yield || ts.main {
            tail!(native_call)
        }
        let (a, b, c) = call.abc();
        let bi = ts.slot_index(base);
        let yb = bi + a as usize + HDR;
        let n = if b == 0 { ts.top - yb } else { b as usize - 1 };
        let nc = unsafe { closure.as_native() };
        let yb_ptr = unsafe { base.add(a as usize + HDR) };
        unsafe {
            frame::write_hdr(
                base.add(a as usize),
                NativeHdr { at: 0, cont: 0, ok: ok::RETURN }.pack(nc),
                handler_bits(rt.ret(c)) | flag::NATIVE,
                base,
                pc,
            );
        }
        ts.top_base = yb_ptr;
        ts.top = yb + n;
        let Some(a) = (unsafe { begin_yield(rt, thread_ptr!(), yb) }) else {
            unreachable!("a coroutine without a resumer")
        };
        switch_out!(a)
    }

    /// Where a thread that waited in a handler (`switch_out!`) continues,
    /// its waiting native frame on top: a resume's frame gets what the
    /// coroutine yielded, returned or died of, a yield's frame what it was
    /// resumed with.
    slow fn switch_land {
        crate::vm::stack::release_dead(rt);
        let ts = thread!();
        let win = ts.top_base_index();
        let j = match ts.transfer.take() {
            Some(err) => native_catch(rt, thread_ptr!(), win, err),
            None => native_continue(rt, thread_ptr!(), win),
        };
        jump!(j)
    }

    /// Continuation of a coroutine's body: the coroutine is dead, and its
    /// results go to the resumer.
    cont fn ret_coroutine_end {
        let ts = thread!();
        let vi = ts.slot_index(values);
        ts.top = vi + nret;
        unsafe { end_to(rt, thread_ptr!(), vi) }
    }
}
