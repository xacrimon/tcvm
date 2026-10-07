//! Coroutine switches inside dispatch: the `resume`, `wrap` and `yield`
//! fast entries, the coroutine body's bottom continuation, and the switch
//! routines the native driver and the unwinder share.

use crate::env::Thread;
use crate::env::error::Error;
use crate::env::thread::{ThreadState, ThreadStatus};
use crate::env::value::Value;
use crate::instruction::Op;
use crate::lua::Context;
use crate::vm::abi::{Exit, Jump, handler, handler_bits};
use crate::vm::frame::{self, HDR, NativeHdr, copy_values, flag};
use crate::vm::native::{native_call, native_continue, ok};
use crate::vm::ops::call::seed_header;

/// `LUAI_MAXCCALLS`: threads resumed inside one another.
pub(crate) const MAX_RESUME_DEPTH: u16 = 200;

/// Whether `ts`'s top frame is a native's waiting for a coroutine or a call:
/// where a switch back can deliver values.
#[inline]
fn waits_in_dispatch(ts: &ThreadState<'_>) -> bool {
    ts.top_is_native()
}

/// Switch from `ts` to the suspended coroutine `co`, resumed with
/// `ts.stack[args .. top]`: the arguments land where it waits for them and
/// its continuation runs there.
pub(crate) fn resume_into<'gc>(
    ctx: Context<'gc>,
    ts: &mut &mut ThreadState<'gc>,
    co: Thread<'gc>,
    args: usize,
) -> Jump<'gc> {
    let t = &mut **ts;
    let n = t.top - args;
    // SAFETY: `co` is suspended, so nothing else uses its state.
    let cs = unsafe { co.state_mut(ctx.mutation()) };
    let yb = if cs.started {
        cs.yield_bottom
            .take()
            .expect("resumed a coroutine that didn't yield")
    } else {
        HDR
    };
    cs.ensure_slots(yb + n);
    unsafe { copy_values(cs.slot_ptr(yb), t.stack.as_ptr().add(args), n) };
    cs.top = yb + n;
    t.set_top_unchecked(args - 1);
    t.status = ThreadStatus::Normal;
    cs.status = ThreadStatus::Normal;
    cs.resumer = Some(t.handle());
    cs.resume_depth = t.resume_depth + 1;
    *ts = cs;
    ctx.set_thread(*ts);
    if !ts.started {
        let nargs = seed_header(ts, ret_coroutine_end);
        return Jump::Enter {
            hdr: ts.slot_ptr(0),
            nargs,
            base: std::ptr::null_mut(),
        };
    }
    deliver(ctx, ts, yb)
}

/// Deliver the values at `yb .. top` to the frame waiting for them at the top
/// of `ts`: a yield frame or a native frame.
pub(crate) fn deliver<'gc>(
    ctx: Context<'gc>,
    ts: &mut &mut ThreadState<'gc>,
    yb: usize,
) -> Jump<'gc> {
    debug_assert!(ts.top_is_native());
    let win = ts.top_base_index();
    debug_assert_eq!(
        win + NativeHdr::unpack(unsafe { frame::func_word(ts.top_base) }).at,
        yb
    );
    native_continue(ctx, ts, win)
}

/// A yield while `coroutine.close`/`wrap` closes the thread's variables.
pub(crate) fn yield_across_close(ctx: Context<'_>) -> Error<'_> {
    let msg = crate::env::LuaString::new(ctx, b"attempt to yield across a C-call boundary");
    Error::new(ctx, Value::string(msg))
}

/// Yield `ts.stack[yb .. top]`: to a resumer waiting in dispatch, switching
/// to it, or to the host. The frame on top of `ts` takes the values a resume
/// delivers at `yb`.
pub(crate) fn yield_from<'gc>(
    ctx: Context<'gc>,
    ts: &mut &mut ThreadState<'gc>,
    yb: usize,
) -> Jump<'gc> {
    let t = &mut **ts;
    if t.no_yield {
        return crate::vm::unwind::unwind(ctx, ts, yield_across_close(ctx));
    }
    t.yield_bottom = Some(yb);
    t.status = ThreadStatus::Suspended;
    let resumer = t
        .resumer
        .filter(|_| !t.main)
        .map(|r| unsafe { r.state_mut(ctx.mutation()) })
        .filter(|rs| waits_in_dispatch(rs));
    let Some(rs) = resumer else {
        // The host takes the yield; the values stay at `yb`.
        return Jump::Exit(Exit::End);
    };
    let n = t.top - yb;
    let rwin = rs.top_base_index();
    let slot = rwin + NativeHdr::unpack(unsafe { frame::func_word(rs.top_base) }).at;
    rs.ensure_slots(slot + n);
    unsafe { copy_values(rs.slot_ptr(slot), t.stack.as_ptr().add(yb), n) };
    rs.set_top_unchecked(slot + n);
    rs.status = ThreadStatus::Normal;
    t.resumer = None;
    t.set_top_unchecked(yb);
    *ts = rs;
    ctx.set_thread(*ts);
    native_continue(ctx, ts, rwin)
}

/// The coroutine `ts` ended with the values at `values .. top`: hand them to
/// the resumer waiting in dispatch, which becomes the running thread.
pub(crate) fn end_into_resumer<'gc>(
    ctx: Context<'gc>,
    ts: &mut &mut ThreadState<'gc>,
    values: usize,
) -> Option<Jump<'gc>> {
    let t = &mut **ts;
    let rs = t
        .resumer
        .map(|r| unsafe { r.state_mut(ctx.mutation()) })
        .filter(|rs| waits_in_dispatch(rs))?;
    let n = t.top - values;
    let rwin = rs.top_base_index();
    let slot = rwin + NativeHdr::unpack(unsafe { frame::func_word(rs.top_base) }).at;
    rs.ensure_slots(slot + n);
    unsafe { copy_values(rs.slot_ptr(slot), t.stack.as_ptr().add(values), n) };
    rs.set_top_unchecked(slot + n);
    rs.status = ThreadStatus::Normal;
    t.resumer = None;
    t.top_base = std::ptr::null_mut();
    t.top_pc = std::ptr::null();
    t.discard_above(0);
    t.status = ThreadStatus::Result { bottom: 0 };
    *ts = rs;
    ctx.set_thread(*ts);
    Some(native_continue(ctx, ts, rwin))
}

/// Resume `co` from the CALL running this entry (`ff_resume`, `ff_wrap`): the
/// caller waits in a native frame that takes the results by `$ok`, with
/// `$cont` for errors; the arguments from `R[a + 4 + $skip]` move to where
/// `co` last yielded and its continuation runs there. Anything the switch
/// doesn't cover (a first resume, a non-suspended coroutine, no room) goes to
/// the builtin.
macro_rules! resume_switch {
    ($pc:ident, $base:ident, $rt:ident, $closure:ident, $thread:ident, $call:ident, $co:expr, $skip:literal, $ok:expr, $cont:expr) => {{
        let co: Thread<'gc> = $co;
        let ts = thread!();
        if co.ptr_eq(ts.handle())
            || co.peer_status() != ThreadStatus::Suspended
            || ts.resume_depth + 1 >= MAX_RESUME_DEPTH
        {
            tail!(native_call)
        }
        let (a, b, c) = $call.abc();
        // SAFETY: `co` is suspended, so nothing else uses its state.
        let Some(cs) = (unsafe { co.state_mut_if_clean($rt.mutation()) }) else {
            tail!(native_call)
        };
        let Some(yb) = cs.yield_bottom.filter(|_| cs.started && cs.top_is_native()) else {
            tail!(native_call)
        };
        let bi = ts.slot_index($base);
        let win = bi + a as usize + HDR;
        let args = win + $skip;
        let n = if b == 0 {
            ts.top - args
        } else {
            b as usize - 1 - $skip
        };
        if yb + n > cs.stack.len() || yb + n > cs.stack_limit {
            tail!(native_call)
        }
        let nc = unsafe { $closure.as_native() };
        unsafe {
            frame::write_hdr(
                $base.add(a as usize),
                NativeHdr {
                    at: 0,
                    cont: $rt.cont_index($cont),
                    ok: $ok,
                }
                .pack(nc),
                handler_bits($rt.ret(c)) | flag::NATIVE | flag::PROTECTED,
                $base,
                $pc,
            );
        }
        ts.top_base = ts.slot_ptr(win);
        ts.set_top_unchecked(win);
        ts.status = ThreadStatus::Normal;
        cs.yield_bottom = None;
        cs.status = ThreadStatus::Normal;
        cs.resumer = Some(ts.handle());
        cs.resume_depth = ts.resume_depth + 1;
        // Copied rather than passed by pointer: continuations take values
        // in their own thread's stack.
        unsafe { copy_values(cs.slot_ptr(yb), ts.stack.as_ptr().add(args), n) };
        cs.set_top_unchecked(yb + n);
        switch!(cs);
        let mut tsr: &mut ThreadState<'gc> = thread!();
        let j = deliver($rt, &mut tsr, yb);
        jump!(j)
    }};
}

handler! {
    bind(insn, pc, base, rt, closure, thread, nret, values);

    /// The entry of a `coroutine.wrap` function; see [`resume_switch`].
    entry fn ff_wrap {
        let call = insn.as_insn();
        if call.op() == Op::TAILCALL {
            tail!(native_call)
        }
        let nc = unsafe { closure.as_native() };
        let co = unsafe { nc.upvalues().get_unchecked(0).get_thread().unwrap_unchecked() };
        resume_switch!(pc, base, rt, closure, thread, call, co, 0, ok::RETURN, crate::builtin::wrap_cont)
    }

    /// The entry of `coroutine.resume`; see [`resume_switch`].
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
        resume_switch!(pc, base, rt, closure, thread, call, co, 1, ok::RETURN_TRUE, crate::builtin::resume_cont)
    }

    /// The entry of `coroutine.yield`: with the resumer waiting in
    /// dispatch on a frame that takes the values as they are, this CALL
    /// becomes the yield frame, the values move into the resumer's window and
    /// its frame continues. Anything else goes to the builtin.
    entry fn ff_yield {
        let call = insn.as_insn();
        let ts = thread!();
        let Some(r) = ts.resumer else {
            tail!(native_call)
        };
        if call.op() == Op::TAILCALL || ts.no_yield || ts.main {
            tail!(native_call)
        }
        let (a, b, c) = call.abc();
        // SAFETY: the resumer waits, so nothing else uses its state.
        let Some(rs) = (unsafe { r.state_mut_if_clean(rt.mutation()) }) else {
            tail!(native_call)
        };
        if !rs.top_is_native() {
            tail!(native_call)
        }
        let rh = NativeHdr::unpack(unsafe { frame::func_word(rs.top_base) });
        if rh.ok == ok::CONT {
            tail!(native_call)
        }
        let bi = ts.slot_index(base);
        let yb = bi + a as usize + HDR;
        let n = if b == 0 { ts.top - yb } else { b as usize - 1 };
        let slot = rs.top_base_index() + rh.at;
        if slot + n > rs.stack.len() || slot + n >= rs.stack_limit {
            tail!(native_call)
        }
        let nc = unsafe { closure.as_native() };
        unsafe {
            frame::write_hdr(
                base.add(a as usize),
                NativeHdr { at: 0, cont: 0, ok: ok::RETURN }.pack(nc),
                handler_bits(rt.ret(c)) | flag::NATIVE,
                base,
                pc,
            );
        }
        ts.top_base = ts.slot_ptr(yb);
        ts.yield_bottom = Some(yb);
        ts.status = ThreadStatus::Suspended;
        ts.resumer = None;
        unsafe { copy_values(rs.slot_ptr(slot), ts.stack.as_ptr().add(yb), n) };
        ts.set_top_unchecked(yb);
        rs.set_top_unchecked(slot + n);
        rs.status = ThreadStatus::Normal;
        let rwin = rs.top_base_index();
        switch!(rs);
        let mut tsr: &mut ThreadState<'gc> = thread!();
        let j = native_continue(rt, &mut tsr, rwin);
        jump!(j)
    }

    /// Continuation of a coroutine's body: the coroutine is dead, and its
    /// results go to the resumer waiting in dispatch.
    cont fn ret_coroutine_end {
        let mut tsr: &mut ThreadState<'gc> = thread!();
        let vi = tsr.slot_index(values);
        tsr.set_top_unchecked(vi + nret);
        match end_into_resumer(rt, &mut tsr, vi) {
            Some(j) => jump!(j),
            None => tail!(crate::vm::ops::call::ret_exit),
        }
    }
}
