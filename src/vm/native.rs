//! Natives: the protocol between dispatch and Rust functions. Every
//! native call writes a frame header; a plain return lands through the
//! header's continuation, anything else goes through `native_act`.

use crate::env::Thread;
use crate::env::error::Error;
use crate::env::function::{NativeClosure, NativeKind, Stack};
use crate::env::thread::ThreadState;
use crate::env::value::Value;
use crate::instruction::{Instruction, Op};
use crate::lua::Context;
use crate::vm::abi::{Exit, Jump, Slot, handler, handler_bits};
use crate::vm::frame::{self, HDR, NativeHdr, copy_values, copy_values_up, fill_nil, flag};
use crate::vm::ops::control::close_upvalues;

/// Whether a continuation receives the errors its call raises, instead of
/// them unwinding past it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Protect {
    No,
    /// `pcall`.
    Errors,
    /// `xpcall`: as `Errors`, after running the message handler at
    /// `stack[0]` on top of the failing frames.
    Handler,
    /// As `Errors`, and an exit too: the thread's base level
    /// (`luaD_throwbaselevel`).
    Base,
}

/// What a continuation does with the results of the call or resume that
/// succeeded, when that is simple enough for the VM to do instead of calling
/// it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum OnOk {
    Cont,
    /// They are the native's results.
    Return,
    /// `true` and then them (`pcall`).
    ReturnTrue,
}

/// A native that may ask the VM to act once it returns: call a
/// function and continue in a [`NativeCont`], resume or yield a coroutine,
/// or suspend into the executor. `?` on a `Result<_, Error>` works in one:
/// the error becomes [`NativeOut::error`].
pub(crate) type ContFn = for<'gc, 'a> fn(
    ctx: Context<'gc>,
    closure: &'a NativeClosure<'gc>,
    stack: Stack<'gc, 'a>,
) -> NativeOut;

/// The continuation of a [`NativeOut::call_then`]: the native's window with
/// the call's results at `at`, or with nothing above `at` and the error when
/// a protected call failed.
pub(crate) type NativeCont = for<'gc, 'a> fn(
    ctx: Context<'gc>,
    closure: &'a NativeClosure<'gc>,
    stack: Stack<'gc, 'a>,
    status: Result<(), Error<'gc>>,
) -> NativeOut;

/// A continuation's index in [`CONT_TABLE`]; six bits in a native frame's
/// word 0 and in a [`NativeOut`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct ContIdx(pub(crate) u8);

/// The continuation table and its indices by name. The test
/// continuations follow the others, so every index holds in every build.
macro_rules! cont_table {
    ($($name:ident => $f:path),* $(,)? ; test: $($tname:ident => $tf:path),* $(,)?) => {
        pub(crate) mod cont {
            use super::ContIdx;
            $(pub(crate) const $name: ContIdx = ContIdx(${index()});)*
            #[cfg(test)]
            const TEST_BASE: usize = ${count($name)};
            $(
                #[cfg(test)]
                pub(crate) const $tname: ContIdx = ContIdx((TEST_BASE + ${index()}) as u8);
            )*
        }
        #[cfg(not(test))]
        pub(crate) static CONT_TABLE: &[NativeCont] = &[$($f as NativeCont,)*];
        #[cfg(test)]
        pub(crate) static CONT_TABLE: &[NativeCont] = &[$($f as NativeCont,)* $($tf as NativeCont,)*];
    };
}

cont_table! {
    NONE => no_cont,
    ASYNC => crate::vm::async_native::async_cont,
    RESUME => crate::builtin::resume_cont,
    WRAP => crate::builtin::wrap_cont,
    WRAP_CLOSE => crate::builtin::wrap_close_cont,
    CORO_CLOSE => crate::builtin::coroutine_close_cont,
    PCALL => crate::builtin::pcall_cont,
    XPCALL => crate::builtin::xpcall_cont,
    PAIRS => crate::builtin::pairs_cont,
    TOSTRING => crate::builtin::tostring_cont,
    DOFILE => crate::builtin::return_cont,
    SORT_LEN => crate::builtin::sort_len_cont,
    SORT => crate::builtin::sort_cont,
    GSUB => crate::builtin::gsub_cont,
    ONE_RESULT => crate::builtin::one_result,
    HANDLER => crate::vm::unwind::handler_cont,
    CLOSE => crate::vm::unwind::close_cont,
    CLOSE_ENTRY => crate::vm::close::close_entry_cont,
    CLOSE_RUNNING => crate::vm::close::close_running_cont;
    test:
    TEST_RETURNED => crate::vm::tests::protocol::returned,
    TEST_ADD_ONE => crate::vm::tests::protocol::add_one,
    TEST_CL_ADD_ONE => crate::vm::tests::calls_lua::add_one,
    TEST_CL_RETURNED => crate::vm::tests::calls_lua::returned,
    TEST_THROUGH => crate::vm::tests::pcall_through::through_cont,
}

const _: () = assert!(CONT_TABLE.len() <= 64, "more than 64 continuation natives");

/// The continuation at `idx`.
#[inline(always)]
pub(crate) fn cont_fn(idx: u8) -> NativeCont {
    CONT_TABLE[idx as usize]
}

/// [`cont::NONE`]'s entry: never called.
fn no_cont<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    _stack: Stack<'gc, '_>,
    _status: Result<(), Error<'gc>>,
) -> NativeOut {
    NativeOut::error(Error::from_str(
        ctx,
        "internal: continuation of a frame that awaits nothing",
    ))
}

/// Read-only view of the executor, reached from a native through
/// [`Stack::exec`].
#[derive(Clone, Copy)]
pub struct Execution<'gc> {
    current_thread: Thread<'gc>,
    is_main: bool,
}

impl<'gc> Execution<'gc> {
    pub fn new(current_thread: Thread<'gc>, is_main: bool) -> Self {
        Execution {
            current_thread,
            is_main,
        }
    }

    /// Thread the native is running on.
    pub fn current_thread(self) -> Thread<'gc> {
        self.current_thread
    }

    /// Whether the running thread is the executor's main (entry) thread.
    pub fn is_main(self) -> bool {
        self.is_main
    }
}

/// `OnOk` as the two-bit `ok` field of a native frame's word 0.
pub(crate) mod ok {
    pub(crate) const CONT: u8 = 0;
    pub(crate) const RETURN: u8 = 1;
    pub(crate) const RETURN_TRUE: u8 = 2;
}

impl OnOk {
    pub(crate) fn bits(self) -> u8 {
        match self {
            OnOk::Cont => ok::CONT,
            OnOk::Return => ok::RETURN,
            OnOk::ReturnTrue => ok::RETURN_TRUE,
        }
    }
}

impl Protect {
    /// The frame flags of a native frame protected this way.
    pub(crate) fn flags(self) -> u64 {
        match self {
            Protect::No => 0,
            Protect::Errors | Protect::Base => flag::PROTECTED,
            Protect::Handler => flag::PROTECTED | flag::HANDLER,
        }
    }
}

/// A native's result in one word: tag in the low 3 bits, then the
/// payload the tag needs.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub(crate) struct NativeOut(u64);

/// [`NativeOut`] tags.
pub(crate) mod tag {
    pub(crate) const RETURN: u64 = 0;
    pub(crate) const ERROR: u64 = 1;
    pub(crate) const CALL_THEN: u64 = 2;
    pub(crate) const RESUME: u64 = 3;
    pub(crate) const YIELD: u64 = 4;
    pub(crate) const YIELD_THEN: u64 = 5;
    pub(crate) const ASYNC: u64 = 6;
    pub(crate) const PENDING: u64 = 7;
}

impl NativeOut {
    /// A plain return: the results are the window.
    pub(crate) const RETURN: NativeOut = NativeOut(tag::RETURN);
    /// Yield the window to the resumer; the values it resumes with are the
    /// native's results.
    pub(crate) const YIELD: NativeOut = NativeOut(tag::YIELD);
    /// The async native's future was spawned (`Stack::spawn`): poll it from
    /// a frame of its own.
    pub(crate) const ASYNC: NativeOut = NativeOut(tag::ASYNC);
    /// Leave the native's frame for the host to come back to: its future
    /// waits on the host.
    pub(crate) const PENDING: NativeOut = NativeOut(tag::PENDING);

    const AT_SHIFT: u32 = 48;
    const CONT_SHIFT: u32 = 56;
    const OK_SHIFT: u32 = 62;

    #[inline(always)]
    pub(crate) fn raw(self) -> u64 {
        self.0
    }

    #[inline(always)]
    pub(crate) fn from_raw(v: u64) -> Self {
        NativeOut(v)
    }

    #[inline(always)]
    pub(crate) fn tag(self) -> u64 {
        self.0 & 7
    }

    #[inline(always)]
    pub(crate) fn is_return(self) -> bool {
        self.0 == tag::RETURN
    }

    /// A plain native's result.
    #[inline(always)]
    pub(crate) fn plain(r: Result<(), Error<'_>>) -> Self {
        match r {
            Ok(()) => NativeOut(tag::RETURN),
            Err(e) => Self::error(e),
        }
    }

    #[inline(always)]
    pub(crate) fn error(e: Error<'_>) -> Self {
        NativeOut(tag::ERROR | crate::dmm::Gc::as_ptr(e.inner()) as usize as u64)
    }

    /// The error of an `ERROR` result.
    #[inline(always)]
    pub(crate) fn into_error<'gc>(self) -> Error<'gc> {
        debug_assert_eq!(self.tag(), tag::ERROR);
        unsafe { Error::from_inner(crate::dmm::Gc::from_ptr((self.0 & !7) as usize as *const _)) }
    }

    fn packed(tag: u64, at: usize, cont: ContIdx, okk: OnOk, protect: Protect) -> Self {
        debug_assert!(at < 256);
        // Bits 3 and 4 hold the frame flags themselves, so staging needs no
        // decode; `Base` is `Errors` plus its continuation's identity.
        const _: () = assert!(flag::PROTECTED | flag::HANDLER == 0x18);
        NativeOut(
            tag | protect.flags()
                | (at as u64) << Self::AT_SHIFT
                | (cont.0 as u64) << Self::CONT_SHIFT
                | (okk.bits() as u64) << Self::OK_SHIFT,
        )
    }

    /// Call `stack[at]` with the values above it, then run `cont` with the
    /// call's results in their place, `stack[at..]`. The native keeps its
    /// state in `stack[..at]` meanwhile.
    pub(crate) fn call_then(at: usize, cont: ContIdx, protect: Protect, ok: OnOk) -> Self {
        Self::packed(tag::CALL_THEN, at, cont, ok, protect)
    }

    /// [`call_then`](Self::call_then) for a native that laid the call out
    /// as a frame already, `[f, _, _, _, args..]` at `at`, so the
    /// arguments need no move.
    pub(crate) fn call_then_staged(at: usize, cont: ContIdx, protect: Protect, ok: OnOk) -> Self {
        NativeOut(Self::call_then(at, cont, protect, ok).0 | Self::STAGED)
    }

    /// Set on a `CALL_THEN` whose arguments are in frame layout already.
    const STAGED: u64 = 1 << 5;

    #[inline]
    pub(crate) fn staged(self) -> bool {
        self.0 & Self::STAGED != 0
    }

    /// Resume the coroutine at `stack[at]` with the values above it, then run
    /// `cont` with what it yields or returns in their place, or with the
    /// error that killed it.
    pub(crate) fn resume(at: usize, cont: ContIdx, ok: OnOk) -> Self {
        Self::packed(tag::RESUME, at, cont, ok, Protect::Errors)
    }

    /// Yield `stack[at..]` to the resumer, then run `cont` with the values it
    /// resumes with in their place.
    pub(crate) fn yield_then(at: usize, cont: ContIdx) -> Self {
        Self::packed(tag::YIELD_THEN, at, cont, OnOk::Cont, Protect::No)
    }

    #[inline]
    pub(crate) fn at(self) -> usize {
        ((self.0 >> Self::AT_SHIFT) & 0xff) as usize
    }

    #[inline]
    pub(crate) fn cont(self) -> u8 {
        ((self.0 >> Self::CONT_SHIFT) & 0x3f) as u8
    }

    #[inline]
    pub(crate) fn ok(self) -> u8 {
        (self.0 >> Self::OK_SHIFT) as u8
    }

    /// The `PROTECTED`/`HANDLER` flags the staged call's frame gets.
    #[inline(always)]
    pub(crate) fn protect_flags(self) -> u64 {
        self.0 & (flag::PROTECTED | flag::HANDLER)
    }
}

impl<'gc> std::ops::FromResidual<Result<std::convert::Infallible, Error<'gc>>> for NativeOut {
    #[inline(always)]
    fn from_residual(r: Result<std::convert::Infallible, Error<'gc>>) -> Self {
        match r {
            Ok(never) => match never {},
            Err(e) => NativeOut::error(e),
        }
    }
}

/// Put `true` before the `nret` values at `values`, which start at or above
/// `b`, the base of the frame they return from: in the dead slot below them,
/// or, when they start at `b` (whose header the continuation still reads), by
/// moving them up one. Returns where the values now start.
pub(crate) fn prepend_true<'gc>(
    ts: &mut ThreadState<'gc>,
    b: usize,
    values: usize,
    nret: usize,
) -> usize {
    if values > b {
        ts.stack[values - 1] = Value::boolean(true);
        return values - 1;
    }
    // Growth here walks the chain from `b`, whose header is intact.
    ts.top_base = ts.slot_ptr(b);
    ts.ensure_slots(b + nret + 1);
    ts.stack.copy_within(b..b + nret, b + 1);
    ts.stack[b] = Value::boolean(true);
    ts.set_top_unchecked(b + nret + 1);
    b
}

/// A native pushed without `Stack::check_stack`.
#[cold]
#[inline(never)]
pub(crate) fn native_overflow(ctx: Context<'_>) -> Error<'_> {
    Error::from_str(ctx, "stack overflow")
}

/// Run `nc` on the window `win .. top`, which `top` must already bound.
#[inline(always)]
pub(crate) fn invoke<'gc>(
    ctx: Context<'gc>,
    ts: &mut ThreadState<'gc>,
    nc: &NativeClosure<'gc>,
    win: usize,
) -> NativeOut {
    let out = match nc.function {
        NativeKind::Plain(f) => NativeOut::plain(f(ctx, nc, Stack::new(ts, win))),
        NativeKind::Cont(f) => f(ctx, nc, Stack::new(ts, win)),
        NativeKind::Async(f) => crate::vm::async_native::invoke_async(ctx, ts, f, nc, win),
    };
    if out.tag() != tag::ERROR && ts.native_overflowed() {
        return NativeOut::error(native_overflow(ctx));
    }
    out
}

/// The argument count of the call whose first argument is at `win`, from a
/// CALL's `b`.
#[inline(always)]
fn call_nargs(ts: &ThreadState<'_>, b: u8, win: usize) -> usize {
    if b == 0 { ts.top - win } else { b as usize - 1 }
}

/// Leave for the collector with the native frame at window `win` holding
/// its results; `ok = RETURN` tells the trampoline to land them.
#[cold]
#[inline(never)]
fn gc_exit<'gc>(ts: &mut ThreadState<'gc>, win: *mut Value<'gc>, nc: &NativeClosure<'gc>) -> Exit {
    let nh = NativeHdr::unpack(unsafe { frame::func_word(win) });
    unsafe {
        frame::set_func_word(
            win,
            NativeHdr {
                ok: ok::RETURN,
                ..nh
            }
            .pack(nc),
        )
    };
    ts.top_base = win;
    Exit::Gc
}

/// Return the native frame at window `win`'s results, `win .. top`, through
/// its continuation.
pub(crate) fn land_pending<'gc>(ts: &mut ThreadState<'gc>, win: usize) -> Jump<'gc> {
    let win_ptr = ts.slot_ptr(win);
    Jump::Ret {
        ret: unsafe { frame::ret(win_ptr) },
        nret: ts.top - win,
        values: win_ptr,
        base: win_ptr,
    }
}

handler! {
    bind(insn, pc, base, rt, closure, thread, nret, values);

    /// CALL or TAILCALL of a native (the generic `NativeClosure::entry`):
    /// write its frame, run it, land a plain return through the CALL's
    /// continuation.
    entry fn native_call {
        let call = insn.as_insn();
        if call.op() == Op::TAILCALL {
            tail!(tailcall_native)
        }
        let (a, b, c) = call.abc();
        let nc = unsafe { closure.as_native() };
        let ts = thread!();
        let bi = ts.slot_index(base);
        let hdr = bi + a as usize;
        let win = hdr + HDR;
        let nargs = call_nargs(ts, b, win);
        let ret = rt.ret(c);
        unsafe {
            frame::write_hdr(
                base.add(a as usize),
                NativeHdr { at: 0, cont: 0, ok: ok::CONT }.pack(nc),
                handler_bits(ret) | flag::NATIVE,
                base,
                pc,
            );
        }
        ts.top_base = ts.slot_ptr(win);
        ts.top = win + nargs;
        let out = invoke(rt, ts, nc, win);
        // The native may have grown the stack.
        let sp = ts.stack.as_mut_ptr();
        if std::hint::likely(out.is_return()) {
            let nret = ts.top - win;
            let w = unsafe { sp.add(win) };
            if std::hint::unlikely(rt.gc_due()) {
                return gc_exit(ts, w, nc);
            }
            tail!(ret, pc = w as *const Instruction, insn = Slot::nret(nret), base = w)
        }
        tail!(native_act, pc = unsafe { sp.add(hdr) } as *const Instruction, insn = Slot::from_raw(out.raw()), base = unsafe { sp.add(bi) })
    }

    /// `enter` of a native: the header is written (word 0 as a raw value),
    /// `pc` is the header, `insn` the argument count.
    entry fn native_enter {
        let nargs = insn.as_nret();
        let nc = unsafe { closure.as_native() };
        let ts = thread!();
        let hdr = ts.slot_index(pc as *const Value<'gc>);
        let win = hdr + HDR;
        unsafe {
            let h = pc as *mut u64;
            h.write(NativeHdr { at: 0, cont: 0, ok: ok::CONT }.pack(nc));
            h.add(1).write(h.add(1).read() | flag::NATIVE);
        }
        ts.top_base = ts.slot_ptr(win);
        ts.top = win + nargs;
        let out = invoke(rt, ts, nc, win);
        let sp = ts.stack.as_mut_ptr();
        let w = unsafe { sp.add(win) };
        if std::hint::likely(out.is_return()) {
            let nret = ts.top - win;
            if std::hint::unlikely(rt.gc_due()) {
                return gc_exit(ts, w, nc);
            }
            let ret = unsafe { frame::ret(w) };
            tail!(ret, pc = w as *const Instruction, insn = Slot::nret(nret), base = w)
        }
        // `base` is passed on as it came: null or stale for a native caller,
        // which `native_act` never reads.
        tail!(native_act, pc = unsafe { sp.add(hdr) } as *const Instruction, insn = Slot::from_raw(out.raw()))
    }

    /// TAILCALL of a native: run it on the window `R[a+4] ..` and
    /// return its results from this frame, which writes no header of its
    /// own. An action converts this frame into the native's frame in place.
    entry fn tailcall_native {
        let call = insn.as_insn();
        let (a, b) = call.ab();
        let nc = unsafe { closure.as_native() };
        let ts = thread!();
        let bi = ts.slot_index(base);
        let win = bi + a as usize + HDR;
        let nargs = call_nargs(ts, b, win);
        // A tail call leaves this frame: close what it captured first.
        debug_assert!(!crate::vm::ops::control::has_tbc_from(ts, bi));
        close_upvalues(rt.mutation(), ts, bi);
        ts.top_base = base;
        ts.top_pc = pc;
        ts.top = win + nargs;
        let out = invoke(rt, ts, nc, win);
        let sp = ts.stack.as_mut_ptr();
        base = unsafe { sp.add(bi) };
        let returned = out.is_return();
        if std::hint::likely(returned) && std::hint::likely(!rt.gc_due()) {
            let nret = ts.top - win;
            let ret = unsafe { frame::ret(base) };
            tail!(ret, pc = unsafe { sp.add(win) } as *const Instruction, insn = Slot::nret(nret))
        }
        if out.tag() == tag::ERROR {
            // The native has no frame here, so its level 1 is this frame.
            let err = crate::vm::debug::locate_unframed(rt, ts, out.into_error());
            throw!(err)
        }
        // The native takes this frame's place: its window moves down to
        // `base`, so its results land where this frame's would.
        let n = ts.top - win;
        ts.stack.copy_within(win..win + n, bi);
        ts.top = bi + n;
        unsafe {
            frame::set_func_word(base, NativeHdr { at: 0, cont: 0, ok: ok::CONT }.pack(nc));
            let rw = frame::ret_word(base) & !(flag::HAS_OPEN | flag::HAS_TBC);
            frame::set_ret_word(base, rw | flag::NATIVE);
        }
        if returned {
            return gc_exit(ts, base, nc);
        }
        tail!(native_act, pc = unsafe { base.sub(HDR) } as *const Instruction, insn = Slot::from_raw(out.raw()))
    }

    /// A native asked for something other than a plain return: `pc` is its
    /// frame's header, `insn` the `NativeOut`. A call is staged here and
    /// entered directly; the other requests go to `drive`.
    slow fn native_act {
        let out = NativeOut::from_raw(insn.raw());
        let mut ts: &mut ThreadState<'gc> = thread!();
        let hdr = ts.slot_index(pc as *const Value<'gc>);
        if std::hint::likely(out.tag() == tag::CALL_THEN) {
            let nc = unsafe { frame::native_closure(ts.slot_ptr(hdr + HDR)) };
            let (callee, nargs) = stage_call(ts, hdr, nc, out);
            tail!(
                crate::vm::ops::call::enter,
                pc = callee as *const Instruction,
                insn = Slot::nret(nargs),
                base = std::ptr::null_mut()
            )
        }
        let j = drive(rt, &mut ts, hdr, out);
        jump!(j)
    }

    /// Continuation of a call a native frame made: the results go to
    /// the frame's window at `at`, its continuation runs, and its answer is
    /// acted on here: a return lands through the frame's own continuation, a
    /// call is staged and entered, the rest goes to `drive`. The pass-through
    /// modes are `native_continue`'s.
    cont fn ret_native {
        let win = unsafe { frame::caller_base(base) };
        let mut ts: &mut ThreadState<'gc> = thread!();
        let wi = ts.slot_index(win);
        let w0 = unsafe { frame::func_word(win) };
        let nh = NativeHdr::unpack(w0);
        let dst = unsafe { win.add(nh.at) };
        unsafe { copy_values(dst, values, nret) };
        // A single store: the continuation reads `top` alone.
        unsafe { std::ptr::write_volatile(&mut ts.top, wi + nh.at + nret) };
        ts.top_base = win;
        if std::hint::unlikely(nh.ok != ok::CONT) {
            let j = native_continue(rt, &mut ts, wi);
            jump!(j)
        }
        let nc = unsafe { frame::native_closure_of(w0) };
        // Errors the continuation itself raises unwind past it.
        unsafe { frame::clear_flags(win, flag::PROTECTED | flag::HANDLER) };
        let mut out = cont_fn(nh.cont)(rt, nc, Stack::new(ts, wi), Ok(()));
        if out.tag() != tag::ERROR && ts.native_overflowed() {
            out = NativeOut::error(native_overflow(rt));
        }
        // The continuation may have grown the stack.
        let win = ts.slot_ptr(wi);
        if std::hint::likely(out.is_return()) {
            if std::hint::unlikely(rt.gc_due()) {
                return gc_exit(ts, win, nc);
            }
            let nret = ts.top - wi;
            let ret = unsafe { frame::ret(win) };
            tail!(ret, pc = win as *const Instruction, insn = Slot::nret(nret), base = win)
        }
        if std::hint::likely(out.tag() == tag::CALL_THEN) {
            let (callee, nargs) = stage_call(ts, wi - HDR, nc, out);
            tail!(
                crate::vm::ops::call::enter,
                pc = callee as *const Instruction,
                insn = Slot::nret(nargs),
                base = std::ptr::null_mut()
            )
        }
        let j = drive(rt, &mut ts, wi - HDR, out);
        jump!(j)
    }

    /// Continuation of the call a `pcall` entry made without a frame of its
    /// own: `true` and the results, to the CALL that made it.
    cont fn ret_pcall {
        let ts = thread!();
        let b = ts.slot_index(base);
        let vi = prepend_true(ts, b, ts.slot_index(values), nret);
        base = ts.slot_ptr(b);
        let (_, cpc) = caller!();
        let call: Instruction = unsafe { *cpc.sub(1) };
        let r = rt.ret(call.c());
        tail!(r, pc = ts.slot_ptr(vi) as *const Instruction, insn = Slot::nret(nret + 1))
    }

    /// As [`ret_pcall`], for `xpcall`, whose frames the unwinder tells apart
    /// by this continuation's address: the handler below the header is
    /// cleared here so the two bodies never fold into one function.
    cont fn ret_xpcall {
        let ts = thread!();
        let b = ts.slot_index(base);
        let nv = unsafe { frame::extras(base) };
        ts.stack[b - HDR - nv - 1] = Value::nil();
        let vi = prepend_true(ts, b, ts.slot_index(values), nret);
        base = ts.slot_ptr(b);
        let (_, cpc) = caller!();
        let call: Instruction = unsafe { *cpc.sub(1) };
        let r = rt.ret(call.c());
        tail!(r, pc = ts.slot_ptr(vi) as *const Instruction, insn = Slot::nret(nret + 1))
    }

    /// The entry of `pcall`: the callee's header overlays the hidden slots of
    /// the CALL, so the callee and its arguments are already in call layout.
    /// Everything the shared `enter` handles (natives, `__call`,
    /// varargs, growth) works through it.
    entry fn ff_pcall {
        let call = insn.as_insn();
        let (a, b) = (call.a(), call.b());
        let ts = thread!();
        let bi = ts.slot_index(base);
        let present = call_nargs(ts, b, bi + a as usize + HDR);
        if call.op() == Op::TAILCALL || present < 1 {
            tail!(native_call)
        }
        let hdr = unsafe { base.add(a as usize + 1) };
        let f: Value<'gc> = reg![a + HDR as u8];
        unsafe { frame::write_hdr(hdr, f.to_raw(), handler_bits(ret_pcall), base, pc) };
        tail!(crate::vm::ops::call::enter, pc = hdr as *const Instruction, insn = Slot::nret(present - 1))
    }

    /// The entry of `xpcall`: the handler moves to the first hidden slot,
    /// where the unwinder finds it below the callee's header.
    entry fn ff_xpcall {
        let call = insn.as_insn();
        let (a, b) = (call.a(), call.b());
        let ts = thread!();
        let bi = ts.slot_index(base);
        let present = call_nargs(ts, b, bi + a as usize + HDR);
        // A missing or non-function handler is the builtin's error.
        if call.op() == Op::TAILCALL || present < 2 || reg![a + 5].get_function().is_none() {
            tail!(native_call)
        }
        let f: Value<'gc> = reg![a + 4];
        reg![a + 1] = reg![a + 5];
        let hdr = unsafe { base.add(a as usize + 2) };
        unsafe { frame::write_hdr(hdr, f.to_raw(), handler_bits(ret_xpcall), base, pc) };
        tail!(crate::vm::ops::call::enter, pc = hdr as *const Instruction, insn = Slot::nret(present - 2))
    }

    /// The entry of `pairs`: for a table without `__pairs`, `(next, t, nil,
    /// nil)` straight into the call's result slots. Other arguments, a
    /// TAILCALL and a call keeping all results go to the builtin.
    entry fn ff_pairs {
        let call = insn.as_insn();
        let (a, b, c) = (call.a(), call.b(), call.c());
        if call.op() == Op::TAILCALL || b != 2 || c == 0 {
            tail!(native_call)
        }
        let Some(t) = reg![a + 4].get_table() else {
            tail!(native_call)
        };
        if t.shape().has_mm(crate::env::MetamethodBits::PAIRS) {
            tail!(native_call)
        }
        reg![a] = Value::function(rt.next_fn());
        let wanted = c as usize - 1;
        if wanted > 1 {
            reg![a + 1] = reg![a + 4];
            unsafe { fill_nil(base.add(a as usize + 2), wanted - 2) };
        }
        set_closure!(unsafe { frame::closure(base) });
        next!()
    }

    /// The entry of `ipairs`: `(iterator, v, 0)` straight into the call's
    /// result slots, for one argument, like [`ff_pairs`].
    entry fn ff_ipairs {
        let call = insn.as_insn();
        let (a, b, c) = (call.a(), call.b(), call.c());
        if call.op() == Op::TAILCALL || b != 2 || c == 0 {
            tail!(native_call)
        }
        let v = reg![a + 4];
        reg![a] = Value::function(rt.ipairs_iter());
        let wanted = c as usize - 1;
        if wanted > 1 {
            reg![a + 1] = v;
        }
        if wanted > 2 {
            reg![a + 2] = Value::small(0);
            unsafe { fill_nil(base.add(a as usize + 3), wanted - 3) };
        }
        set_closure!(unsafe { frame::closure(base) });
        next!()
    }
}

/// Make the native frame at header `hdr` (window `hdr + 4`) wait for the
/// call `out` asks for, at window slot `at` with its arguments after it: the
/// frame's word 0 records `at`, `cont` and `ok`, the arguments move up past
/// the hidden slots (unless the native laid them out there) and the callee's
/// header gets `ret_native`. Returns the callee's header and argument count
/// for `enter`.
#[inline(always)]
fn stage_call<'gc>(
    ts: &mut ThreadState<'gc>,
    hdr: usize,
    nc: &NativeClosure<'gc>,
    out: NativeOut,
) -> (*mut Value<'gc>, usize) {
    let (at, cont, okk) = (out.at(), out.cont(), out.ok());
    let win = hdr + HDR;
    let callee = win + at;
    let nargs = if out.staged() {
        ts.top - (callee + HDR)
    } else {
        // The arguments move up past the hidden slots, highest first.
        let nargs = ts.top - (callee + 1);
        ts.ensure_slots(ts.top + 3);
        let sp = ts.stack.as_mut_ptr();
        unsafe { copy_values_up(sp.add(callee + HDR), sp.add(callee + 1), nargs) };
        ts.top += 3;
        nargs
    };
    let sp = ts.stack.as_mut_ptr();
    let win_ptr = unsafe { sp.add(win) };
    // Single stores throughout: `ret_native` and the callee's RETURN read
    // these words one at a time.
    unsafe {
        let h = sp.add(hdr).cast::<u64>();
        h.write_volatile(NativeHdr { at, cont, ok: okk }.pack(nc));
        let rw = h.add(1).read() & !(flag::PROTECTED | flag::HANDLER);
        h.add(1)
            .write_volatile(rw | flag::NATIVE | out.protect_flags());
        let c = sp.add(callee).cast::<u64>();
        c.add(1).write_volatile(handler_bits(ret_native));
        c.add(2).write_volatile(win_ptr as usize as u64);
        c.add(3).write_volatile(0);
    }
    // Every path here has published the native frame (growth rebases it).
    debug_assert!(std::ptr::eq(ts.top_base, win_ptr));
    (unsafe { sp.add(callee) }, nargs)
}

/// Do what the native of the frame at header `hdr` asked for with `out`.
#[inline(never)]
pub(crate) fn drive<'gc>(
    ctx: Context<'gc>,
    ts: &mut &mut ThreadState<'gc>,
    hdr: usize,
    mut out: NativeOut,
) -> Jump<'gc> {
    loop {
        let t = &mut **ts;
        let win = hdr + HDR;
        let win_ptr = t.slot_ptr(win);
        let nc = unsafe { frame::native_closure(win_ptr) };
        match out.tag() {
            tag::RETURN => {
                if std::hint::unlikely(ctx.gc_due()) {
                    return Jump::Exit(gc_exit(t, win_ptr, nc));
                }
                return land_pending(t, win);
            }
            tag::ERROR => {
                // Raised with the native's frame on top (its level 0); the
                // unwinder pops it, or catches at it when a frameless `pcall`
                // called it. An async native's future dropped itself before
                // returning the error, so the frame no longer owns a task.
                t.top_base = win_ptr;
                let nh = NativeHdr::unpack(unsafe { frame::func_word(win_ptr) });
                if nh.cont == cont::ASYNC.0 {
                    unsafe {
                        frame::set_func_word(
                            win_ptr,
                            NativeHdr {
                                cont: cont::NONE.0,
                                ..nh
                            }
                            .pack(nc),
                        );
                    }
                }
                return crate::vm::unwind::unwind(ctx, ts, out.into_error());
            }
            tag::CALL_THEN => {
                let (hdr, nargs) = stage_call(t, hdr, nc, out);
                return Jump::Enter {
                    hdr,
                    nargs,
                    base: std::ptr::null_mut(),
                };
            }
            tag::RESUME => {
                let at = out.at();
                unsafe {
                    let h = t.slot_ptr(hdr).cast::<u64>();
                    h.write(
                        NativeHdr {
                            at,
                            cont: out.cont(),
                            ok: out.ok(),
                        }
                        .pack(nc),
                    );
                    let rw = h.add(1).read() & !(flag::PROTECTED | flag::HANDLER);
                    h.add(1).write(rw | flag::NATIVE | flag::PROTECTED);
                }
                t.top_base = win_ptr;
                let co = t.stack[win + at]
                    .get_thread()
                    .expect("resume of a non-thread");
                if t.resume_depth + 1 >= crate::vm::coro::MAX_RESUME_DEPTH {
                    // On the resumer, leaving the coroutine untouched
                    // (`lua_resume`'s `resume_error`).
                    let msg = crate::env::LuaString::new(ctx, b"C stack overflow");
                    let err = Error::new(ctx, Value::string(msg));
                    return crate::vm::unwind::unwind(ctx, ts, err);
                }
                return crate::vm::coro::resume_into(ctx, ts, co, win + at + 1);
            }
            tag::YIELD => {
                // The values it resumes with are the native's results.
                unsafe {
                    let h = t.slot_ptr(hdr).cast::<u64>();
                    h.write(
                        NativeHdr {
                            at: 0,
                            cont: 0,
                            ok: ok::RETURN,
                        }
                        .pack(nc),
                    );
                    h.add(1).write(
                        (h.add(1).read() & !(flag::PROTECTED | flag::HANDLER)) | flag::NATIVE,
                    );
                }
                t.top_base = win_ptr;
                return crate::vm::coro::yield_from(ctx, ts, win);
            }
            tag::YIELD_THEN => {
                let at = out.at();
                unsafe {
                    let h = t.slot_ptr(hdr).cast::<u64>();
                    h.write(
                        NativeHdr {
                            at,
                            cont: out.cont(),
                            ok: ok::CONT,
                        }
                        .pack(nc),
                    );
                    h.add(1).write(
                        (h.add(1).read() & !(flag::PROTECTED | flag::HANDLER)) | flag::NATIVE,
                    );
                }
                t.top_base = win_ptr;
                return crate::vm::coro::yield_from(ctx, ts, win + at);
            }
            tag::ASYNC => {
                unsafe {
                    let h = t.slot_ptr(hdr).cast::<u64>();
                    h.write(
                        NativeHdr {
                            at: 0,
                            cont: cont::ASYNC.0,
                            ok: ok::CONT,
                        }
                        .pack(nc),
                    );
                    h.add(1).write(
                        (h.add(1).read() & !(flag::PROTECTED | flag::HANDLER)) | flag::NATIVE,
                    );
                }
                t.top_base = win_ptr;
                out = crate::vm::async_native::async_cont(ctx, nc, Stack::new(t, win), Ok(()));
                if out.tag() != tag::ERROR && t.native_overflowed() {
                    out = NativeOut::error(native_overflow(ctx));
                }
            }
            _ => {
                debug_assert_eq!(out.tag(), tag::PENDING);
                t.top_base = win_ptr;
                return Jump::Exit(Exit::Pending);
            }
        }
    }
}

/// The call the native frame at window `win` waited for returned, its results
/// at `win + at .. top`: pass them through or run the continuation,
/// then do what it asks.
#[inline(never)]
pub(crate) fn native_continue<'gc>(
    ctx: Context<'gc>,
    ts: &mut &mut ThreadState<'gc>,
    win: usize,
) -> Jump<'gc> {
    let t = &mut **ts;
    let win_ptr = t.slot_ptr(win);
    let nh = NativeHdr::unpack(unsafe { frame::func_word(win_ptr) });
    let ret = unsafe { frame::ret(win_ptr) };
    // Past the stack limit the continuation runs after all: it reports the
    // results that don't fit (`resume_cont`, `wrap_cont`).
    let fits = t.top < t.stack_limit;
    match nh.ok {
        ok::RETURN if fits => {
            let nret = t.top - win;
            return Jump::Ret {
                ret,
                nret,
                values: win_ptr,
                base: win_ptr,
            };
        }
        ok::RETURN_TRUE if fits => {
            let at = win + nh.at;
            let nret = t.top - at;
            let vi = prepend_true(t, win, at, nret);
            return Jump::Ret {
                ret,
                nret: nret + 1,
                values: t.slot_ptr(vi),
                base: t.slot_ptr(win),
            };
        }
        _ => {}
    }
    // Errors the continuation itself raises unwind past it.
    unsafe { frame::clear_flags(win_ptr, flag::PROTECTED | flag::HANDLER) };
    let nc = unsafe { frame::native_closure(win_ptr) };
    let mut out = cont_fn(nh.cont)(ctx, nc, Stack::new(t, win), Ok(()));
    if out.tag() != tag::ERROR && t.native_overflowed() {
        out = NativeOut::error(native_overflow(ctx));
    }
    drive(ctx, ts, win - HDR, out)
}

/// An error reached the protected native frame at window `win`: its
/// continuation gets it, with nothing above `at`.
#[inline(never)]
pub(crate) fn native_catch<'gc>(
    ctx: Context<'gc>,
    ts: &mut &mut ThreadState<'gc>,
    win: usize,
    err: Error<'gc>,
) -> Jump<'gc> {
    let t = &mut **ts;
    let win_ptr = t.slot_ptr(win);
    let nh = NativeHdr::unpack(unsafe { frame::func_word(win_ptr) });
    t.set_top_unchecked(win + nh.at);
    t.top_base = win_ptr;
    unsafe { frame::clear_flags(win_ptr, flag::PROTECTED | flag::HANDLER) };
    let nc = unsafe { frame::native_closure(win_ptr) };
    let mut out = cont_fn(nh.cont)(ctx, nc, Stack::new(t, win), Err(err));
    if out.tag() != tag::ERROR && t.native_overflowed() {
        out = NativeOut::error(native_overflow(ctx));
    }
    drive(ctx, ts, win - HDR, out)
}

/// Poll the async native frame at window `win` again, after the host ran.
pub(crate) fn repoll<'gc>(
    ctx: Context<'gc>,
    ts: &mut &mut ThreadState<'gc>,
    win: usize,
) -> Jump<'gc> {
    let t = &mut **ts;
    let win_ptr = t.slot_ptr(win);
    let nc = unsafe { frame::native_closure(win_ptr) };
    let mut out = crate::vm::async_native::async_cont(ctx, nc, Stack::new(t, win), Ok(()));
    if out.tag() != tag::ERROR && t.native_overflowed() {
        out = NativeOut::error(native_overflow(ctx));
    }
    drive(ctx, ts, win - HDR, out)
}
