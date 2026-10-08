//! CALL, TAILCALL, RETURN, the shared `enter` and the CALL continuations.

use crate::env::function::{FunctionKind, LuaFn};
use crate::env::thread::{ThreadState, ThreadStatus};
use crate::env::value::Value;
use crate::instruction::Instruction;
use crate::vm::abi::{Exit, Slot, handler, handler_bits};
use crate::vm::frame::{self, HDR, copy_values, fill_nil, flag, land_results, lua_func_word};
use crate::vm::ops::meta::resolve_call_chain;
use crate::vm::unwind::OpError;

/// Nil-fill the parameters the `nargs` arguments at header `hdr` leave
/// unsupplied and, for a vararg callee, rotate the extras below the header:
/// `[H][p..][e..]` becomes `[e..][H][p..]`. Returns the vararg count,
/// or `Err` when the callee's window would cross the stack limit.
#[inline(never)]
pub(crate) fn fixup_entry<'gc>(
    ts: &mut ThreadState<'gc>,
    hdr: usize,
    nargs: usize,
    callee: LuaFn<'gc>,
) -> Result<usize, ()> {
    let np = callee.num_params as usize;
    let nv = if callee.is_vararg {
        nargs.saturating_sub(np)
    } else {
        0
    };
    let args = hdr + HDR;
    if !ts.ensure_frame_slots(args + nv + callee.max_stack_size as usize) {
        return Err(());
    }
    for i in nargs..np {
        ts.stack[args + i] = Value::nil();
    }
    if nv > 0 {
        ts.stack[hdr..args + np + nv].rotate_left(HDR + np);
    }
    Ok(nv)
}

/// `enter`'s `base`: the calling Lua frame's window, or null for a native
/// caller. As an index, for use across growth.
#[inline(always)]
fn caller_index<'gc>(ts: &ThreadState<'gc>, base: *mut Value<'gc>) -> Option<usize> {
    (!base.is_null()).then(|| ts.slot_index(base))
}

#[inline(always)]
fn caller_ptr<'gc>(ts: &mut ThreadState<'gc>, bi: Option<usize>) -> *mut Value<'gc> {
    bi.map_or(std::ptr::null_mut(), |i| ts.slot_ptr(i))
}

/// Publish the caller of an `enter` path before the stack may grow:
/// the Lua frame `base`, at the pc the staged header holds, or nothing for a
/// native caller, which is published already.
macro_rules! enter_sync {
    ($base:ident, $ts:ident, $hdr:expr) => {
        if !$base.is_null() {
            $ts.top_base = $base;
            $ts.top_pc = unsafe { frame::caller_pc($ts.slot_ptr($hdr + HDR)) };
        }
    };
}

/// After the stack grew under an `enter` path: the staged header is not in
/// the frame chain yet, so its caller word was not rebased; rewrite it from
/// the caller `base`, or the published native frame.
macro_rules! enter_rebase {
    ($base:ident, $ts:ident, $hdr:expr) => {{
        let caller = if $base.is_null() { $ts.top_base } else { $base };
        unsafe { frame::set_caller_base($ts.slot_ptr($hdr + HDR), caller) };
    }};
}

/// Raise from an `enter` path, before the callee's frame exists: the caller's
/// frame is published from the staged header's pc, or already published for
/// a native caller.
macro_rules! enter_raise {
    ($rt:ident, $base:ident, $pc:ident, $ts:ident, $hdr:expr, $bi:expr, $err:expr) => {{
        let hdr: usize = $hdr;
        $base = caller_ptr($ts, $bi);
        $pc = unsafe { frame::caller_pc($ts.slot_ptr(hdr + HDR)) };
        // Staged by a frameless `pcall`: the reference raises with the
        // `pcall` (a C function) running, so without a position, and the
        // `pcall` catches. Its catch point is the staged header, which only
        // the frame chain reaches: make the header a frame of nothing (word 0
        // empty, flagged native) on top, which the unwinder pops or catches at.
        let nb = $ts.slot_ptr(hdr + HDR);
        let marker = unsafe { frame::ret_word(nb) } & !flag::MASK;
        let pcall = marker == handler_bits(crate::vm::native::ret_pcall)
            || marker == handler_bits(crate::vm::native::ret_xpcall);
        if !$base.is_null() {
            $ts.top_base = $base;
            $ts.top_pc = $pc;
        }
        let err = crate::vm::unwind::render($rt, $ts, $err, !pcall);
        if pcall {
            unsafe {
                frame::set_func_word(nb, 0);
                frame::set_flags(nb, flag::NATIVE);
            }
            $ts.top_base = nb;
            $ts.set_top_unchecked(hdr + HDR);
            $base = std::ptr::null_mut();
        }
        throw!(err)
    }};
}

/// The argument count of the call at `nb` (its first argument slot) from a
/// CALL's `b` operand: `b - 1`, or everything up to `top` for the MULTRET
/// sentinel.
#[inline(always)]
fn call_nargs(ts: &ThreadState<'_>, b: u8, nb: usize) -> usize {
    if b == 0 { ts.top - nb } else { b as usize - 1 }
}

/// The Lua arm of CALL, shared by the CALL forms; `$fv` is the callee
/// (`R[a]`, or the source register of a CALLS, which has just stored it
/// there) and `$ret` the callee's continuation.
macro_rules! call_body {
    ($insn:ident, $pc:ident, $base:ident, $rt:ident, $closure:ident, $fv:expr, $ret:expr) => {{
        let (a, b) = ($insn.a(), $insn.b());
        let fv: Value<'gc> = $fv;
        let Some(f) = fv.get_function() else {
            tail!(call_meta)
        };
        match f.inner().as_ref() {
            FunctionKind::Native(nc) => {
                let entry = nc.entry;
                tail!(entry, closure = Slot::native(nc))
            }
            FunctionKind::Lua(p) => {
                let hdr = unsafe { $base.add(a as usize) };
                let nb = unsafe { hdr.add(HDR) };
                if std::hint::unlikely(
                    unsafe { nb.add(p.max_stack_size as usize) }.cast_const() > thread!().stack_end,
                ) {
                    tail!(call_grow)
                }
                if std::hint::unlikely(b as u16 <= p.fixed_arity) {
                    tail!(call_fixup)
                }
                let callee = unsafe { LuaFn::from_function_unchecked(f) };
                unsafe {
                    frame::write_hdr(
                        hdr,
                        lua_func_word(callee, 0),
                        handler_bits($ret),
                        $base,
                        $pc,
                    )
                };
                $closure = callee;
                $base = nb;
                $pc = callee.code;
                next!()
            }
        }
    }};
}

handler! {
    bind(insn, pc, base, rt, closure, thread, nret, values);

    /// `R[a], ...` = `R[a](R[a+4], ...)`; `b` is nargs + 1 (0: up to `top`),
    /// `c` wanted + 1 (0: publish `top`).
    op fn op_call {
        call_body!(insn, pc, base, rt, closure, reg![insn.a()], rt.ret(insn.c()))
    }

    /// CALL wanting no result.
    op fn op_call_r0 {
        call_body!(insn, pc, base, rt, closure, reg![insn.a()], ret_call0)
    }

    /// CALL wanting one result.
    op fn op_call_r1 {
        call_body!(insn, pc, base, rt, closure, reg![insn.a()], ret_call1)
    }

    /// `R[a], ...` = `R[src](R[a+4], ...)`: CALL with the callee in `src`
    /// (the fused MOVE). `R[a]` gets the callee so the slow paths, which
    /// re-read the function slot, see a plain CALL.
    op fn op_calls {
        let fv = reg![insn.imm() as u8];
        reg![insn.a()] = fv;
        call_body!(insn, pc, base, rt, closure, fv, rt.ret(insn.c()))
    }

    /// CALLS wanting no result.
    op fn op_calls_r0 {
        let fv = reg![insn.imm() as u8];
        reg![insn.a()] = fv;
        call_body!(insn, pc, base, rt, closure, fv, ret_call0)
    }

    /// CALLS wanting one result.
    op fn op_calls_r1 {
        let fv = reg![insn.imm() as u8];
        reg![insn.a()] = fv;
        call_body!(insn, pc, base, rt, closure, fv, ret_call1)
    }

    /// CALL of a Lua closure whose window does not fit: grow, then retry
    /// the CALL, which has changed nothing yet.
    slow fn call_grow {
        let call = insn_at!();
        sync!();
        let ts = thread!();
        let bi = ts.slot_index(base);
        let hdr = bi + call.a() as usize;
        let max_stack = match ts.stack[hdr].get_function().map(|f| f.inner().as_ref()) {
            Some(FunctionKind::Lua(p)) => p.max_stack_size as usize,
            _ => unreachable!("call_grow on a non-Lua callee"),
        };
        if !ts.ensure_frame_slots(hdr + HDR + max_stack) {
            raise!(OpError::StackOverflow)
        }
        base = ts.slot_ptr(bi);
        let h = rt.handler(call.opcode());
        tail!(h, insn = Slot::insn(call))
    }

    /// CALL of a Lua closure with missing parameters, varargs or a MULTRET
    /// argument count: the general accounting, then the entry.
    slow fn call_fixup {
        let call = insn_at!();
        let (a, b, c) = (call.a(), call.b(), call.c());
        sync!();
        let ts = thread!();
        let bi = ts.slot_index(base);
        let hdr = bi + a as usize;
        let callee = unsafe { LuaFn::from_function_unchecked(ts.stack[hdr].get_function().unwrap_unchecked()) };
        let nargs = call_nargs(ts, b, hdr + HDR);
        let Ok(nv) = fixup_entry(ts, hdr, nargs, callee) else {
            raise!(OpError::StackOverflow)
        };
        base = ts.slot_ptr(bi);
        let hdr = ts.slot_ptr(hdr + nv);
        unsafe { frame::write_hdr(hdr, lua_func_word(callee, nv), handler_bits(rt.ret(c)), base, pc) };
        set_closure!(callee);
        base = unsafe { hdr.add(HDR) };
        pc = callee.code;
        next!()
    }

    /// CALL of a value that is not a function: resolve its `__call` chain
    /// (each hop inserts the callable as the first argument), then retry.
    slow fn call_meta {
        let call = insn_at!();
        let (a, b) = (call.a(), call.b());
        sync!();
        let ts = thread!();
        let bi = ts.slot_index(base);
        let hdr = bi + a as usize;
        let nargs = call_nargs(ts, b, hdr + HDR);
        let nargs = match resolve_call_chain(rt, ts, hdr, nargs) {
            Ok(n) => n,
            Err(e) => {
                base = ts.slot_ptr(bi);
                raise!(e)
            }
        };
        base = ts.slot_ptr(bi);
        // The chain's hops changed the argument count, which the CALL's `b`
        // cannot carry: enter from a header, which takes the count.
        let hdr = ts.slot_ptr(hdr);
        unsafe {
            let h = hdr.cast::<u64>();
            h.add(1).write(handler_bits(rt.ret(call.c())));
            h.add(2).write(base as usize as u64);
            h.add(3).write(pc as usize as u64);
        }
        tail!(enter, pc = hdr as *const Instruction, insn = Slot::nret(nargs))
    }

    /// Call from a written header: `pc` is the header, whose word 0
    /// holds the callee as a raw value and words 1 to 3 the continuation,
    /// caller base and caller pc; `insn` is the argument count.
    slow fn enter {
        let hdr = pc as *mut Value<'gc>;
        let nargs = insn.as_nret();
        let v: Value<'gc> = unsafe { hdr.read() };
        let Some(f) = v.get_function() else {
            tail!(enter_meta)
        };
        match f.inner().as_ref() {
            FunctionKind::Native(nc) => {
                tail!(crate::vm::native::native_enter, closure = Slot::native(nc))
            }
            FunctionKind::Lua(p) => {
                let callee = unsafe { LuaFn::from_function_unchecked(f) };
                let nb = unsafe { hdr.add(HDR) };
                if std::hint::unlikely(
                    unsafe { nb.add(p.max_stack_size as usize) }.cast_const() > thread!().stack_end,
                ) {
                    tail!(enter_grow)
                }
                if std::hint::unlikely(nargs < p.fixed_arity as usize) {
                    tail!(enter_fixup)
                }
                unsafe { hdr.cast::<u64>().write(lua_func_word(callee, 0)) };
                set_closure!(callee);
                base = nb;
                pc = callee.code;
                next!()
            }
        }
    }

    /// `enter` of a Lua callee whose window does not fit.
    slow fn enter_grow {
        let ts = thread!();
        let hdr = ts.slot_index(pc as *const Value<'gc>);
        enter_sync!(base, ts, hdr);
        let bi = caller_index(ts, base);
        let max_stack = match ts.stack[hdr].get_function().map(|f| f.inner().as_ref()) {
            Some(FunctionKind::Lua(p)) => p.max_stack_size as usize,
            _ => unreachable!("enter_grow on a non-Lua callee"),
        };
        if !ts.ensure_frame_slots(hdr + HDR + max_stack) {
            enter_raise!(rt, base, pc, ts, hdr, bi, OpError::StackOverflow)
        }
        base = caller_ptr(ts, bi);
        enter_rebase!(base, ts, hdr);
        tail!(enter, pc = ts.slot_ptr(hdr) as *const Instruction, insn = insn)
    }

    /// `enter` of a Lua callee with missing parameters or varargs.
    slow fn enter_fixup {
        let nargs = insn.as_nret();
        let ts = thread!();
        let hdr = ts.slot_index(pc as *const Value<'gc>);
        enter_sync!(base, ts, hdr);
        let bi = caller_index(ts, base);
        let callee = unsafe { LuaFn::from_function_unchecked(ts.stack[hdr].get_function().unwrap_unchecked()) };
        let Ok(nv) = fixup_entry(ts, hdr, nargs, callee) else {
            enter_raise!(rt, base, pc, ts, hdr, bi, OpError::StackOverflow)
        };
        base = caller_ptr(ts, bi);
        enter_rebase!(base, ts, hdr + nv);
        let hdr = ts.slot_ptr(hdr + nv);
        unsafe { hdr.cast::<u64>().write(lua_func_word(callee, nv)) };
        set_closure!(callee);
        base = unsafe { hdr.add(HDR) };
        pc = callee.code;
        next!()
    }

    /// `enter` of a value that is not a function.
    slow fn enter_meta {
        let nargs = insn.as_nret();
        let ts = thread!();
        let hdr = ts.slot_index(pc as *const Value<'gc>);
        enter_sync!(base, ts, hdr);
        let bi = caller_index(ts, base);
        let nargs = match resolve_call_chain(rt, ts, hdr, nargs) {
            Ok(n) => n,
            Err(e) => enter_raise!(rt, base, pc, ts, hdr, bi, e),
        };
        base = caller_ptr(ts, bi);
        enter_rebase!(base, ts, hdr);
        tail!(enter, pc = ts.slot_ptr(hdr) as *const Instruction, insn = Slot::nret(nargs))
    }

    /// `return R[a](R[a+4], ...)`. Fast path: a non-vararg Lua callee whose
    /// window fits, from a frame with nothing to close. The callee
    /// takes over this frame's header, so it returns to this frame's caller.
    op fn op_tailcall {
        let (a, b) = insn.ab();
        let fv = reg![a];
        let Some(f) = fv.get_function() else {
            tail!(tailcall_slow)
        };
        match f.inner().as_ref() {
            FunctionKind::Native(nc) => {
                let entry = nc.entry;
                tail!(entry, closure = Slot::native(nc))
            }
            FunctionKind::Lua(p) => {
                let ts = thread!();
                let flags = unsafe { frame::flags(base) };
                let fits = unsafe { base.add(p.max_stack_size as usize) }.cast_const() <= ts.stack_end;
                if std::hint::likely(flags & (flag::HAS_OPEN | flag::HAS_TBC) == 0 && !p.is_vararg && fits) {
                    let callee = unsafe { LuaFn::from_function_unchecked(f) };
                    let src = unsafe { base.add(a as usize + HDR) };
                    let nargs = call_nargs(ts, b, ts.slot_index(src));
                    let np = callee.num_params as usize;
                    unsafe {
                        copy_values(base, src, nargs);
                        if nargs < np {
                            fill_nil(base.add(nargs), np - nargs);
                        }
                        // The frame keeps its vararg count: its results
                        // still land where its caller expects.
                        let w = frame::func_word(base) & !frame::PTR_MASK;
                        frame::set_func_word(base, w | callee.as_ptr() as usize as u64);
                    }
                    closure = callee;
                    pc = callee.code;
                    next!()
                }
                tail!(tailcall_slow)
            }
        }
    }

    /// TAILCALL through a `__call` chain, from a frame with upvalues to
    /// close, into a callee whose window must be grown, or into a vararg
    /// callee, whose header moves to leave room for its extras.
    slow fn tailcall_slow {
        let call = insn_at!();
        let (a, b) = call.ab();
        sync!();
        let ts = thread!();
        let bi = ts.slot_index(base);
        let hdr = bi + a as usize;
        let nargs = call_nargs(ts, b, hdr + HDR);
        let nargs = match resolve_call_chain(rt, ts, hdr, nargs) {
            Ok(n) => n,
            Err(e) => {
                base = ts.slot_ptr(bi);
                raise!(e)
            }
        };
        base = ts.slot_ptr(bi);
        let f = unsafe { ts.stack[hdr].get_function().unwrap_unchecked() };
        let FunctionKind::Lua(_) = f.inner().as_ref() else {
            // The chain left the native in the function slot and its count
            // in `top`.
            let nc = unsafe { f.as_native().unwrap_unchecked() };
            let entry = nc.entry;
            let mut call = call;
            call.set_b(0);
            tail!(entry, insn = Slot::insn(call), closure = Slot::native(nc))
        };
        let callee = unsafe { LuaFn::from_function_unchecked(f) };
        // The results go to this frame's function slot in its caller, below
        // any extras.
        let nv_old = unsafe { frame::nv(base) };
        let f_slot = bi - HDR - nv_old;
        // No TAILCALL is emitted inside a `<close>` scope, so `HAS_TBC` only
        // marks scopes that already closed.
        debug_assert!(!crate::vm::ops::control::has_tbc_from(ts, bi));
        // Before the arguments overwrite these slots, so an open upvalue
        // keeps the local, not the argument.
        crate::vm::ops::control::close_upvalues(rt.mutation(), ts, bi);
        // Room first, while the header the rebase walks through is intact.
        let np = callee.num_params as usize;
        let nv = if callee.is_vararg { nargs.saturating_sub(np) } else { 0 };
        if !ts.ensure_frame_slots(f_slot + HDR + nv + callee.max_stack_size as usize) {
            raise!(OpError::StackOverflow)
        }
        base = ts.slot_ptr(bi);
        let (rword, caller, cpc) = unsafe { (frame::ret_word(base), frame::caller_base(base), frame::caller_pc(base)) };
        // Arguments down to where a fixed callee's would be; the copy is
        // ascending, so the overlap is safe.
        let src = hdr + HDR;
        ts.stack.copy_within(src..src + nargs, f_slot + HDR);
        let Ok(nv) = fixup_entry(ts, f_slot, nargs, callee) else {
            unreachable!("the window was grown above")
        };
        let hdr = ts.slot_ptr(f_slot + nv);
        unsafe {
            frame::write_hdr(hdr, lua_func_word(callee, nv), rword & !(flag::HAS_OPEN | flag::HAS_TBC), caller, cpc)
        };
        set_closure!(callee);
        base = unsafe { hdr.add(HDR) };
        pc = callee.code;
        next!()
    }

    /// `return R[a], ... R[a+b-2]` (`b` 0: up to `top`). Nothing to close on
    /// the fast path; `return_close` otherwise.
    op fn op_return {
        let (a, b) = insn.ab();
        if std::hint::unlikely(unsafe { frame::flags(base) } != 0) {
            tail!(return_close)
        }
        let vals = unsafe { base.add(a as usize) };
        let ts = thread!();
        let n = call_nargs(ts, b, ts.slot_index(vals));
        let r = unsafe { frame::ret(base) };
        tail!(r, pc = vals as *const Instruction, insn = Slot::nret(n))
    }

    /// `return`, from a function the assembler found has nothing to close.
    op fn op_return0 {
        debug_assert_eq!(unsafe { frame::flags(base) }, 0);
        let r = unsafe { frame::ret(base) };
        tail!(r, pc = base as *const Instruction, insn = Slot::nret(0))
    }

    /// `return R[a]`, as RETURN0.
    op fn op_return1 {
        debug_assert_eq!(unsafe { frame::flags(base) }, 0);
        let r = unsafe { frame::ret(base) };
        tail!(r, pc = unsafe { base.add(insn.a() as usize) } as *const Instruction, insn = Slot::nret(1))
    }

    /// RETURN from a frame with upvalues or to-be-closed variables to close:
    /// one `__close` call at a time through `ret_return`, then the return.
    slow fn return_close {
        let ret_insn = insn_at!();
        let (a, b) = ret_insn.ab();
        sync!();
        let ts = thread!();
        let bi = ts.slot_index(base);
        let vals = bi + a as usize;
        let n = call_nargs(ts, b, vals);
        if crate::vm::ops::control::has_tbc_from(ts, bi) {
            let (v, tm) = match crate::vm::ops::control::pop_tbc(rt, ts) {
                Ok(c) => c,
                Err(err) => throw!(err),
            };
            // Past the frame and the results, below the call: where they
            // end, which the RETURN finds through `top` again.
            let max_stack = unsafe { frame::closure(base) }.max_stack_size as usize;
            let mark = (vals + n).max(bi + max_stack);
            let hdr = mark + 1;
            ts.ensure_slots(hdr + HDR + 1);
            ts.stack[mark] = Value::small((vals + n) as i32);
            base = ts.slot_ptr(bi);
            let hdr = ts.slot_ptr(hdr);
            unsafe {
                frame::write_hdr(hdr, tm.to_raw(), handler_bits(crate::vm::ops::meta::ret_return), base, pc);
                hdr.add(HDR).write(v);
            }
            tail!(enter, pc = hdr as *const Instruction, insn = Slot::nret(1))
        }
        crate::vm::ops::control::close_upvalues(rt.mutation(), ts, bi);
        let r = unsafe { frame::ret(base) };
        tail!(r, pc = ts.slot_ptr(vals) as *const Instruction, insn = Slot::nret(n))
    }

    /// Continuation of a CALL: land as many results as its `c` asks for.
    cont fn ret_call {
        let (caller, cpc) = caller!();
        let call: Instruction = unsafe { *cpc.sub(1) };
        let c = call.c();
        let dst = unsafe { caller.add(call.a() as usize) };
        let wanted = if c == 0 { nret } else { c as usize - 1 };
        unsafe { land_results(dst, values, nret, wanted) };
        if c == 0 {
            let ts = thread!();
            ts.top = ts.slot_index(dst) + wanted;
        }
        resume!(caller, cpc)
    }

    /// Continuation of a CALL wanting no result.
    cont fn ret_call0 {
        let (caller, cpc) = caller!();
        resume!(caller, cpc)
    }

    /// Continuation of a CALL wanting one result.
    cont fn ret_call1 {
        let (caller, cpc) = caller!();
        let call: Instruction = unsafe { *cpc.sub(1) };
        let v = if nret > 0 { unsafe { values.read() } } else { Value::nil() };
        unsafe { caller.add(call.a() as usize).write(v) };
        resume!(caller, cpc)
    }

    /// Continuation of a CALL wanting two results.
    cont fn ret_call2 {
        let (caller, cpc) = caller!();
        let call: Instruction = unsafe { *cpc.sub(1) };
        let dst = unsafe { caller.add(call.a() as usize) };
        let v0 = if nret > 0 { unsafe { values.read() } } else { Value::nil() };
        let v1 = if nret > 1 { unsafe { values.add(1).read() } } else { Value::nil() };
        unsafe {
            dst.write(v0);
            dst.add(1).write(v1);
        }
        resume!(caller, cpc)
    }

    /// Continuation of a thread's bottom frame: the results go to
    /// slot 0 and the thread is done.
    cont fn ret_exit {
        let ts = thread!();
        let sp = ts.stack.as_mut_ptr();
        unsafe { copy_values(sp, values, nret) };
        ts.top_base = std::ptr::null_mut();
        ts.top_pc = std::ptr::null();
        ts.set_top_unchecked(nret);
        ts.discard_above(nret);
        ts.status = ThreadStatus::Result { bottom: 0 };
        return Exit::End;
    }
}

/// Start the seeded call of a thread that has not run yet: header 0
/// gets `ret` as its continuation; `[f, _, _, _, args..]` is already there.
pub(crate) fn seed_header<'gc>(ts: &mut ThreadState<'gc>, ret: crate::vm::abi::Handler) -> usize {
    debug_assert!(!ts.started && ts.top >= HDR);
    ts.started = true;
    let hdr = ts.slot_ptr(0);
    unsafe {
        let p = hdr.cast::<u64>();
        p.add(1).write(handler_bits(ret));
        p.add(2).write(0);
        p.add(3).write(0);
    }
    ts.top - HDR
}
