//! Jumps, loops, closures, varargs, to-be-closed variables and the
//! miscellaneous opcodes.

use crate::dmm::Mutation;
use crate::env::MetamethodBits;
use crate::env::error::Error;
use crate::env::function::{Function, LuaFn, UpvalueCell, UpvalueSlot};
use crate::env::table::{Step, Table};
use crate::env::thread::{TbcEntry, ThreadState};
use crate::env::value::Value;
use crate::instruction::{Instruction, Op, TFOR_VARS, UpvalSource};
use crate::lua::Context;
use crate::vm::abi::{Exit, handler};
use crate::vm::frame::{self, HDR, flag, land_results};
use crate::vm::num;
use crate::vm::ops::count_hot;
use crate::vm::ops::meta::{call_chain_error, ret_close, ret_tfor};
use crate::vm::unwind::OpError;

/// The position slot's mark for an `ipairs` loop (see `op_tforprep`).
const TFOR_IPAIRS: i32 = -1;

/// Whether a to-be-closed variable is registered at or above `level`.
#[inline]
pub(crate) fn has_tbc_from(ts: &ThreadState<'_>, level: usize) -> bool {
    ts.tbc_list.last().is_some_and(|e| e.pos() >= level)
}

/// Whether any open upvalue points at stack index `level` or above.
#[inline(always)]
fn has_open_from(ts: &ThreadState<'_>, level: usize) -> bool {
    ts.open_upvalues
        .last()
        .is_some_and(|uv| uv.slot().addr() >= ts.stack.as_ptr().wrapping_add(level).addr())
}

/// Close every open upvalue pointing at stack index `level` or above.
/// Inlined so the usual nothing-to-close case is a load or two, not a call.
#[inline(always)]
pub(crate) fn close_upvalues<'gc>(mc: &Mutation<'gc>, ts: &mut ThreadState<'gc>, level: usize) {
    if has_open_from(ts, level) {
        close_upvalues_slow(mc, ts, level);
    }
}

#[inline(never)]
fn close_upvalues_slow<'gc>(mc: &Mutation<'gc>, ts: &mut ThreadState<'gc>, level: usize) {
    let level = ts.stack.as_ptr().wrapping_add(level).addr();
    // Sorted by slot, so the ones to close are the tail.
    while let Some(&uv) = ts.open_upvalues.last() {
        if uv.slot().addr() < level {
            break;
        }
        UpvalueCell::close(uv, mc);
        ts.open_upvalues.pop();
    }
}

/// Take the innermost to-be-closed variable off the list: its value and
/// `__close`, or the error for a `__close` that can't be called, raised as
/// the closing frame's rather than as the call's.
pub(crate) fn pop_tbc<'gc>(
    ctx: Context<'gc>,
    ts: &mut ThreadState<'gc>,
) -> Result<(Value<'gc>, Value<'gc>), Error<'gc>> {
    let entry = ts
        .tbc_list
        .pop()
        .expect("no to-be-closed variable to close");
    let v = entry.value(&ts.stack);
    let tm = ctx.mm_of(v, MetamethodBits::CLOSE);
    if let Some(e) = call_chain_error(ctx, tm) {
        let suffix = match e {
            OpError::Call(_) => " (metamethod 'close')",
            _ => "",
        };
        let msg = crate::vm::debug::op_error_message(ctx, ts, e) + suffix;
        return Err(crate::vm::debug::error_at(ctx, ts, &msg, 0));
    }
    Ok((v, tm))
}

/// Lua's `forlimit`: the limit of an integer loop, floored for an ascending
/// loop and ceiled for a descending one. `Some(None)` means the loop must not
/// run; `None` is a limit that isn't a number at all.
fn for_limit(init: i64, limit: Value, step: i64) -> Option<Option<i64>> {
    use crate::builtin::util::Number;
    let n = match limit.get_string() {
        Some(s) => crate::builtin::util::str_to_number(s.as_bytes())?,
        None => match (limit.get_integer(), limit.get_float()) {
            (Some(i), _) => Number::Int(i),
            (_, Some(f)) => Number::Float(f),
            _ => return None,
        },
    };
    let lim = match n {
        Number::Int(i) => i,
        Number::Float(f) => {
            let rounded = if step < 0 { f.ceil() } else { f.floor() };
            match num::exact_float_to_int(rounded) {
                Some(i) => i,
                // Out of range, infinite, or NaN. NaN fails `0 < f` and so is
                // treated as too negative, like the reference.
                None if 0.0 < f => {
                    if step < 0 {
                        return Some(None);
                    }
                    i64::MAX
                }
                None => {
                    if step > 0 {
                        return Some(None);
                    }
                    i64::MIN
                }
            }
        }
    };
    let runs = if step > 0 { init <= lim } else { init >= lim };
    Some(runs.then_some(lim))
}

/// Finish a TFORCALL step taken without calling the iterator: store `(k, v)`
/// in the loop's variables and take the following TFORLOOP's jump, or once
/// the walk is done, nil them and step past it.
macro_rules! tfor_finish {
    ($pc:ident, $base:ident, $step:expr, $vars:expr, $count:expr) => {{
        let vars: u8 = $vars;
        let count: u8 = $count;
        match $step {
            Some((k, v)) => {
                reg![vars] = k;
                if count > 1 {
                    reg![vars + 1] = v;
                }
                for i in 2..count {
                    reg![vars + i] = Value::nil();
                }
                debug_assert_eq!(unsafe { (*$pc).op() }, Op::TFORLOOP);
                let (_, offset) = unsafe { *$pc }.a_offset();
                $pc = unsafe { $pc.add(1).offset(offset as isize) };
            }
            None => {
                for i in 0..count {
                    reg![vars + i] = Value::nil();
                }
                $pc = unsafe { $pc.add(1) };
            }
        }
        next!()
    }};
}

/// Rewrite the FORLOOP a FORPREP at `pc - 1` guards (the instruction before
/// its jump target) to `form` when it is not that already. Only on a
/// change: a store to a word dispatch loads a few instructions later stalls
/// every loop start (queen 1.06).
#[inline(always)]
fn write_forloop_form(pc: *const Instruction, offset: i32, form: Op) {
    // SAFETY: `Code` keeps instructions in cells; the FORPREP's target is
    // the instruction after its FORLOOP.
    unsafe {
        let site = pc.offset(offset as isize - 1).cast_mut();
        let word = *site;
        // A compiled loop entry guards the loop kind itself.
        if word.is_jit_word() {
            return;
        }
        debug_assert_eq!(word.generic_op(), Op::FORLOOP);
        if word.op() != form {
            site.write(word.with_op(form));
        }
    }
}

handler! {
    bind(insn, pc, base, rt, closure, thread, nret, values);

    /// `pc += imm`
    op fn op_jmp {
        jump_by!(insn.branch_offset());
        next!()
    }

    /// Prepare a numeric for. Before: `R[a] = init, R[a+1] = limit, R[a+2] =
    /// step`. After: `R[a]` = the last value the control variable takes
    /// (integer loop) or the limit (float loop), `R[a+1]` = step, `R[a+2]` =
    /// the hidden control variable, `R[a+3]` = its visible copy. Jumps past
    /// the body if the loop won't run.
    ///
    /// The store-only visible copy keeps the body off the control slot's
    /// loop-carried chain; comparing against a precomputed last value
    /// keeps the chain to one slot.
    op fn op_forprep {
        let (a, offset) = insn.a_offset();
        let init = reg![a];
        let limit = reg![a + 1];
        let step = reg![a + 2];
        let mc = rt.mutation();
        let skip = if let (Some(i), Some(s)) = (init.get_integer(), step.get_integer()) {
            if s == 0 {
                raise!(OpError::ForStepZero)
            }
            match for_limit(i, limit, s) {
                None => raise!(OpError::ForNotNumber("limit", limit)),
                Some(None) => true,
                Some(Some(lim)) => {
                    // The reference's unsigned iteration count, turned back
                    // into the final value of the control variable: `last`
                    // lies in [init, lim], so the wrapping arithmetic is exact
                    // and FORLOOP can stop on equality with no overflow check.
                    let count = if s > 0 {
                        let span = (lim as u64).wrapping_sub(i as u64);
                        if s == 1 { span } else { span / s as u64 }
                    } else {
                        // `-(s + 1) + 1` avoids negating `i64::MIN`.
                        (i as u64).wrapping_sub(lim as u64) / ((-(s + 1)) as u64 + 1)
                    };
                    let last = (i as u64).wrapping_add(count.wrapping_mul(s as u64)) as i64;
                    reg![a] = Value::integer(mc, last);
                    reg![a + 1] = Value::integer(mc, s);
                    reg![a + 2] = Value::integer(mc, i);
                    reg![a + 3] = Value::integer(mc, i);
                    // The loop's FORLOOP takes the integer form when every
                    // value it compares is small.
                    let all_small = i32::try_from(last).is_ok()
                        && i32::try_from(s).is_ok()
                        && i32::try_from(i).is_ok();
                    write_forloop_form(pc, offset, if all_small { Op::FORLOOP_I } else { Op::FORLOOP });
                    if !all_small {
                        // A generic FORLOOP whose byte is empty never ran.
                        let site = unsafe { pc.offset(offset as isize - 1) };
                        crate::jit::feedback::record(closure, site, crate::jit::feedback::BIGINT);
                    }
                    false
                }
            }
        } else {
            // Same coercion and check order as the reference `forprep`.
            use crate::builtin::util::to_number;
            let Some(lim) = to_number(limit) else {
                raise!(OpError::ForNotNumber("limit", limit))
            };
            let Some(s) = to_number(step) else {
                raise!(OpError::ForNotNumber("step", step))
            };
            let Some(i) = to_number(init) else {
                raise!(OpError::ForNotNumber("initial value", init))
            };
            if s == 0.0 {
                raise!(OpError::ForStepZero)
            }
            let skip = if 0.0 < s { lim < i } else { i < lim };
            if !skip {
                reg![a] = Value::float(lim);
                reg![a + 1] = Value::float(s);
                reg![a + 2] = Value::float(i);
                reg![a + 3] = Value::float(i);
                write_forloop_form(pc, offset, Op::FORLOOP_F);
            }
            skip
        };
        branch!(skip, offset)
    }

    /// `FORLOOP_I`: the integer loop whose last value, step and control
    /// variable are all small; anything else (`debug.setlocal` on the
    /// hidden slots) goes to the generic FORLOOP. The FORLOOP forms count
    /// a taken back edge, before writing anything.
    op fn op_forloop_i {
        let (a, offset) = insn.a_offset();
        let step = &reg![a + 1];
        let Some(s) = step.get_small() else {
            tail!(op_forloop)
        };
        let Some((last, idx)) = Value::both_small(&reg![a], &reg![a + 2]) else {
            tail!(op_forloop)
        };
        let go = idx != last;
        if go {
            count_hot!(rt, insn, b);
            let idx = Value::small(idx.wrapping_add(s));
            reg![a + 2] = idx;
            reg![a + 3] = idx;
        }
        branch!(go, offset)
    }

    /// `FORLOOP_F`: the float loop.
    op fn op_forloop_f {
        let (a, offset) = insn.a_offset();
        let step = &reg![a + 1];
        if !step.is_float() {
            tail!(op_forloop)
        }
        let s = step.read_float();
        let lim = reg![a].read_float();
        let idx = reg![a + 2].read_float() + s;
        let go = if 0.0 < s { idx <= lim } else { lim <= idx };
        if go {
            count_hot!(rt, insn, b);
            let idx = Value::float(idx);
            reg![a + 2] = idx;
            reg![a + 3] = idx;
        }
        branch!(go, offset)
    }

    /// Numeric for step: advance the control variable and jump back while
    /// iterations remain, reading the layout FORPREP leaves behind. The
    /// generic form, for a loop with boxed values and for the forms' misses,
    /// which have not counted yet.
    op fn op_forloop {
        let (a, offset) = insn.a_offset();
        // The step's type tells the loop kind, and the hidden slots match it:
        // FORPREP wrote them and nothing else can (the visible copy is never
        // read here).
        // Read in place: a copy would be spilled for the volatile float load.
        let step = &reg![a + 1];
        if let Some(s) = step.get_small()
            && let Some((last, idx)) = Value::both_small(&reg![a], &reg![a + 2])
        {
            // `idx` walks init, init+step, ..., last exactly, so `idx != last`
            // also guarantees `idx + step` stays in range.
            let go = idx != last;
            if go {
                count_hot!(rt, insn, b);
                let idx = Value::small(idx.wrapping_add(s));
                reg![a + 2] = idx;
                reg![a + 3] = idx;
            }
            branch!(go, offset)
        } else if step.is_float() {
            let s = step.read_float();
            let lim = reg![a].read_float();
            let idx = reg![a + 2].read_float() + s;
            let go = if 0.0 < s { idx <= lim } else { lim <= idx };
            if go {
                count_hot!(rt, insn, b);
                let idx = Value::float(idx);
                reg![a + 2] = idx;
                reg![a + 3] = idx;
            }
            branch!(go, offset)
        }
        // Boxing the new index allocates, which would give this handler a
        // stack frame.
        tail!(forloop_slow)
    }

    /// FORLOOP for an integer loop whose values don't all fit a small int.
    slow fn forloop_slow {
        // `op_forloop`'s word: the one in `Code` may be a `JIT_LOOP`.
        let insn = insn.as_insn();
        let (a, offset) = insn.a_offset();
        let ts_closure = unsafe { frame::closure(base) };
        crate::jit::feedback::record(ts_closure, unsafe { pc.sub(1) }, crate::jit::feedback::BIGINT);
        let (s, last, idx) = (reg![a + 1], reg![a], reg![a + 2]);
        let (s, last, idx) = unsafe {
            (
                s.get_integer().unwrap_unchecked(),
                last.get_integer().unwrap_unchecked(),
                idx.get_integer().unwrap_unchecked(),
            )
        };
        let go = idx != last;
        if go {
            count_hot!(rt, insn, b);
            let idx = Value::integer(rt.mutation(), idx.wrapping_add(s));
            reg![a + 2] = idx;
            reg![a + 3] = idx;
        }
        branch!(go, offset)
    }

    /// Generic for preparation: move the closing value from `R[a+3]` to
    /// `R[a+2]`, mark it to be closed, move the initial control from `R[a+2]`
    /// to the first loop variable, set the position slot `R[a+3]` (see
    /// `op_tforcall`), and jump to the loop test.
    op fn op_tforprep {
        let (a, offset) = insn.a_offset();
        let control = reg![a + 2];
        let closing = reg![a + 3];
        reg![a + TFOR_VARS] = control;
        reg![a + 2] = closing;
        // Decided once per loop, as LuaJIT's `ISNEXT` does: the iterator and
        // state are hidden, so only `debug.setlocal` can change them.
        let (iter, state) = (reg![a], reg![a + 1]);
        let (pos, form) = if state.get_table().is_none() {
            (Value::nil(), Op::TFORCALL)
        } else if iter.same_bits(&Value::function(rt.next_fn())) && control.is_nil() {
            (Value::small(0), Op::TFORCALL_NEXT)
        } else if iter.same_bits(&Value::function(rt.ipairs_iter())) {
            (Value::small(TFOR_IPAIRS), Op::TFORCALL_IPAIRS)
        } else {
            (Value::nil(), Op::TFORCALL)
        };
        reg![a + 3] = pos;
        // The TFORCALL this jumps to takes the matching form, written
        // only on a change: a store to a word dispatch is about to load
        // stalls every loop start.
        unsafe {
            let site = pc.offset(offset as isize).cast_mut();
            let word = *site;
            debug_assert_eq!(word.generic_op(), Op::TFORCALL);
            if word.op() != form {
                site.write(word.with_op(form));
            }
        }
        if !closing.is_falsy() {
            if rt.mm_of(closing, MetamethodBits::CLOSE).is_nil() {
                raise!(OpError::NonClosable(a + 2))
            }
            let ts = thread!();
            let bi = ts.slot_index(base);
            ts.tbc_list.push(TbcEntry::Slot(bi + a as usize + 2));
            unsafe { frame::set_flags(base, flag::HAS_TBC) };
        }
        jump_by!(offset);
        next!()
    }

    /// `TFORCALL_NEXT`: the `next` step over a table, kept at the position
    /// `R[a+3]`; anything else goes to the generic TFORCALL.
    op fn op_tforcall_next {
        let (a, count) = insn.ab();
        let vars = a + TFOR_VARS;
        let Some(t) = reg![a + 1].get_table() else {
            tail!(op_tforcall)
        };
        let Some(pos) = reg![a + 3].get_small().filter(|&p| p >= 0) else {
            tail!(op_tforcall)
        };
        let step = match t.inner().borrow().next_at_inline(pos as u32) {
            Step::Entry(next, k, v) => {
                reg![a + 3] = Value::small(next as i32);
                Some((k, v))
            }
            Step::End => None,
            Step::Slow => tail!(tfor_next),
        };
        tfor_finish!(pc, base, step, vars, count)
    }

    /// `TFORCALL_IPAIRS`: the `ipairs` step through the array part.
    op fn op_tforcall_ipairs {
        let (a, count) = insn.ab();
        let vars = a + TFOR_VARS;
        let Some(t) = reg![a + 1].get_table() else {
            tail!(op_tforcall)
        };
        if reg![a + 3].get_small() != Some(TFOR_IPAIRS) {
            tail!(op_tforcall)
        }
        let Some(i) = reg![vars].get_small() else {
            tail!(op_tforcall)
        };
        let state = t.inner().borrow();
        let step = match i.checked_add(1) {
            Some(k)
                if let Some(v) = state.array_get(k as usize)
                    && !v.is_nil() =>
            {
                Some((Value::small(k), v))
            }
            _ => {
                drop(state);
                tail!(tfor_ipairs)
            }
        };
        drop(state);
        tfor_finish!(pc, base, step, vars, count)
    }

    /// Generic for call: the loop variables = `R[a](R[a+1], first variable)`.
    ///
    /// When TFORPREP found the iterator to be the `next` `pairs` returns, or
    /// `ipairs`'s, and the state a table, it set the position slot `R[a+3]`,
    /// and the step is taken without a call, and so is the following
    /// TFORLOOP's jump. `next`'s walk keeps its position in `R[a+3]`, so a
    /// step doesn't look up the previous key; like LuaJIT's `ITERN`, it then
    /// doesn't follow a key or iterator `debug.setlocal` changes.
    op fn op_tforcall {
        let (a, count) = insn.ab();
        let vars = a + TFOR_VARS;
        let Some(t) = reg![a + 1].get_table() else {
            tail!(tforcall_generic)
        };
        let step = match reg![a + 3].get_small() {
            Some(TFOR_IPAIRS) if let Some(i) = reg![vars].get_small() => {
                let state = t.inner().borrow();
                match i.checked_add(1) {
                    Some(k)
                        if let Some(v) = state.array_get(k as usize)
                            && !v.is_nil() =>
                    {
                        Some((Value::small(k), v))
                    }
                    _ => {
                        drop(state);
                        tail!(tfor_ipairs)
                    }
                }
            }
            Some(pos) if pos >= 0 => match t.inner().borrow().next_at_inline(pos as u32) {
                Step::Entry(next, k, v) => {
                    reg![a + 3] = Value::small(next as i32);
                    Some((k, v))
                }
                Step::End => None,
                Step::Slow => tail!(tfor_next),
            },
            _ => tail!(tforcall_generic),
        };
        tfor_finish!(pc, base, step, vars, count)
    }

    /// TFORCALL's `next` step through a part of integer or other keys.
    slow fn tfor_next {
        let insn = insn_at!();
        let (a, count) = insn.ab();
        let vars = a + TFOR_VARS;
        // `op_tforcall` checked both.
        let (t, pos) = (reg![a + 1].get_table(), reg![a + 3].get_small());
        let (t, pos) = unsafe { (t.unwrap_unchecked(), pos.unwrap_unchecked() as u32) };
        let step = t.inner().borrow().next_at(rt.mutation(), pos);
        // Past a position that doesn't fit, `next` resumes from the key.
        let step = step.map(|(next, k, v)| {
            reg![a + 3] = next.map_or(Value::nil(), |p| Value::small(p as i32));
            (k, v)
        });
        tfor_finish!(pc, base, step, vars, count)
    }

    /// TFORCALL's `ipairs` step past the array part: through the integer
    /// keys' hash part, to the end, or to `__index`.
    slow fn tfor_ipairs {
        let insn = insn_at!();
        let (a, count) = insn.ab();
        let vars = a + TFOR_VARS;
        let t = unsafe { reg![a + 1].get_table().unwrap_unchecked() };
        let i = unsafe { reg![vars].get_small().unwrap_unchecked() } as i64 + 1;
        let state = t.inner().borrow();
        let v = state.get_int(i);
        let step = if !v.is_nil() {
            Some((Value::integer(rt.mutation(), i), v))
        } else if state.shape().has_mm(MetamethodBits::INDEX) {
            drop(state);
            tail!(tforcall_generic)
        } else {
            None
        };
        tfor_finish!(pc, base, step, vars, count)
    }

    /// TFORCALL by calling the iterator.
    slow fn tforcall_generic {
        let insn = insn_at!();
        let a = insn.a();
        let iter = reg![a];
        let state = reg![a + 1];
        let control = reg![a + TFOR_VARS];
        let closure: LuaFn<'gc> = unsafe { frame::closure(base) };
        let hdr = unsafe { base.add(closure.max_stack_size as usize) };
        let ts = thread!();
        if unsafe { hdr.add(HDR + 2) }.cast_const() > ts.stack_end {
            tail!(crate::vm::ops::meta::stage_grow)
        }
        unsafe {
            frame::write_hdr(hdr, iter.to_raw(), crate::vm::abi::handler_bits(ret_tfor), base, pc);
            hdr.add(HDR).write(state);
            hdr.add(HDR + 1).write(control);
        }
        tail!(crate::vm::ops::call::enter, pc = hdr as *const Instruction, insn = crate::vm::abi::Slot::nret(2))
    }

    /// Generic for loop test: jump back while the first variable is not nil.
    op fn op_tforloop {
        let (a, offset) = insn.a_offset();
        branch!(!reg![a + TFOR_VARS].is_nil(), offset)
    }

    /// Close all upvalues and to-be-closed variables from `R[a]`.
    op fn op_close {
        let ts = thread!();
        let level = ts.slot_index(base) + insn.a() as usize;
        close_upvalues(rt.mutation(), ts, level);
        if has_tbc_from(ts, level) {
            tail!(close_tbc)
        }
        next!()
    }

    /// CLOSE of a to-be-closed variable: call its `__close`, taking it off the
    /// list first so a failing one leaves the rest to the unwinder, then run
    /// the CLOSE again for the next.
    slow fn close_tbc {
        sync!();
        // Room before the entry leaves the list: `stage_grow` runs the CLOSE
        // again, which must still find it.
        let closure: LuaFn<'gc> = unsafe { frame::closure(base) };
        let hdr = unsafe { base.add(closure.max_stack_size as usize) };
        if unsafe { hdr.add(HDR + 1) }.cast_const() > thread!().stack_end {
            tail!(crate::vm::ops::meta::stage_grow)
        }
        let (v, tm) = match pop_tbc(rt, thread!()) {
            Ok(c) => c,
            Err(err) => throw!(err),
        };
        unsafe {
            frame::write_hdr(hdr, tm.to_raw(), crate::vm::abi::handler_bits(ret_close), base, pc);
            hdr.add(HDR).write(v);
        }
        tail!(crate::vm::ops::call::enter, pc = hdr as *const Instruction, insn = crate::vm::abi::Slot::nret(1))
    }

    /// Mark `R[a]` as to-be-closed.
    op fn op_tbc {
        let a = insn.a();
        let v = reg![a];
        // `false` and `nil` need no closing.
        if v.is_falsy() {
            next!()
        }
        if rt.mm_of(v, MetamethodBits::CLOSE).is_nil() {
            raise!(OpError::NonClosable(a))
        }
        let ts = thread!();
        let bi = ts.slot_index(base);
        ts.tbc_list.push(TbcEntry::Slot(bi + a as usize));
        unsafe { frame::set_flags(base, flag::HAS_TBC) };
        next!()
    }

    /// `R[a] = closure(proto[d])`
    op fn op_closure {
        let (dst, proto_idx) = insn.ad();
        let proto = closure.proto.prototypes[proto_idx as usize];
        let ts = thread!();
        let mut self_slot = None;
        let mut captured = false;
        let func = Function::new_lua(rt.mutation(), proto, |slots| {
            for (i, desc) in proto.upvalue_desc.iter().enumerate() {
                let slot = match desc.source {
                    // A local function's capture of itself: its register is
                    // only written below.
                    UpvalSource::ParentLocal(idx) if desc.by_value && idx == dst => {
                        self_slot = Some(i);
                        UpvalueSlot { value: Value::nil() }
                    }
                    UpvalSource::ParentLocal(idx) if desc.by_value => UpvalueSlot { value: reg![idx] },
                    UpvalSource::ParentLocal(idx) => {
                        let slot = unsafe { base.add(idx as usize) };
                        // Sorted by slot, so this frame's are at the end.
                        let open = &ts.open_upvalues;
                        let below = open.iter().rposition(|uv| uv.slot() <= slot);
                        let cell = match below {
                            Some(i) if open[i].slot() == slot => open[i],
                            _ => {
                                let uv = UpvalueCell::new_open(rt.mutation(), ts.handle(), slot);
                                let at = below.map_or(0, |i| i + 1);
                                ts.open_upvalues.insert(at, uv);
                                captured = true;
                                uv
                            }
                        };
                        UpvalueSlot { cell }
                    }
                    UpvalSource::ParentUpvalue(idx) => upval!(idx),
                };
                unsafe { slots.add(i).write(slot) };
            }
        });
        if captured {
            unsafe { frame::set_flags(base, flag::HAS_OPEN) };
        }
        if let Some(i) = self_slot {
            // The fresh closure pointing at itself needs no barrier.
            unsafe {
                let f = LuaFn::from_function_unchecked(func);
                (*f.upvalue_ptr().add(i)).value = Value::function(func);
            }
        }
        reg![dst] = Value::function(func);
        gc_check!();
        next!()
    }

    /// Copy varargs into `R[a..]`. `b == 0` is MULTRET (copy all, set `top`);
    /// `b > 0` copies `b - 1`, nil-padding short.
    ///
    /// Source depends on `Prototype::needs_vararg_table`: optimized reads the
    /// extras below the header; materialized reads `1..=t.n` from the table
    /// in `R[num_params]`, so mutations to it are visible.
    op fn op_vararg {
        let (dst, count) = insn.ab();
        if std::hint::unlikely(closure.proto.needs_vararg_table) {
            tail!(vararg_slow)
        }
        let nv = unsafe { frame::nv(base) };
        let wanted = if count == 0 { nv } else { count as usize - 1 };
        let target = unsafe { base.add(dst as usize) };
        if count == 0 {
            let end = unsafe { target.add(wanted) };
            if std::hint::unlikely(end.cast_const() > thread!().stack_end) {
                tail!(vararg_grow)
            }
            let ts = thread!();
            ts.top = ts.slot_index(end);
        }
        // The extras end at the header and the target starts at or above
        // the base, so the ranges don't overlap.
        let extras = unsafe { base.sub(HDR + nv) };
        unsafe { land_results(target, extras, nv, wanted) };
        next!()
    }

    /// VARARG of all the extras into a window that does not fit the stack:
    /// grow (or overflow) and run it again.
    slow fn vararg_grow {
        let insn = insn_at!();
        let (dst, _) = insn.ab();
        sync!();
        let ts = thread!();
        let bi = ts.slot_index(base);
        let nv = unsafe { frame::nv(base) };
        if !ts.ensure_frame_slots(bi + dst as usize + nv) {
            raise!(OpError::StackOverflow)
        }
        base = ts.slot_ptr(bi);
        jump_by!(-1);
        next!()
    }

    /// VARARG of a function whose varargs were materialized into a table
    /// (`VARARGPREP`).
    slow fn vararg_slow {
        let insn = insn_at!();
        let (dst, count) = insn.ab();
        let closure = unsafe { frame::closure(base) };
        // The copy may grow the stack.
        sync!();
        let ts = thread!();
        let bi = ts.slot_index(base);
        let target = bi + dst as usize;
        {
            let table = reg![closure.num_params].get_table().expect("materialized vararg slot must hold a table");
            // PUC's `getnumargs` bound, checked even when `count` is fixed.
            let Some(navail) = table
                .inner()
                .borrow()
                .raw_get(Value::string(rt.symbols().n))
                .get_integer()
                .filter(|n| (0..=i64::from(i32::MAX / 2)).contains(n))
            else {
                raise!(OpError::VarargN)
            };
            let navail = navail as usize;
            let wanted = if count == 0 { navail } else { count as usize - 1 };
            let new_top = target + wanted;
            if !ts.ensure_frame_slots(new_top) {
                raise!(OpError::StackOverflow)
            }
            base = ts.slot_ptr(bi);
            let filled = wanted.min(navail);
            let t = table.inner().borrow();
            for i in 0..filled {
                ts.stack[target + i] = t.raw_get(Value::integer(rt.mutation(), i as i64 + 1));
            }
            ts.stack[target + filled..new_top].fill(Value::nil());
            if count == 0 {
                ts.top = new_top;
            }
            next!()
        }
    }

    /// Optimized below-base read of an un-escaped named vararg: integer key
    /// `1..=nv`, `"n"` = count, else nil. Escaped varargs are rewritten to
    /// `GETTABLE` at compile time, so `b` is unused here. Lua 5.5 `OP_GETVARG`.
    op fn op_varargget {
        let (dst, _, key) = insn.abc();
        let key_val = reg![key];
        let nv = unsafe { frame::nv(base) };
        let extras = unsafe { base.sub(HDR + nv) };
        // Normalize integral float keys (`args[1.0]` == `args[1]`) so the
        // optimized path agrees with the GETTABLE an escaped vararg would use.
        let int_key = key_val.get_integer().or_else(|| {
            let f = key_val.get_float()?;
            let i = f as i64;
            (i as f64 == f).then_some(i)
        });
        let v = if let Some(k) = int_key {
            if k >= 1 && (k as usize) <= nv {
                unsafe { extras.add(k as usize - 1).read() }
            } else {
                Value::nil()
            }
        } else if let Some(s) = key_val.get_string() {
            if s.as_bytes() == b"n" {
                Value::integer(rt.mutation(), nv as i64)
            } else {
                Value::nil()
            }
        } else {
            Value::nil()
        };
        reg![dst] = v;
        next!()
    }

    /// Entry of a vararg function. The extras were rotated below the header
    /// on entry; here only `needs_vararg_table` has work: materialize
    /// `{ extras...; n = count }` into `R[num_params]` (`createvarargtab`).
    op fn op_varargprep {
        if closure.proto.needs_vararg_table {
            let nv = unsafe { frame::nv(base) };
            let np = closure.num_params as usize;
            // Store into the register before filling so a mid-fill allocation
            // can't collect the table.
            let table = Table::new(rt);
            reg![np] = Value::table(table);
            let extras = unsafe { base.sub(HDR + nv) };
            for i in 0..nv {
                let v = unsafe { extras.add(i).read() };
                table.raw_set(rt, Value::integer(rt.mutation(), i as i64 + 1), v);
            }
            table.raw_set(rt, Value::string(rt.symbols().n), Value::integer(rt.mutation(), nv as i64));
            gc_check!();
        }
        next!()
    }

    /// Lua 5.5 ERRNNIL: raise if `R[a]` is **not** nil.
    op fn op_errnnil {
        let (src, name_key) = insn.ad();
        if std::hint::unlikely(!reg![src].is_nil()) {
            raise!(OpError::GlobalRedefined(name_key))
        }
        next!()
    }

    /// A function entry: count it.
    op fn op_func {
        count_hot!(rt, insn, a);
        next!()
    }

    /// A loop header: count an iteration.
    op fn op_loop {
        count_hot!(rt, insn, a);
        next!()
    }

    op fn op_nop {
        next!()
    }

    op fn op_stop {
        exit!(Exit::End)
    }

    /// Dispatch table filler for opcode bytes no instruction uses.
    op fn op_invalid {
        unreachable!("dispatch of an unused opcode byte {:#x}", insn.opcode())
    }
}
