//! Per-opcode emission (`jit-design.md` 7, Appendix D): each instruction's
//! guards from its quickened form and feedback byte, its operation, and its
//! result writes. What milestone 1 does not compile is a deopt.

use crate::env::MetamethodBits;
use crate::env::value::Value;
use crate::instruction::TFOR_VARS;
use crate::instruction::{ArithKind, Family, Instruction, Op as BcOp, UpvalSource};
use crate::jit::build::cfg::RegSet;
use crate::jit::build::intrinsics::Emitted;
use crate::jit::build::{Builder, intrinsics, iter, tables};
use crate::jit::feedback as fb;
use crate::jit::ir::ops::{Cc, ExitTag, HELPER_FAIL_VAL, HelperId, Op};
use crate::jit::ir::types::{Rep, TypeSet};
use crate::jit::ir::{Block, Val};
use crate::vm::frame::flag;

/// What the hot path assumes about one arithmetic operand.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum NumKind {
    I32,
    I64,
    F64,
    /// A small integer or a float, told apart where the result depends on it.
    Num,
}

/// The kind a value has for certain.
fn own_kind(b: &Builder<'_, '_>, v: Val) -> Option<NumKind> {
    let t = b.ty(v);
    match t.rep {
        Rep::I32 => return Some(NumKind::I32),
        Rep::I64 => return Some(NumKind::I64),
        Rep::F64 => return Some(NumKind::F64),
        _ => {}
    }
    if t.set.is_empty() {
        None
    } else if t.within(TypeSet::SMALL) {
        Some(NumKind::I32)
    } else if t.within(TypeSet::FLOAT) {
        Some(NumKind::F64)
    } else if t.within(TypeSet::INT) {
        Some(NumKind::I64)
    } else if t.within(TypeSet::SMALL | TypeSet::FLOAT) {
        Some(NumKind::Num)
    } else {
        None
    }
}

/// The operand kinds a feedback byte gives both operands of its site, or
/// `None` when it saw values no number kind covers (non-numbers, a
/// metamethod, big integers with floats): those take the slow arm.
fn byte_num_kind(byte: u8) -> Option<NumKind> {
    let kinds = byte & fb::KINDS;
    if byte & fb::MM != 0 || kinds & (fb::STR | fb::TAB | fb::OTHER) != 0 {
        return None;
    }
    let wide = byte & (fb::OVERFLOW | fb::BIGINT) != 0;
    match (kinds & fb::INT != 0, kinds & fb::FLOAT != 0) {
        // Only what the inline paths cover: a site locked on numbers.
        (false, false) => Some(NumKind::Num),
        (true, false) if wide => Some(NumKind::I64),
        (true, false) => Some(NumKind::I32),
        (false, true) => Some(NumKind::F64),
        (true, true) if kinds & fb::BIGINT != 0 => None,
        (true, true) => Some(NumKind::Num),
    }
}

/// The operand kinds the quickened form at `pc` stands for, unless compiled
/// code saw it miss (6.4): `(left, right)`, an immediate's side `None`.
fn form_kinds(b: &Builder<'_, '_>, pc: u32) -> Option<(Option<NumKind>, Option<NumKind>)> {
    let insn = b.code(pc);
    let op = insn.op();
    let info = insn.generic_op().info();
    if op == insn.generic_op() || b.kind_exited(pc) {
        return None;
    }
    let wide = b.feedback(pc) & (fb::OVERFLOW | fb::BIGINT) != 0;
    let int = if wide { NumKind::I64 } else { NumKind::I32 };
    use NumKind::*;
    let idx = info.forms.iter().position(|f| *f == Some(op))?;
    if matches!(info.family, Family::ImmArith | Family::ImmBit) {
        let reg = match idx {
            1 => F64,
            // `_IF`: a small register whose immediate or operation makes
            // the result a float.
            _ => int,
        };
        return Some(if info.reversed {
            (None, Some(reg))
        } else {
            (Some(reg), None)
        });
    }
    Some(match idx {
        0 => (Some(int), Some(int)),
        1 => (Some(F64), Some(F64)),
        2 => (Some(I32), Some(F64)),
        3 => (Some(F64), Some(I32)),
        _ => (Some(Num), Some(Num)),
    })
}

/// Whether the instruction at `pc` compiles to an unconditional deopt.
pub(super) fn always_deopts(b: &Builder<'_, '_>, pc: u32) -> bool {
    let insn = b.code(pc);
    use BcOp::*;
    match insn.op() {
        MOVE | LOAD | LOADI | LOADNIL | LFALSESKIP | GETUPVAL | NOP | FUNC | LOOP | JMP
        | JMP_BACK | NOT | ERRNNIL | JT | JF | JTSET | JFSET | RETURN0 | RETURN1 | FORLOOP_I
        | FORLOOP_F | FORLOOP | JEQS | JNEQS | LEN | CONCAT | GETUPVAL_REF | SETUPVAL | CLOSURE
        | CLOSE | GETTABLE | SETTABLE | TFORPREP | TFORCALL | TFORCALL_NEXT | TFORCALL_IPAIRS
        | TFORLOOP => false,
        // A tail call of an intrinsic compiles; any other deopts at itself.
        TAILCALL => false,
        // A vararg table makes VARARG read it, and MULTRET sets `top`.
        VARARG => b.proto.needs_vararg_table,
        VARARGPREP | VARARGGET | NEWTABLE => false,
        SETLIST | RETURN | CALL | CALL_R0 | CALL_R1 | CALLS | CALLS_R0 | CALLS_R1 => false,
        FORPREP => false,
        UNM | BNOT => false,
        op => match op.info().family {
            Family::RegArith | Family::RegBit | Family::ImmArith | Family::ImmBit => false,
            Family::CmpReg | Family::CmpImm => false,
            _ if is_get(op) || is_set(op) => false,
            _ => !matches!(op, JEQI | JNEQI),
        },
    }
}

/// How the numeric loop whose FORLOOP is at `pc` compiles: as its form,
/// unless the feedback of the FORLOOP or its FORPREP saw floats or big
/// integers (`FORPREP` does not rewrite a FORLOOP that is a JIT word).
/// `None` when it saw both integer and float loops, which deopt: a loop's
/// kind is visible, so no one representation runs both.
fn loop_kind(b: &Builder<'_, '_>, pc: u32) -> Option<BcOp> {
    let insn = b.code(pc);
    let seen = loop_seen(b, pc);
    match (seen & fb::INT != 0, seen & fb::FLOAT != 0) {
        (true, true) => None,
        (false, true) => Some(BcOp::FORLOOP_F),
        _ if seen & fb::BIGINT != 0 => Some(BcOp::FORLOOP),
        _ => Some(insn.op()),
    }
}

/// The kinds recorded at the FORLOOP at `pc` and its FORPREP.
fn loop_seen(b: &Builder<'_, '_>, pc: u32) -> u8 {
    let prep = (pc as i64 + b.code(pc).branch_offset() as i64) as u32;
    (b.feedback(pc) | b.feedback(prep)) & fb::KINDS
}

/// The f64 of a number operand of a float loop, which may be an integer.
fn any_f64(b: &mut Builder<'_, '_>, v: Val) -> Val {
    let t = b.ty(v);
    if matches!(t.rep, Rep::I32 | Rep::I64 | Rep::F64)
        || (!t.set.is_empty() && (t.within(TypeSet::SMALL) || t.within(TypeSet::FLOAT)))
    {
        return b.num_f64(v);
    }
    let i = b.ins_exit(Op::ToF64, &[v], ExitTag::Type);
    b.f.result(i)
}

/// Emit the instruction at `pc`. `succs` are the IR blocks of its bytecode
/// block's successors when it is the block's last instruction:
/// `[fallthrough, target]` for a branch, `[target]` for a jump.
pub(super) fn emit(b: &mut Builder<'_, '_>, pc: u32, succs: &[Block]) {
    let insn = b.code(pc);
    if always_deopts(b, pc) {
        b.deopt(ExitTag::Unsupported);
        return;
    }
    use BcOp::*;
    let (a, rb) = (insn.a(), insn.b());
    match insn.op() {
        NOP | FUNC | LOOP => {}
        VARARGPREP => {
            if b.proto.needs_vararg_table {
                let np = b.closure.num_params;
                let t = b.ins(Op::Helper(HelperId::VarargTable(np)), &[]);
                b.set(np, t);
            }
        }
        VARARG if rb == 0 => {
            // The callee and first argument of a call it passes the extras
            // to, for its intrinsic, which the slot reloads below would hide.
            let next = b.code(pc + 1);
            if matches!(
                next.op(),
                BcOp::CALL | BcOp::CALL_R0 | BcOp::CALL_R1 | BcOp::TAILCALL
            ) && next.b() == 0
                && next.a() as usize + 5 == a as usize
            {
                let (f, first) = (b.reg(next.a()), b.reg(next.a() + 4));
                b.callee_hint = Some((pc + 1, f, first));
            }
            // Registers below `a` live after it are stored; the block after
            // reads every register from its slot.
            let mut live = b
                .cfg
                .live_in
                .get(pc as usize + 1)
                .copied()
                .unwrap_or_default();
            for r in a as usize..b.cfg.nregs {
                live.remove(r);
            }
            let tys = b.store_live(live);
            let r = b.ins(Op::Helper(HelperId::VarargAll(a)), &[]);
            let fail = b.ins(Op::KVal(HELPER_FAIL_VAL), &[]);
            let failed = b.ins(Op::SameBits, &[r, fail]);
            let s = b.snap_before();
            b.ins_snap(Op::GuardFalse, &[failed], s, ExitTag::Slow);
            b.sync_root(tys);
        }
        VARARG => {
            let nv = b.ins(Op::NumVarargs, &[]);
            let vals: Vec<Val> = (0..rb - 1).map(|k| b.ins(Op::VarargAt(k), &[nv])).collect();
            for (k, &v) in vals.iter().enumerate() {
                b.set(a + k as u8, v);
            }
            b.note_def(&vals, None);
        }
        VARARGGET => {
            let key = b.reg(insn.c());
            let v = if own_kind(b, key) == Some(NumKind::I32) {
                let nv = b.ins(Op::NumVarargs, &[]);
                let k = b.as_i32(key);
                b.ins(Op::VarargIdx, &[nv, k])
            } else {
                let key = b.boxed(key);
                b.ins(Op::Helper(HelperId::VarargGet), &[key])
            };
            b.set(a, v);
            b.note_def(&[v], None);
        }
        NEWTABLE => {
            let t = b.ins(Op::Helper(HelperId::NewTable(insn.d())), &[]);
            b.set(a, t);
        }
        SETLIST => {
            // The helper reads the table and the items from their slots (up
            // to `top`: those before the ones a call or VARARG landed).
            let end = if rb != 0 {
                a as usize + rb as usize
            } else {
                b.cfg.nregs - 1
            };
            b.store_regs(a as usize..=end);
            let off = insn.d() as u32;
            b.push(Op::Helper(HelperId::SetList { a, n: rb, off }), &[]);
        }
        JMP | JMP_BACK => b.jump(succs[0]),
        MOVE => {
            let v = b.reg(rb);
            b.set(a, v);
        }
        LOAD => {
            let k = b.closure.proto.constants[insn.d() as usize];
            let v = b.konst(k);
            b.set(a, v);
        }
        LOADI => {
            let v = b.konst(Value::small(insn.imm()));
            b.set(a, v);
        }
        LOADNIL => {
            for r in a..a + rb {
                let v = b.konst(Value::nil());
                b.set(r, v);
            }
        }
        LFALSESKIP => {
            let v = b.konst(Value::boolean(false));
            b.set(a, v);
            b.flush();
            b.jump(succs[0]);
        }
        // An inlined callee's closure is a constant, and so its by-value
        // upvalues (E1).
        GETUPVAL if b.inline.is_some() => {
            let now = unsafe { (*b.closure.upvalue_ptr().add(rb as usize)).value };
            let k = b.konst(now);
            b.set(a, k);
        }
        // A by-value upvalue is a constant of the closure, guarded once per
        // region since other closures of the prototype may run it.
        GETUPVAL if b.feedback(pc) & fb::CONST_MISS == 0 => {
            let v = b.ins(Op::UpvalValue(rb), &[]);
            let now = unsafe { (*b.closure.upvalue_ptr().add(rb as usize)).value };
            let k = b.konst(now);
            b.ins_exit(Op::GuardSame, &[v, k], ExitTag::Const);
            b.set(a, k);
        }
        GETUPVAL => {
            let v = b.ins(Op::UpvalValue(rb), &[]);
            b.set(a, v);
            b.note_def(&[v], None);
        }
        NOT => {
            let v = b.reg(rb);
            let c = b.ins(Op::IsFalsy, &[v]);
            let r = b.ins(Op::Box, &[c]);
            b.set(a, r);
        }
        ERRNNIL => {
            let v = b.reg(a);
            b.ins_exit(Op::Guard(TypeSet::NIL), &[v], ExitTag::Slow);
        }
        UNM => emit_unm(b, pc, a, rb),
        BNOT => emit_bnot(b, pc, a, rb),
        LEN => emit_len(b, a, rb),
        CONCAT => {
            let (x, y) = (b.reg(rb), b.reg(insn.c()));
            let (x, y) = (b.boxed(x), b.boxed(y));
            helper_or_mm(
                b,
                a,
                HelperId::Concat,
                &[x, y],
                MetamethodBits::CONCAT,
                &[x, y],
            );
        }
        JT | JF => {
            let v = b.reg(a);
            let falsy = b.ins(Op::IsFalsy, &[v]);
            // JT jumps when truthy: when not falsy.
            let jump_if_falsy = insn.op() == JF;
            branch(b, falsy, jump_if_falsy, succs);
        }
        JTSET | JFSET => {
            let v = b.reg(rb);
            let falsy = b.ins(Op::IsFalsy, &[v]);
            let jump_if_falsy = insn.op() == JFSET;
            let edge = b.new_sealed_succ();
            if jump_if_falsy {
                b.br(falsy, edge, succs[0]);
            } else {
                b.br(falsy, succs[0], edge);
            }
            b.switch_to(edge);
            b.set(a, v);
            b.flush();
            b.jump(succs[1]);
        }
        JEQS | JNEQS => {
            let v = b.reg(a);
            let k = b.closure.proto.constants[insn.h() as usize];
            let kv = b.konst(k);
            let eq = b.ins(Op::SameBits, &[v, kv]);
            branch(b, eq, insn.op() == JEQS, succs);
        }
        RETURN0 | RETURN1 | RETURN if b.inline.is_some() && (insn.op() != RETURN || rb != 0) => {
            let n = match insn.op() {
                RETURN0 => 0,
                RETURN1 => 1,
                _ => rb - 1,
            };
            super::inline::inline_return(b, a, n);
        }
        RETURN0 => {
            b.terminate(Op::Return { a: 0, n: 0 }, &[], &[], None);
        }
        RETURN1 => {
            store_results(b, a, 1);
            b.terminate(Op::Return { a, n: 1 }, &[], &[], None);
        }
        // Up to `top`: the interpreter's RETURN, which closes too.
        RETURN if rb == 0 => dispatch(b),
        RETURN => emit_return(b, a, rb - 1),
        TAILCALL => {
            let callee = callee_at(b, pc, a);
            // The frame before the TAILCALL, for the return's exit.
            let before = b.snap_before();
            let slow = match intrinsics::emit(b, pc, insn, callee) {
                Emitted::No => Some(b.cur),
                Emitted::Done => None,
                Emitted::Fallback(slow, _) => Some(slow),
            };
            if slow != Some(b.cur) {
                b.flush();
                emit_return_from(b, a, 1, before);
            }
            if let Some(slow) = slow {
                b.switch_to(slow);
                if rb != 0 {
                    emit_tailcall(b, a, rb - 1);
                } else {
                    dispatch(b);
                }
            }
        }
        GETUPVAL_REF => {
            let c = b.ins(Op::UpvalCell(rb), &[]);
            let v = b.ins(Op::CellLoad, &[c]);
            b.set(a, v);
            b.note_def(&[v], None);
        }
        SETUPVAL => {
            let v = b.reg(a);
            let v = b.boxed(v);
            let c = b.ins(Op::UpvalCell(rb), &[]);
            b.push(Op::CellStore, &[c, v]);
            let set = b.ty(v).set;
            if set.is_empty() || set.intersects(TypeSet::HEAP) {
                b.push(Op::Helper(HelperId::UpvalBarrier(rb)), &[]);
            }
        }
        CLOSURE => {
            let proto = insn.d();
            // The helper reads the captured registers from their slots.
            let child = b.proto.prototypes[proto as usize];
            let regs: Vec<usize> = child
                .upvalue_desc
                .iter()
                .filter_map(|d| match d.source {
                    UpvalSource::ParentLocal(r) if !(d.by_value && r == a) => Some(r as usize),
                    _ => None,
                })
                .collect();
            b.store_regs(regs);
            let f = b.ins(Op::Helper(HelperId::Closure { proto, dst: a }), &[]);
            b.set(a, f);
        }
        CLOSE => {
            // A `__close` runs Lua: the interpreter does it.
            let s = b.snap_before();
            b.ins_snap(
                Op::GuardNoFlag(flag::HAS_TBC as u8),
                &[],
                s,
                ExitTag::Unsupported,
            );
            let regs: Vec<usize> = b.cfg.captured.iter().filter(|&r| r >= a as usize).collect();
            b.store_regs(regs);
            b.push(Op::Helper(HelperId::Close(a)), &[]);
        }
        CALL | CALL_R0 | CALL_R1 | CALLS | CALLS_R0 | CALLS_R1 => emit_call(b, pc, insn),
        TFORPREP => iter::emit_tforprep(b, pc, insn, succs),
        TFORCALL => iter::emit_tforcall(b, a, rb),
        TFORCALL_NEXT => iter::emit_tforcall_next(b, a, rb),
        TFORCALL_IPAIRS => iter::emit_tforcall_ipairs(b, a, rb),
        TFORLOOP => {
            let v = b.reg(a + TFOR_VARS);
            let v = b.boxed(v);
            let nil = b.ins(Op::IsType(TypeSet::NIL), &[v]);
            branch(b, nil, false, succs);
        }
        GETTABLE => tables::emit_gettable(b, pc, insn),
        SETTABLE => tables::emit_settable(b, pc, insn),
        FORPREP => emit_forprep(b, pc, insn, succs),
        FORLOOP_I | FORLOOP_F | FORLOOP => match loop_kind(b, pc) {
            Some(FORLOOP_I) => emit_forloop_i(b, a, succs),
            Some(FORLOOP_F) => emit_forloop_f(b, a, succs),
            // FORPREP records where it makes a FORLOOP generic, and
            // `forloop_slow` on each iteration: no record, no iteration.
            Some(_) if insn.op() == FORLOOP && loop_seen(b, pc) == 0 => b.deopt(ExitTag::NeverRan),
            Some(_) => emit_forloop_l(b, a, succs),
            None => b.deopt(ExitTag::Unsupported),
        },
        JEQI | JNEQI => emit_eqi(b, pc, insn, succs),
        op => match op.info().family {
            Family::RegArith | Family::RegBit | Family::ImmArith | Family::ImmBit => {
                emit_arith(b, pc, insn)
            }
            Family::CmpReg => emit_cmp_reg(b, pc, insn, succs),
            _ if is_get(op) => tables::emit_get(b, insn),
            _ if is_set(op) => tables::emit_set(b, insn),
            Family::CmpImm => emit_cmp_imm(b, pc, insn, succs),
            _ => b.deopt(ExitTag::Unsupported),
        },
    }
}

/// A constant-key read in any form.
fn is_get(op: BcOp) -> bool {
    matches!(
        op.unquickened(),
        BcOp::GETFIELD | BcOp::GETTABUP | BcOp::GETTABUP_REF | BcOp::SELF
    )
}

/// A constant-key write in any form.
fn is_set(op: BcOp) -> bool {
    matches!(
        op.unquickened(),
        BcOp::SETFIELD | BcOp::SETTABUP | BcOp::SETTABUP_REF
    )
}

/// End the block on `cond`: to the jump target when `cond == jump_when`.
fn branch(b: &mut Builder<'_, '_>, cond: Val, jump_when: bool, succs: &[Block]) {
    let (fall, target) = (succs[0], succs[1]);
    if jump_when {
        b.br(cond, target, fall);
    } else {
        b.br(cond, fall, target);
    }
}

/// Leave for the interpreter's handler of the instruction being emitted,
/// with what is live stored (R1): what is left of it leaves the frame.
fn dispatch(b: &mut Builder<'_, '_>) {
    let pc = b.pc;
    let live = b.cfg.live_in[pc as usize];
    b.store_live(live);
    let word = b.code(pc).raw();
    b.terminate(Op::Dispatch { pc, word }, &[], &[], None);
}

/// `TAILCALL a` with `nargs` arguments: the fast arm of 5.5 for a Lua
/// closure without varargs whose window fits, from a frame with nothing to
/// close; any other callee the interpreter's TAILCALL, from the slots.
fn emit_tailcall(b: &mut Builder<'_, '_>, a: u8, nargs: u8) {
    use crate::vm::frame::flag::{HAS_OPEN, HAS_TBC};
    let f = b.reg(a);
    let f = b.boxed(f);
    // Read before the stores below overwrite their slots.
    let args: Vec<Val> = (0..nargs)
        .map(|k| {
            let v = b.reg(a + 4 + k);
            b.boxed(v)
        })
        .collect();
    let slow = b.new_block();
    let mut others = vec![arm(b, Op::IsType(TypeSet::FUN), f)];
    let fun = b.ins(Op::Refine(TypeSet::FUN), &[f]);
    let p = b.ins(Op::Unbox(Rep::Ptr), &[fun]);
    others.push(arm(b, Op::FnFixedLua, p));
    let closes = b.ins(Op::HasFlag((HAS_OPEN | HAS_TBC) as u8), &[]);
    let (closing, fast) = (b.new_sealed_succ(), b.new_sealed_succ());
    b.br(closes, closing, fast);
    b.switch_to(fast);
    others.push(closing);
    others.push(arm(b, Op::FnFits, p));
    for (k, &v) in args.iter().enumerate() {
        b.push(Op::Store(k as u8), &[v]);
    }
    b.terminate(Op::TailEnter { nargs }, &[p], &[], None);
    for o in others {
        b.switch_to(o);
        b.jump(slow);
    }
    enter_join(b, slow);
    dispatch(b);
}

/// `RETURN` of `n` results from `R[a..]`. A frame with to-be-closed
/// variables deopts; one whose closures captured registers closes their
/// cells first, from the captured registers' slots.
fn emit_return(b: &mut Builder<'_, '_>, a: u8, n: u8) {
    let s = b.snap_before();
    emit_return_from(b, a, n, s);
}

/// `emit_return` exiting with the frame `s`.
fn emit_return_from(b: &mut Builder<'_, '_>, a: u8, n: u8, s: crate::jit::ir::Snap) {
    let captures = b.cfg.captured.iter().next().is_some();
    let mask = if captures {
        flag::HAS_TBC
    } else {
        flag::HAS_OPEN | flag::HAS_TBC
    };
    b.ins_snap(Op::GuardNoFlag(mask as u8), &[], s, ExitTag::Unsupported);
    if captures {
        // Results may overwrite captured slots, as in the interpreter.
        let regs: Vec<usize> = b.cfg.captured.iter().collect();
        b.store_regs(regs);
        store_results(b, a, n);
        let join = b.new_block();
        let open = b.ins(Op::HasFlag(flag::HAS_OPEN as u8), &[]);
        let rest = branch_arm(b, open);
        b.push(Op::Helper(HelperId::Close(0)), &[]);
        b.jump(join);
        b.switch_to(rest);
        b.jump(join);
        enter_join(b, join);
    } else {
        store_results(b, a, n);
    }
    b.terminate(Op::Return { a, n }, &[], &[], None);
}

/// Store `R[a..a+n]` before a return.
fn store_results(b: &mut Builder<'_, '_>, a: u8, n: u8) {
    for r in a..a + n {
        let v = b.reg(r);
        let bv = b.boxed(v);
        b.push(Op::Store(r), &[bv]);
    }
}

// --- arithmetic -------------------------------------------------------------

fn emit_arith(b: &mut Builder<'_, '_>, pc: u32, insn: Instruction) {
    if matches!(
        insn.op(),
        BcOp::ARITH_MM | BcOp::ARITH_MM_R | BcOp::ARITH_MMI
    ) && !b.kind_exited(pc)
    {
        emit_arith_mm(b, insn);
        return;
    }
    let info = insn.generic_op().info();
    let kind = info.kind;
    let imm = matches!(info.family, Family::ImmArith | Family::ImmBit);
    let dst = insn.a();
    // Operands in source order: the immediate is on the left for the `R` forms.
    let (l, r): (Operand, Operand) = if imm {
        let reg = Operand::Reg(b.reg(insn.b()));
        let k = if insn.imm_is_int() {
            Operand::Int(insn.imm_int())
        } else {
            Operand::Float(insn.imm_float())
        };
        if info.reversed { (k, reg) } else { (reg, k) }
    } else {
        (Operand::Reg(b.reg(insn.b())), Operand::Reg(b.reg(insn.c())))
    };
    let byte = b.feedback(pc);
    let (fl, fr) = form_kinds(b, pc).unwrap_or((None, None));
    let (lk, rk) = (l.kind(b, fl, byte), r.kind(b, fr, byte));
    let wide = byte & (fb::OVERFLOW | fb::BIGINT) != 0;
    if let (Some(lk), Some(rk)) = (lk, rk) {
        arith_kinds(b, kind, dst, (l, lk), (r, rk), wide, None);
        return;
    }
    // An operand that may be no number: the numeric path behind a test,
    // the rest in the slow arm.
    let join = b.new_block();
    let unknown: Vec<Val> = [(l, lk), (r, rk)]
        .into_iter()
        .filter(|&(_, k)| k.is_none())
        .map(|(o, _)| o.val(b))
        .collect();
    let slow = all_in(b, TypeSet::SMALL | TypeSet::FLOAT, &unknown);
    let num = |k: Option<NumKind>| k.unwrap_or(NumKind::Num);
    arith_kinds(b, kind, dst, (l, num(lk)), (r, num(rk)), wide, Some(join));
    enter_join(b, slow);
    cold(b, |b| {
        let (x, y) = (l.val(b), r.val(b));
        slow_arm(
            b,
            join,
            dst,
            HelperId::ArithCoerce(kind),
            &[x, y],
            kind_mm(kind),
            &[x, y],
        );
    });
    enter_join(b, join);
}

/// `dst = l <kind> r` for operands of these kinds, at the end of the
/// current block, or as an arm into `join`.
fn arith_kinds(
    b: &mut Builder<'_, '_>,
    kind: ArithKind,
    dst: u8,
    (l, lk): (Operand, NumKind),
    (r, rk): (Operand, NumKind),
    wide: bool,
    join: Option<Block>,
) {
    use ArithKind::*;
    use NumKind::*;
    let bitwise = matches!(kind, BAnd | BOr | BXor | Shl | Shr);
    let float = matches!(kind, Div | Pow) || lk == F64 || rk == F64;
    let res = if bitwise {
        let (x, y) = (l.int(b, lk), r.int(b, rk));
        int_arith(b, kind, x, y, wide)
    } else if float {
        let (x, y) = (l.f64(b, lk), r.f64(b, rk));
        float_arith(b, kind, x, y, r)
    } else if lk == Num || rk == Num {
        // Two small integers make an integer, anything else a float.
        let to = join.unwrap_or_else(|| b.new_block());
        let nums: Vec<Val> = [(l, lk), (r, rk)]
            .into_iter()
            .filter(|&(_, k)| k == Num)
            .map(|(o, _)| o.val(b))
            .collect();
        let floats = all_in(b, TypeSet::SMALL, &nums);
        let (x, y) = (l.small(b, lk), r.small(b, rk));
        let v = int_arith(b, kind, x, y, wide);
        arm_end(b, to, dst, v);
        enter_join(b, floats);
        let (x, y) = (l.f64(b, lk), r.f64(b, rk));
        let v = float_arith(b, kind, x, y, r);
        arm_end(b, to, dst, v);
        if join.is_none() {
            enter_join(b, to);
        }
        return;
    } else {
        let (x, y) = (l.int(b, lk), r.int(b, rk));
        int_arith(b, kind, x, y, wide)
    };
    if let Some(j) = join {
        arm_end(b, j, dst, res);
    } else {
        b.set(dst, res);
    }
}

/// Continue where every value of `vals` is in `set`; the block where one is
/// not, unsealed until its predecessors are all there (`enter_join`).
fn all_in(b: &mut Builder<'_, '_>, set: TypeSet, vals: &[Val]) -> Block {
    let other = b.new_block();
    for &v in vals {
        let c = b.ins(Op::IsType(set), &[v]);
        let no = branch_arm(b, c);
        let yes = b.cur;
        b.switch_to(no);
        b.jump(other);
        b.switch_to(yes);
    }
    other
}

/// The slow arm of a site whose feedback saw what no number kind covers:
/// the helper and the metamethod (`helper_arms`) where it saw strings, big
/// integers or a metamethod; otherwise only what the interpreter raises on
/// gets here.
fn slow_arm(
    b: &mut Builder<'_, '_>,
    join: Block,
    dst: u8,
    h: HelperId,
    args: &[Val],
    bit: MetamethodBits,
    mm_args: &[Val],
) {
    if b.feedback(b.pc) & (fb::STR | fb::BIGINT | fb::MM) != 0 {
        helper_arms(b, join, dst, h, args, bit, mm_args);
    } else {
        b.deopt(ExitTag::Slow);
    }
}

/// An arithmetic operand: a register's value or the instruction's immediate.
#[derive(Clone, Copy)]
enum Operand {
    Reg(Val),
    Int(i64),
    Float(f64),
}

impl Operand {
    /// The kind the hot path assumes: the value's own, else its form's,
    /// else the site's feedback (`byte_num_kind`).
    fn kind(self, b: &Builder<'_, '_>, form: Option<NumKind>, byte: u8) -> Option<NumKind> {
        match self {
            Operand::Reg(v) => own_kind(b, v).or(form).or_else(|| byte_num_kind(byte)),
            Operand::Int(_) => Some(NumKind::I32),
            Operand::Float(_) => Some(NumKind::F64),
        }
    }

    fn val(self, b: &mut Builder<'_, '_>) -> Val {
        match self {
            Operand::Reg(v) => b.boxed(v),
            Operand::Int(n) => b.konst(Value::small(n as i32)),
            Operand::Float(x) => b.konst(Value::float(x)),
        }
    }

    /// The integer of an operand of kind `k`, guarded to it.
    fn int(self, b: &mut Builder<'_, '_>, k: NumKind) -> Val {
        match (self, k) {
            (Operand::Reg(v), NumKind::I32 | NumKind::Num) => b.as_i32(v),
            (Operand::Reg(v), NumKind::I64) => b.as_i64(v),
            // Exits where the float has no integer representation in i32.
            (Operand::Reg(v), NumKind::F64) => {
                let x = b.as_f64(v);
                let i = b.ins_exit(Op::FToIExact, &[x], ExitTag::Type);
                b.f.result(i)
            }
            (Operand::Int(n), _) => b.ki32(n as i32),
            (Operand::Float(x), _) => b.ki32(x as i32),
        }
    }

    /// The integer of an operand that is a small integer here.
    fn small(self, b: &mut Builder<'_, '_>, k: NumKind) -> Val {
        match (self, k) {
            (Operand::Reg(v), NumKind::Num) => {
                let v = b.boxed(v);
                b.ins(Op::Unbox(Rep::I32), &[v])
            }
            (o, k) => o.int(b, k),
        }
    }

    /// The f64 of an operand of kind `k`: integers converted as Lua's
    /// arithmetic does (rounding), a `Num` by a test of its tag.
    fn f64(self, b: &mut Builder<'_, '_>, k: NumKind) -> Val {
        match (self, k) {
            (Operand::Reg(v), NumKind::I32) => {
                let x = b.as_i32(v);
                b.ins(Op::IToF, &[x])
            }
            (Operand::Reg(v), NumKind::I64) => {
                let x = b.as_i64(v);
                b.ins(Op::LToF, &[x])
            }
            (Operand::Reg(v), NumKind::F64) => b.as_f64(v),
            (Operand::Reg(v), NumKind::Num) => {
                let v = b.boxed(v);
                let i = b.ins_exit(Op::ToF64, &[v], ExitTag::Type);
                b.f.result(i)
            }
            (Operand::Int(n), _) => b.kf64(n as f64),
            (Operand::Float(x), _) => b.kf64(x),
        }
    }

    /// The operand's number when it is a constant.
    fn konst(self, b: &Builder<'_, '_>) -> Option<f64> {
        match self {
            Operand::Int(n) => Some(n as f64),
            Operand::Float(x) => Some(x),
            Operand::Reg(v) => match b.f.def_op(v)? {
                Op::KI32(n) => Some(n as f64),
                Op::KF64(bits) => Some(f64::from_bits(bits)),
                Op::KVal(bits) => {
                    let k: Value<'_> = unsafe { std::mem::transmute::<u64, Value<'static>>(bits) };
                    k.get_small().map(f64::from).or(k.get_float())
                }
                _ => None,
            },
        }
    }
}

fn emit_unm(b: &mut Builder<'_, '_>, pc: u32, dst: u8, src: u8) {
    let v = b.reg(src);
    let byte = b.feedback(pc);
    // It negated `i32::MIN`.
    let wide = byte & fb::OVERFLOW != 0;
    let neg_int = |b: &mut Builder<'_, '_>, x: Val| {
        if wide {
            let x = b.ins(Op::IToL, &[x]);
            b.ins(Op::LNeg, &[x])
        } else {
            let i = b.ins_exit(Op::INeg, &[x], ExitTag::Overflow);
            b.f.result(i)
        }
    };
    // The interpreter's fast path, which records nothing, takes small
    // integers and floats: `byte_num_kind` makes an empty byte `Num`.
    let r = match own_kind(b, v).or_else(|| byte_num_kind(byte)) {
        Some(NumKind::F64) => {
            let x = b.as_f64(v);
            b.ins(Op::FNeg, &[x])
        }
        Some(NumKind::I32) => {
            let x = b.as_i32(v);
            neg_int(b, x)
        }
        Some(NumKind::I64) => {
            let x = b.as_i64(v);
            b.ins(Op::LNeg, &[x])
        }
        k => {
            let x = b.boxed(v);
            let join = b.new_block();
            let slow = if k.is_none() {
                Some(all_in(b, TypeSet::SMALL | TypeSet::FLOAT, &[x]))
            } else {
                None
            };
            let floats = all_in(b, TypeSet::SMALL, &[x]);
            let i = b.ins(Op::Unbox(Rep::I32), &[x]);
            let r = neg_int(b, i);
            arm_end(b, join, dst, r);
            enter_join(b, floats);
            let f = b.ins_exit(Op::ToF64, &[x], ExitTag::Type);
            let f = b.f.result(f);
            let r = b.ins(Op::FNeg, &[f]);
            arm_end(b, join, dst, r);
            if let Some(slow) = slow {
                enter_join(b, slow);
                // Lua passes the operand twice for unary metamethods.
                cold(b, |b| {
                    slow_arm(
                        b,
                        join,
                        dst,
                        HelperId::Unm,
                        &[x],
                        MetamethodBits::UNM,
                        &[x, x],
                    )
                });
            }
            enter_join(b, join);
            return;
        }
    };
    b.set(dst, r);
}

fn emit_bnot(b: &mut Builder<'_, '_>, pc: u32, dst: u8, src: u8) {
    let v = b.reg(src);
    // The fast path takes small integers only.
    let r = match own_kind(b, v).or_else(|| byte_num_kind(b.feedback(pc))) {
        Some(NumKind::I64) => {
            let x = b.as_i64(v);
            b.ins(Op::LNot, &[x])
        }
        Some(k) => {
            let x = Operand::Reg(v).int(b, k);
            b.ins(Op::INot, &[x])
        }
        None => {
            let x = b.boxed(v);
            let join = b.new_block();
            let slow = all_in(b, TypeSet::SMALL, &[x]);
            let i = b.ins(Op::Unbox(Rep::I32), &[x]);
            let r = b.ins(Op::INot, &[i]);
            arm_end(b, join, dst, r);
            enter_join(b, slow);
            cold(b, |b| {
                slow_arm(
                    b,
                    join,
                    dst,
                    HelperId::BNot,
                    &[x],
                    MetamethodBits::BNOT,
                    &[x, x],
                )
            });
            enter_join(b, join);
            return;
        }
    };
    b.set(dst, r);
}

/// `ARITH_MM`, `_MM_R` and `_MMI` (7.2): the metamethod the table operand's
/// class has, called. Operands of other kinds, or a class without it, exit
/// as a type guard would, and the site compiles generically after.
fn emit_arith_mm(b: &mut Builder<'_, '_>, insn: Instruction) {
    let op = insn.op();
    let orig = insn.mm_orig();
    let idx = kind_mm(orig.info().kind).bits().trailing_zeros() as u8;
    let (l, r, number_left) = match op {
        BcOp::ARITH_MMI => {
            let v = b.reg(insn.b());
            let k = b.konst(if insn.imm_is_int() {
                Value::small(insn.imm_int() as i32)
            } else {
                Value::float(insn.imm_float())
            });
            if insn.c() & 1 != 0 || orig.info().reversed {
                (k, v, true)
            } else {
                (v, k, false)
            }
        }
        BcOp::ARITH_MM_R => (b.reg(insn.b()), b.reg(insn.c()), true),
        _ => (b.reg(insn.b()), b.reg(insn.c()), false),
    };
    let (l, t) = if number_left {
        let l = b.guard(l, TypeSet::NUM);
        let t = b.guard(r, TypeSet::TAB);
        // The numbers' own metamethod would come first.
        let set = b.ins(Op::NumMtSet, &[]);
        let s = b.snap_before();
        b.ins_snap(Op::GuardFalse, &[set], s, ExitTag::Slow);
        (l, t)
    } else {
        let t = b.guard(l, TypeSet::TAB);
        (t, t)
    };
    let p = b.ins(Op::Unbox(Rep::Ptr), &[t]);
    let f = b.ins(Op::ClassMm(idx), &[p]);
    let none = b.ins(Op::IsType(TypeSet::NIL), &[f]);
    let s = b.snap_before();
    b.ins_snap(Op::GuardFalse, &[none], s, ExitTag::Type);
    let (lv, rv) = if number_left {
        (b.boxed(l), t)
    } else {
        (t, b.boxed(r))
    };
    let v = mm_call(b, f, &[lv, rv]);
    b.set(insn.a(), v);
}

/// The metamethod of an arithmetic kind (`arith_slow`'s).
fn kind_mm(kind: ArithKind) -> MetamethodBits {
    use ArithKind::*;
    match kind {
        Add => MetamethodBits::ADD,
        Sub => MetamethodBits::SUB,
        Mul => MetamethodBits::MUL,
        Mod => MetamethodBits::MOD,
        Pow => MetamethodBits::POW,
        Div => MetamethodBits::DIV,
        IDiv => MetamethodBits::IDIV,
        BAnd => MetamethodBits::BAND,
        BOr => MetamethodBits::BOR,
        BXor => MetamethodBits::BXOR,
        Shl => MetamethodBits::SHL,
        Shr => MetamethodBits::SHR,
        None => unreachable!("an arithmetic kind"),
    }
}

/// Integer arithmetic: on `I64`s, which wrap, when `wide` or an operand is
/// one; else on `I32`s, exiting where the result leaves i32.
fn int_arith(b: &mut Builder<'_, '_>, kind: ArithKind, x: Val, y: Val, wide: bool) -> Val {
    use ArithKind::*;
    if wide || b.ty(x).rep == Rep::I64 || b.ty(y).rep == Rep::I64 {
        let (x, y) = (b.as_i64(x), b.as_i64(y));
        let op = match kind {
            Add => Op::LAdd,
            Sub => Op::LSub,
            Mul => Op::LMul,
            BAnd => Op::LAnd,
            BOr => Op::LOr,
            BXor => Op::LXor,
            Shl => Op::LShl,
            Shr => Op::LShr,
            // A zero divisor raises in the interpreter.
            Mod | IDiv => {
                let op = if kind == Mod {
                    Op::LModFloor
                } else {
                    Op::LDivFloor
                };
                let i = b.ins_exit(op, &[x, y], ExitTag::Slow);
                return b.f.result(i);
            }
            Div | Pow | None => unreachable!("{kind:?} on integers"),
        };
        return b.ins(op, &[x, y]);
    }
    let (op, exits) = match kind {
        Add => (Op::IAdd, true),
        Sub => (Op::ISub, true),
        Mul => (Op::IMul, true),
        Mod => (Op::IModFloor, true),
        IDiv => (Op::IDivFloor, true),
        BAnd => (Op::IAnd, false),
        BOr => (Op::IOr, false),
        BXor => (Op::IXor, false),
        Shl => (Op::IShl, true),
        Shr => (Op::IShr, true),
        Div | Pow | None => unreachable!("{kind:?} on integers"),
    };
    if exits {
        let i = b.ins_exit(op, &[x, y], ExitTag::Overflow);
        b.f.result(i)
    } else {
        b.ins(op, &[x, y])
    }
}

/// Float arithmetic; `^` by the constant 0 or 1 folds. Not 2: libm's `pow`,
/// which the interpreter calls, differs from `x * x` in the last bit.
fn float_arith(b: &mut Builder<'_, '_>, kind: ArithKind, x: Val, y: Val, r: Operand) -> Val {
    use ArithKind::*;
    match kind {
        Add => b.ins(Op::FAdd, &[x, y]),
        Sub => b.ins(Op::FSub, &[x, y]),
        Mul => b.ins(Op::FMul, &[x, y]),
        Div => b.ins(Op::FDiv, &[x, y]),
        IDiv => b.ins(Op::FIDiv, &[x, y]),
        Mod => b.ins(Op::Helper(HelperId::FMod), &[x, y]),
        // `-0.0` matches `0.0`: `pow(x, -0.0)` is 1 too.
        Pow => match r.konst(b) {
            Some(0.0) => b.kf64(1.0),
            Some(1.0) => x,
            _ => b.ins(Op::Helper(HelperId::FPow), &[x, y]),
        },
        _ => unreachable!("{kind:?} on floats"),
    }
}

/// Call the metamethod `f` with `args` (5.4) and continue in its resume
/// block with its one result. What is live after the instruction is stored
/// first, as for a CALL; a window too small for the header exits to the
/// interpreter, which grows the stack.
pub(super) fn mm_call(b: &mut Builder<'_, '_>, f: Val, args: &[Val]) -> Val {
    let resume = staged_call(b, f, args, 0, 2);
    let first = b.f.insts_of(resume)[0];
    b.f.result(first)
}

/// Call `f` with `args` from a header past the frame's registers, wanting
/// `c - 1` results (landed from `R[land]` when more than two), and continue
/// in the resume block.
pub(super) fn staged_call(b: &mut Builder<'_, '_>, f: Val, args: &[Val], land: u8, c: u8) -> Block {
    let pc = b.pc;
    let s = b.snap_before();
    b.ins_snap(Op::GuardFits(args.len() as u8), &[], s, ExitTag::Slow);
    let insn = b.code(pc);
    let mut live = b.cfg.live_in[pc as usize];
    if let Some(next) = b.cfg.live_in.get(pc as usize + 1) {
        live.union(next);
    }
    if insn.op().branch_sense().is_some() {
        let t = (pc as i64 + 1 + insn.branch_offset() as i64) as usize;
        if let Some(target) = b.cfg.live_in.get(t) {
            live.union(target);
        }
    }
    let mut ops = vec![b.boxed(f)];
    for &a in args {
        ops.push(b.boxed(a));
    }
    // The callee may read the captured registers through their cells.
    let captured: Vec<usize> = b.cfg.captured.iter().collect();
    b.store_regs(captured);
    // The header goes past the frame's registers: only more than two
    // results land in them.
    let mut written = RegSet::EMPTY;
    if c > 3 {
        for r in land as usize..land as usize + c as usize - 1 {
            written.insert(r);
        }
    }
    let op = Op::CallMm {
        pc,
        nargs: args.len() as u8,
        land,
        c,
    };
    b.call(op, &ops, live, written, c)
}

/// The metamethod of `bit` for `mm_args` (its first two), deopting when
/// there is none: the interpreter raises.
fn metamethod(b: &mut Builder<'_, '_>, bit: MetamethodBits, mm_args: &[Val]) -> Val {
    let f = b.ins(Op::Helper(HelperId::Mm(bit.bits())), &mm_args[..2]);
    let none = b.ins(Op::IsType(TypeSet::NIL), &[f]);
    let s = b.snap_before();
    b.ins_snap(Op::GuardFalse, &[none], s, ExitTag::Slow);
    f
}

/// Arms into `join` setting `dst`: the helper's result, or where it fails
/// the result of calling the metamethod of `bit` with `mm_args`.
pub(super) fn helper_arms(
    b: &mut Builder<'_, '_>,
    join: Block,
    dst: u8,
    h: HelperId,
    args: &[Val],
    bit: MetamethodBits,
    mm_args: &[Val],
) {
    let r = b.ins(Op::Helper(h), args);
    let fail = b.ins(Op::KVal(HELPER_FAIL_VAL), &[]);
    let failed = b.ins(Op::SameBits, &[r, fail]);
    if !calls_mm(b) {
        let s = b.snap_before();
        b.ins_snap(Op::GuardFalse, &[failed], s, ExitTag::Mm);
        arm_end(b, join, dst, r);
        return;
    }
    let ok = branch_arm(b, failed);
    cold(b, |b| {
        let f = metamethod(b, bit, mm_args);
        let res = mm_call(b, f, mm_args);
        arm_end(b, join, dst, res);
    });
    b.switch_to(ok);
    arm_end(b, join, dst, r);
}

/// Whether a helper's failure at the instruction calls a metamethod: only
/// once its feedback saw one run, since a call in a loop keeps the loop's
/// loads and guards in it; until then the failure exits and records one.
pub(super) fn calls_mm(b: &Builder<'_, '_>) -> bool {
    b.feedback(b.pc) & fb::MM != 0
}

/// `R[dst] = #R[src]` (7.2): a string's length inline; a table's by helper
/// unless its class has `__len` now; anything else, and `__len`, as the
/// generic helper and metamethod arms do.
fn emit_len(b: &mut Builder<'_, '_>, dst: u8, src: u8) {
    let v = b.reg(src);
    let v = b.boxed(v);
    let set = b.ty(v).set;
    let may = |s: TypeSet| set.is_empty() || set.intersects(s);
    let only = |s: TypeSet| !set.is_empty() && s.contains(set);
    let join = b.new_block();
    if may(TypeSet::STR) {
        let other = (!only(TypeSet::STR)).then(|| arm(b, Op::IsType(TypeSet::STR), v));
        let s = b.ins(Op::Refine(TypeSet::STR), &[v]);
        let p = b.ins(Op::Unbox(Rep::Ptr), &[s]);
        let n = b.ins(Op::StrLenOf, &[p]);
        let n = b.ins(Op::Box, &[n]);
        arm_end(b, join, dst, n);
        match other {
            Some(o) => b.switch_to(o),
            None => {
                enter_join(b, join);
                return;
            }
        }
    }
    if may(TypeSet::TAB) {
        let other = (!only(TypeSet::TAB)).then(|| arm(b, Op::IsType(TypeSet::TAB), v));
        let t = b.ins(Op::Refine(TypeSet::TAB), &[v]);
        let p = b.ins(Op::Unbox(Rep::Ptr), &[t]);
        let has = tables::class_bit(b, p, MetamethodBits::LEN);
        let plain = branch_arm(b, has);
        helper_arms(
            b,
            join,
            dst,
            HelperId::Len,
            &[v],
            MetamethodBits::LEN,
            &[v, v],
        );
        b.switch_to(plain);
        let n = b.ins(Op::Helper(HelperId::RawLen), &[v]);
        arm_end(b, join, dst, n);
        match other {
            Some(o) => b.switch_to(o),
            None => {
                enter_join(b, join);
                return;
            }
        }
    }
    helper_arms(
        b,
        join,
        dst,
        HelperId::Len,
        &[v],
        MetamethodBits::LEN,
        &[v, v],
    );
    enter_join(b, join);
}

/// `dst` = the helper's result, or the metamethod's (`helper_arms`).
fn helper_or_mm(
    b: &mut Builder<'_, '_>,
    dst: u8,
    h: HelperId,
    args: &[Val],
    bit: MetamethodBits,
    mm_args: &[Val],
) {
    let join = b.new_block();
    helper_arms(b, join, dst, h, args, bit, mm_args);
    enter_join(b, join);
}

/// Branch on the test `op` of `v`: continue in the arm where it holds and
/// return the block where it does not.
fn arm(b: &mut Builder<'_, '_>, op: Op, v: Val) -> Block {
    let c = b.ins(op, &[v]);
    branch_arm(b, c)
}

pub(super) fn branch_arm(b: &mut Builder<'_, '_>, c: Val) -> Block {
    let (yes, no) = (b.new_sealed_succ(), b.new_sealed_succ());
    b.br(c, yes, no);
    b.switch_to(yes);
    no
}

/// End an arm of the instruction with `dst = v`, at the join.
pub(super) fn arm_end(b: &mut Builder<'_, '_>, join: Block, dst: u8, v: Val) {
    // The instruction's earlier writes (`SELF`'s receiver) hold in every arm.
    let earlier = b.pending.clone();
    b.set(dst, v);
    b.flush();
    b.jump(join);
    b.pending = earlier;
}

/// Emit `arm` into the current block as a cold arm (A5).
pub(super) fn cold<R>(b: &mut Builder<'_, '_>, arm: impl FnOnce(&mut Builder<'_, '_>) -> R) -> R {
    let was = b.begin_cold();
    let r = arm(b);
    b.end_cold(was);
    r
}

/// Continue after the arms, where `dst` is their join.
pub(super) fn enter_join(b: &mut Builder<'_, '_>, join: Block) {
    b.seal(join);
    b.switch_to(join);
    b.init_slots(join);
}

// --- compares ---------------------------------------------------------------

/// What a compare speculates one operand to be.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum CmpKind {
    Num(NumKind),
    Str,
}

/// The kind a compare operand has for certain.
fn cmp_own(b: &Builder<'_, '_>, v: Val) -> Option<CmpKind> {
    if let Some(k) = own_kind(b, v) {
        return Some(CmpKind::Num(k));
    }
    let t = b.ty(v);
    (t.rep == Rep::Val && !t.set.is_empty() && t.within(TypeSet::STR)).then_some(CmpKind::Str)
}

/// The kind a compare site's byte gives both operands: numbers as
/// `byte_num_kind` has them (the fast paths record nothing, so an empty byte
/// is small integers and floats), or strings only.
fn byte_cmp_kind(byte: u8) -> Option<CmpKind> {
    if byte & fb::KINDS == fb::STR && byte & fb::MM == 0 {
        Some(CmpKind::Str)
    } else {
        byte_num_kind(byte).map(CmpKind::Num)
    }
}

/// The f64 of a number of kind `k`, exact for the compare: an i64 outside
/// ±2^53 exits.
fn cmp_f64(b: &mut Builder<'_, '_>, v: Val, k: NumKind) -> Val {
    match k {
        NumKind::I64 => {
            let x = b.as_i64(v);
            let i = b.ins_exit(Op::LToFExact, &[x], ExitTag::Type);
            b.f.result(i)
        }
        k => Operand::Reg(v).f64(b, k),
    }
}

/// `x cc y` of numbers of these kinds.
fn cmp_num(
    b: &mut Builder<'_, '_>,
    cc: Cc,
    (x, kx): (Val, NumKind),
    (y, ky): (Val, NumKind),
) -> Val {
    use NumKind::*;
    match (kx, ky) {
        (I32, I32) => {
            let (x, y) = (b.as_i32(x), b.as_i32(y));
            b.ins(Op::ICmp(cc), &[x, y])
        }
        (I32 | I64, I32 | I64) => {
            let (x, y) = (b.as_i64(x), b.as_i64(y));
            b.ins(Op::LCmp(cc), &[x, y])
        }
        _ => {
            let (x, y) = (cmp_f64(b, x, kx), cmp_f64(b, y, ky));
            b.ins(Op::FCmp(cc), &[x, y])
        }
    }
}

/// `x < y` (`le`: `<=`) of two strings.
fn cmp_str(b: &mut Builder<'_, '_>, le: bool, x: Val, y: Val) -> Val {
    let (x, y) = (b.guard(x, TypeSet::STR), b.guard(y, TypeSet::STR));
    let (px, py) = (
        b.ins(Op::Unbox(Rep::Ptr), &[x]),
        b.ins(Op::Unbox(Rep::Ptr), &[y]),
    );
    b.ins(if le { Op::StrLe } else { Op::StrLt }, &[px, py])
}

fn emit_cmp_reg(b: &mut Builder<'_, '_>, pc: u32, insn: Instruction, succs: &[Block]) {
    let op = insn.op();
    let generic = insn.generic_op();
    let (x, y) = (b.reg(insn.a()), b.reg(insn.b()));
    let (cc, sense) = match generic {
        BcOp::JLT => (Cc::Lt, true),
        BcOp::JNLT => (Cc::Lt, false),
        BcOp::JLE => (Cc::Le, true),
        BcOp::JNLE => (Cc::Le, false),
        BcOp::JEQ => (Cc::Eq, true),
        BcOp::JNEQ => (Cc::Eq, false),
        _ => unreachable!(),
    };
    let eq = cc == Cc::Eq;
    let byte = b.feedback(pc);
    let (tx, ty) = (b.ty(x), b.ty(y));
    if eq && !tx.set.is_empty() && !ty.set.is_empty() {
        // Values of disjoint types are never equal, but for an integer and
        // a float; `__eq` needs two tables or two userdata.
        let nums = tx.set.intersects(TypeSet::NUM) && ty.set.intersects(TypeSet::NUM);
        if !tx.set.intersects(ty.set) && !nums {
            let c = b.ins(Op::KB1(false), &[]);
            branch(b, c, sense, succs);
            return;
        }
        let by_bits = |t: crate::jit::ir::types::Ty| {
            t.rep == Rep::Val
                && t.within(
                    TypeSet::NIL | TypeSet::BOOL | TypeSet::STR | TypeSet::FUN | TypeSet::THR,
                )
        };
        if by_bits(tx) || by_bits(ty) {
            let (x, y) = (b.boxed(x), b.boxed(y));
            let c = b.ins(Op::SameBits, &[x, y]);
            branch(b, c, sense, succs);
            return;
        }
    }
    // An `_II` form, which exits saw leave i32 when the byte has big integers.
    let form =
        (op != generic && !b.kind_exited(pc)).then_some(CmpKind::Num(if byte & fb::BIGINT != 0 {
            NumKind::I64
        } else {
            NumKind::I32
        }));
    let kind = |b: &Builder<'_, '_>, v| cmp_own(b, v).or(form).or_else(|| byte_cmp_kind(byte));
    let c = match (kind(b, x), kind(b, y)) {
        (Some(CmpKind::Num(kx)), Some(CmpKind::Num(ky))) => cmp_num(b, cc, (x, kx), (y, ky)),
        (Some(CmpKind::Str), Some(CmpKind::Str)) if eq => {
            let (x, y) = (b.boxed(x), b.boxed(y));
            let (x, y) = (b.guard(x, TypeSet::STR), b.guard(y, TypeSet::STR));
            b.ins(Op::SameBits, &[x, y])
        }
        (Some(CmpKind::Str), Some(CmpKind::Str)) => cmp_str(b, cc == Cc::Le, x, y),
        _ if eq => {
            generic_eq(b, x, y, sense, succs);
            return;
        }
        _ => {
            generic_cmp(b, cc, x, y, sense, succs);
            return;
        }
    };
    branch(b, c, sense, succs);
}

/// The targets of a compare's branch: where it goes when equal (or less)
/// and when not.
fn targets(sense: bool, succs: &[Block]) -> (Block, Block) {
    let (fall, target) = (succs[0], succs[1]);
    if sense {
        (target, fall)
    } else {
        (fall, target)
    }
}

/// `x == y` where a kind is unknown, as `eq_slow` decides it: floats by
/// value first (a NaN's bits equal another NaN's), equal bits, an integer
/// and a float, two big integers by value, then `__eq` for two tables or
/// two userdata, which the site's feedback must have seen run to call.
fn generic_eq(b: &mut Builder<'_, '_>, x: Val, y: Val, sense: bool, succs: &[Block]) {
    let (yes, no) = targets(sense, succs);
    let (x, y) = (b.boxed(x), b.boxed(y));
    let byte = b.feedback(b.pc);
    let big = byte & fb::BIGINT != 0;
    // [x float]
    let not_float = all_in(b, TypeSet::FLOAT, &[x]);
    let fx = b.ins(Op::Unbox(Rep::F64), &[x]);
    let other = all_in(b, TypeSet::SMALL | TypeSet::FLOAT, &[y]);
    let fy = b.ins_exit(Op::ToF64, &[y], ExitTag::Type);
    let fy = b.f.result(fy);
    let c = b.ins(Op::FCmp(Cc::Eq), &[fx, fy]);
    b.br(c, yes, no);
    enter_join(b, other);
    big_float_eq(b, big, y, fx, yes, no);
    // [x not float, y float]
    enter_join(b, not_float);
    let neither = all_in(b, TypeSet::FLOAT, &[y]);
    let fy = b.ins(Op::Unbox(Rep::F64), &[y]);
    let other = all_in(b, TypeSet::SMALL, &[x]);
    let ix = b.ins(Op::Unbox(Rep::I32), &[x]);
    let fx = b.ins(Op::IToF, &[ix]);
    let c = b.ins(Op::FCmp(Cc::Eq), &[fx, fy]);
    b.br(c, yes, no);
    enter_join(b, other);
    big_float_eq(b, big, x, fy, yes, no);
    // [neither float]
    enter_join(b, neither);
    let same = b.ins(Op::SameBits, &[x, y]);
    let differ = b.new_sealed_succ();
    b.br(same, yes, differ);
    b.switch_to(differ);
    if big {
        let other = all_in(b, TypeSet::BIGINT, &[x, y]);
        let (ix, iy) = (
            b.ins(Op::Unbox(Rep::I64), &[x]),
            b.ins(Op::Unbox(Rep::I64), &[y]),
        );
        let c = b.ins(Op::LCmp(Cc::Eq), &[ix, iy]);
        b.br(c, yes, no);
        enter_join(b, other);
    }
    let not_tables = all_in(b, TypeSet::TAB, &[x, y]);
    if calls_mm(b) {
        cold(b, |b| eq_mm_arm(b, x, y, yes, no));
    } else {
        let (px, py) = (
            b.ins(Op::Unbox(Rep::Ptr), &[x]),
            b.ins(Op::Unbox(Rep::Ptr), &[y]),
        );
        let has = tables::class_bit(b, px, MetamethodBits::EQ);
        let s = b.snap_before();
        b.ins_snap(Op::GuardFalse, &[has], s, ExitTag::Mm);
        let has = tables::class_bit(b, py, MetamethodBits::EQ);
        b.ins_snap(Op::GuardFalse, &[has], s, ExitTag::Mm);
        b.jump(no);
    }
    enter_join(b, not_tables);
    let not_udata = all_in(b, TypeSet::UDATA, &[x, y]);
    if calls_mm(b) {
        cold(b, |b| eq_mm_arm(b, x, y, yes, no));
    } else {
        // Whether `__eq` runs is the interpreter's to find out.
        b.deopt(ExitTag::Mm);
    }
    enter_join(b, not_udata);
    b.jump(no);
}

/// `f == v` for a float `f` and a value `v` known not to be a small integer
/// or a float: equal only as a big integer of the same value.
fn big_float_eq(b: &mut Builder<'_, '_>, big: bool, v: Val, f: Val, yes: Block, no: Block) {
    if !big {
        b.jump(no);
        return;
    }
    let not_big = all_in(b, TypeSet::BIGINT, &[v]);
    let i = b.ins(Op::Unbox(Rep::I64), &[v]);
    let g = b.ins_exit(Op::LToFExact, &[i], ExitTag::Slow);
    let g = b.f.result(g);
    let c = b.ins(Op::FCmp(Cc::Eq), &[g, f]);
    b.br(c, yes, no);
    enter_join(b, not_big);
    b.jump(no);
}

/// `__eq` of two tables or two userdata with different bits: none means
/// unequal, else its result's truth.
fn eq_mm_arm(b: &mut Builder<'_, '_>, x: Val, y: Val, yes: Block, no: Block) {
    let f = b.ins(Op::Helper(HelperId::Mm(MetamethodBits::EQ.bits())), &[x, y]);
    let none = b.ins(Op::IsType(TypeSet::NIL), &[f]);
    let call = b.new_sealed_succ();
    b.br(none, no, call);
    b.switch_to(call);
    let res = mm_call(b, f, &[x, y]);
    let falsy = b.ins(Op::IsFalsy, &[res]);
    b.br(falsy, no, yes);
}

/// `x < y` (or `<=`) where a kind is unknown: numbers and strings behind
/// tests of what the site's feedback saw; the metamethod arm once it saw one
/// run; anything else raises in the interpreter.
fn generic_cmp(b: &mut Builder<'_, '_>, cc: Cc, x: Val, y: Val, sense: bool, succs: &[Block]) {
    let (yes, no) = targets(sense, succs);
    let (x, y) = (b.boxed(x), b.boxed(y));
    let byte = b.feedback(b.pc);
    let le = cc == Cc::Le;
    if byte & (fb::SMALL | fb::FLOAT) != 0 || byte & fb::KINDS == 0 {
        let other = all_in(b, TypeSet::SMALL | TypeSet::FLOAT, &[x, y]);
        let (fx, fy) = (cmp_f64(b, x, NumKind::Num), cmp_f64(b, y, NumKind::Num));
        let c = b.ins(Op::FCmp(cc), &[fx, fy]);
        b.br(c, yes, no);
        enter_join(b, other);
    }
    if byte & fb::STR != 0 {
        let other = all_in(b, TypeSet::STR, &[x, y]);
        let c = cmp_str(b, le, x, y);
        b.br(c, yes, no);
        enter_join(b, other);
    }
    if byte & fb::BIGINT != 0 {
        let other = all_in(b, TypeSet::INT, &[x, y]);
        let c = cmp_num(b, cc, (x, NumKind::I64), (y, NumKind::I64));
        b.br(c, yes, no);
        enter_join(b, other);
    }
    if !calls_mm(b) {
        // Mixed big integers and floats, or what raises.
        b.deopt(ExitTag::Slow);
        return;
    }
    let bit = if le {
        MetamethodBits::LE
    } else {
        MetamethodBits::LT
    };
    cold(b, |b| {
        let f = metamethod(b, bit, &[x, y]);
        let res = mm_call(b, f, &[x, y]);
        let falsy = b.ins(Op::IsFalsy, &[res]);
        b.br(falsy, no, yes);
    });
}

fn emit_cmp_imm(b: &mut Builder<'_, '_>, pc: u32, insn: Instruction, succs: &[Block]) {
    let op = insn.op();
    let generic = insn.generic_op();
    let v = b.reg(insn.a());
    let k = insn.cmp_imm_int();
    let (cc, swap, sense) = match generic {
        BcOp::JLTI => (Cc::Lt, false, true),
        BcOp::JNLTI => (Cc::Lt, false, false),
        BcOp::JLEI => (Cc::Le, false, true),
        BcOp::JNLEI => (Cc::Le, false, false),
        BcOp::JGTI => (Cc::Lt, true, true),
        BcOp::JNGTI => (Cc::Lt, true, false),
        BcOp::JGEI => (Cc::Le, true, true),
        BcOp::JNGEI => (Cc::Le, true, false),
        _ => unreachable!(),
    };
    // `k <cc> v` is `v <swapped cc> k`.
    let cc = if swap { cc.swap() } else { cc };
    let byte = b.feedback(pc);
    // The fast path takes big integers too, unrecorded: an empty byte is any
    // number, which the `Num` path's conversion exits on when big.
    let form = (op != generic && !b.kind_exited(pc)).then_some(NumKind::F64);
    let kind = own_kind(b, v).or(form).or_else(|| byte_num_kind(byte));
    let imm = |b: &mut Builder<'_, '_>, kind: NumKind| match kind {
        NumKind::I32 => b.ki32(k as i32),
        NumKind::I64 => b.ins(Op::KI64(k), &[]),
        _ => b.kf64(k as f64),
    };
    let c = match kind {
        Some(kind) => {
            let x = match kind {
                NumKind::I32 => b.as_i32(v),
                NumKind::I64 => b.as_i64(v),
                kind => cmp_f64(b, v, kind),
            };
            let kk = imm(b, kind);
            let op = match kind {
                NumKind::I32 => Op::ICmp(cc),
                NumKind::I64 => Op::LCmp(cc),
                _ => Op::FCmp(cc),
            };
            b.ins(op, &[x, kk])
        }
        None => {
            let (yes, no) = targets(sense, succs);
            let x = b.boxed(v);
            let other = all_in(b, TypeSet::SMALL | TypeSet::FLOAT, &[x]);
            let fx = cmp_f64(b, x, NumKind::Num);
            let kk = imm(b, NumKind::F64);
            let c = b.ins(Op::FCmp(cc), &[fx, kk]);
            b.br(c, yes, no);
            enter_join(b, other);
            if byte & fb::BIGINT != 0 {
                let other = all_in(b, TypeSet::BIGINT, &[x]);
                let ix = b.ins(Op::Unbox(Rep::I64), &[x]);
                let kk = imm(b, NumKind::I64);
                let c = b.ins(Op::LCmp(cc), &[ix, kk]);
                b.br(c, yes, no);
                enter_join(b, other);
            }
            if !calls_mm(b) {
                b.deopt(ExitTag::Slow);
                return;
            }
            let le = cc == Cc::Le || cc == Cc::Ge;
            let bit = if le {
                MetamethodBits::LE
            } else {
                MetamethodBits::LT
            };
            // The metamethod takes the operands in source order.
            cold(b, |b| {
                let kv = b.konst(Value::small(k as i32));
                let args = if swap { [kv, x] } else { [x, kv] };
                let f = metamethod(b, bit, &args);
                let res = mm_call(b, f, &args);
                let falsy = b.ins(Op::IsFalsy, &[res]);
                b.br(falsy, no, yes);
            });
            return;
        }
    };
    branch(b, c, sense, succs);
}

fn emit_eqi(b: &mut Builder<'_, '_>, _pc: u32, insn: Instruction, succs: &[Block]) {
    let v = b.reg(insn.a());
    let k = insn.cmp_imm_int();
    let sense = insn.op() == BcOp::JEQI;
    let c = match own_kind(b, v) {
        Some(kind @ (NumKind::F64 | NumKind::Num)) => {
            let x = cmp_f64(b, v, kind);
            let kk = b.kf64(k as f64);
            b.ins(Op::FCmp(Cc::Eq), &[x, kk])
        }
        Some(NumKind::I64) => {
            let x = b.as_i64(v);
            let kk = b.ins(Op::KI64(k), &[]);
            b.ins(Op::LCmp(Cc::Eq), &[x, kk])
        }
        Some(NumKind::I32) => {
            let x = b.as_i32(v);
            let kk = b.ki32(k as i32);
            b.ins(Op::ICmp(Cc::Eq), &[x, kk])
        }
        // A float compares by value; anything else equals the small
        // constant only as its bits (a big integer is never small).
        None => {
            let v = b.boxed(v);
            let float = b.ins(Op::IsType(TypeSet::FLOAT), &[v]);
            let (ff, other) = (b.new_sealed_succ(), b.new_sealed_succ());
            b.br(float, ff, other);
            b.switch_to(ff);
            let x = b.ins(Op::Unbox(Rep::F64), &[v]);
            let kk = b.kf64(k as f64);
            let c = b.ins(Op::FCmp(Cc::Eq), &[x, kk]);
            branch(b, c, sense, succs);
            b.switch_to(other);
            let kk = b.konst(Value::small(k as i32));
            b.ins(Op::SameBits, &[v, kk])
        }
    };
    branch(b, c, sense, succs);
}

// --- calls --------------------------------------------------------------------

fn emit_call(b: &mut Builder<'_, '_>, pc: u32, insn: Instruction) {
    let op = insn.op();
    let fused = matches!(op, BcOp::CALLS | BcOp::CALLS_R0 | BcOp::CALLS_R1);
    let callee = if fused {
        b.reg(insn.imm() as u8)
    } else {
        callee_at(b, pc, insn.a())
    };
    match intrinsics::emit(b, pc, insn, callee) {
        Emitted::No => {
            if !super::inline::try_inline(b, pc, insn, callee) {
                call(b, pc, insn, callee)
            }
        }
        Emitted::Done => {}
        Emitted::Fallback(slow, f) => {
            // The builtin's results join the inline code's.
            let join = b.new_block();
            b.flush();
            b.jump(join);
            b.switch_to(slow);
            call(b, pc, insn, f);
            b.flush();
            b.jump(join);
            enter_join(b, join);
        }
    }
}

/// `R[a]`, the callee of the call at `pc`.
fn callee_at(b: &mut Builder<'_, '_>, pc: u32, a: u8) -> Val {
    match b.callee_hint {
        Some((at, f, _)) if at == pc => f,
        _ => b.reg(a),
    }
}

/// A native `callee` as a constant, which lowering calls through
/// `jit_call_native` without testing it (I1): one `value_now` takes for a
/// constant is guarded by identity.
fn native_callee(b: &mut Builder<'_, '_>, pc: u32, callee: Val) -> Val {
    if matches!(b.f.def_op(callee), Some(Op::KObj(_))) {
        return callee;
    }
    let guess = b.feedback(pc) & fb::CONST_MISS == 0;
    let Some(f) = intrinsics::value_now(b, callee, guess)
        .and_then(|v| v.get_function())
        .filter(|f| f.as_native().is_some())
    else {
        return callee;
    };
    let k = b.konst(Value::function(f));
    let cv = b.boxed(callee);
    b.ins_exit(Op::GuardSame, &[cv, k], ExitTag::Const);
    k
}

/// The CALL at `pc` as a call of `callee` (5.4).
fn call(b: &mut Builder<'_, '_>, pc: u32, insn: Instruction, callee: Val) {
    let (a, nb, c) = (insn.a(), insn.b(), insn.c());
    let callee = native_callee(b, pc, callee);
    let live = b
        .cfg
        .live_in
        .get(pc as usize + 1)
        .copied()
        .unwrap_or_default();
    // The arguments in the callee's slots (up to `top`: those before the
    // ones a call or VARARG landed there), and the captured registers, which
    // it may read through their cells.
    let mut stored: Vec<usize> = b.cfg.captured.iter().collect();
    let end = if nb != 0 {
        a as usize + 4 + nb as usize - 1
    } else {
        b.cfg.nregs
    };
    stored.extend(a as usize + 4..end);
    b.store_regs(stored);
    let cv = b.boxed(callee);
    b.push(Op::Store(a), &[cv]);
    // The callee's frame and its results take every register from `a` up.
    let mut written = RegSet::EMPTY;
    for r in a as usize..b.cfg.nregs {
        written.insert(r);
    }
    // Arguments up to `top`: their count from it.
    let (nargs, count) = if nb == 0 {
        let n = b.ins(Op::Helper(HelperId::TopArgs(a)), &[]);
        (u8::MAX, vec![n])
    } else {
        (nb - 1, vec![])
    };
    let resume = b.call(Op::Call { a, nargs, c, pc }, &count, live, written, c);
    let first = b.f.insts_of(resume)[0];
    let rs: Vec<Val> = b.f.results(first).collect();
    for (k, &v) in rs.iter().enumerate() {
        b.set(a + k as u8, v);
    }
    b.note_def(&rs, Some(resume));
}

// --- loops --------------------------------------------------------------------

fn emit_forprep(b: &mut Builder<'_, '_>, pc: u32, insn: Instruction, succs: &[Block]) {
    let a = insn.a();
    let (fall, skip_to) = (succs[0], succs[1]);
    let (init, limit, step) = (b.reg(a), b.reg(a + 1), b.reg(a + 2));
    let at = (pc as i64 + insn.branch_offset() as i64) as u32;
    // An integer loop's limit is a float when known to be or seen so.
    let lt = b.ty(limit);
    let int_typed =
        matches!(lt.rep, Rep::I32 | Rep::I64) || (!lt.set.is_empty() && lt.within(TypeSet::INT));
    let float_limit = is_float(b, limit) || (!int_typed && b.feedback(pc) & fb::FLOAT_LIMIT != 0);
    match loop_kind(b, at) {
        Some(BcOp::FORLOOP_I) => {
            let (i, s) = (b.as_i32(init), b.as_i32(step));
            let l = if float_limit {
                // The interpreter's `for_limit`: floored up, ceiled down.
                let x = any_f64(b, limit);
                let r = match b.f.def_op(s) {
                    Some(Op::KI32(n)) if n > 0 => b.ins(Op::FFloor, &[x]),
                    Some(Op::KI32(_)) => b.ins(Op::FCeil, &[x]),
                    _ => {
                        let zero = b.ki32(0);
                        let pos = b.ins(Op::ICmp(Cc::Gt), &[s, zero]);
                        let fl = b.ins(Op::FFloor, &[x]);
                        let ce = b.ins(Op::FCeil, &[x]);
                        b.ins(Op::Select, &[pos, fl, ce])
                    }
                };
                let li = b.ins_exit(Op::FToIExact, &[r], ExitTag::Type);
                b.f.result(li)
            } else {
                b.as_i32(limit)
            };
            let zero = b.ki32(0);
            // A zero step raises in the interpreter.
            let nz = b.ins(Op::ICmp(Cc::Ne), &[s, zero]);
            let snap = b.snap_before();
            b.ins_snap(Op::GuardTrue, &[nz], snap, ExitTag::Slow);
            let last = match b.f.def_op(s) {
                Some(Op::KI32(1 | -1)) => l,
                _ => {
                    // last = init + (limit - init) / step * step, exact when the loop runs.
                    let (i64_, l64, s64) = (
                        b.ins(Op::IToL, &[i]),
                        b.ins(Op::IToL, &[l]),
                        b.ins(Op::IToL, &[s]),
                    );
                    let span = b.ins(Op::LSub, &[l64, i64_]);
                    let q = b.ins_exit(Op::LDivFloor, &[span, s64], ExitTag::Slow);
                    let q = b.f.result(q);
                    let off = b.ins(Op::LMul, &[q, s64]);
                    let last = b.ins(Op::LAdd, &[i64_, off]);
                    let li = b.ins_exit(Op::LToI, &[last], ExitTag::Type);
                    b.f.result(li)
                }
            };
            // Skip when the loop does not run: past the limit in the step's direction.
            let skip = match b.f.def_op(s) {
                Some(Op::KI32(n)) if n > 0 => b.ins(Op::ICmp(Cc::Gt), &[i, l]),
                Some(Op::KI32(_)) => b.ins(Op::ICmp(Cc::Lt), &[i, l]),
                _ => {
                    let pos = b.ins(Op::ICmp(Cc::Gt), &[s, zero]);
                    let up = b.ins(Op::ICmp(Cc::Gt), &[i, l]);
                    let down = b.ins(Op::ICmp(Cc::Lt), &[i, l]);
                    b.ins(Op::Select, &[pos, up, down])
                }
            };
            b.set(a, last);
            b.set(a + 1, s);
            b.set(a + 2, i);
            b.set(a + 3, i);
            b.flush();
            b.br(skip, skip_to, fall);
        }
        Some(BcOp::FORLOOP_F) => {
            let (i, l, s) = (any_f64(b, init), any_f64(b, limit), any_f64(b, step));
            let zero = b.kf64(0.0);
            let nz = b.ins(Op::FCmp(Cc::Ne), &[s, zero]);
            let snap = b.snap_before();
            b.ins_snap(Op::GuardTrue, &[nz], snap, ExitTag::Slow);
            // `0 < s ? lim < i : i < lim`.
            let skip = match const_f64(b, s) {
                Some(x) if x > 0.0 => b.ins(Op::FCmp(Cc::Lt), &[l, i]),
                Some(_) => b.ins(Op::FCmp(Cc::Lt), &[i, l]),
                None => {
                    let pos = b.ins(Op::FCmp(Cc::Gt), &[s, zero]);
                    let up = b.ins(Op::FCmp(Cc::Lt), &[l, i]);
                    let down = b.ins(Op::FCmp(Cc::Lt), &[i, l]);
                    b.ins(Op::Select, &[pos, up, down])
                }
            };
            b.set(a, l);
            b.set(a + 1, s);
            b.set(a + 2, i);
            b.set(a + 3, i);
            b.flush();
            b.br(skip, skip_to, fall);
        }
        // A loop whose values did not all fit, or that never ran: integers,
        // with the reference's unsigned iteration count.
        Some(BcOp::FORLOOP) => {
            let (i, s) = (b.as_i64(init), b.as_i64(step));
            let zero = b.ins(Op::KI64(0), &[]);
            let nz = b.ins(Op::LCmp(Cc::Ne), &[s, zero]);
            let snap = b.snap_before();
            b.ins_snap(Op::GuardTrue, &[nz], snap, ExitTag::Slow);
            let pos = b.ins(Op::LCmp(Cc::Gt), &[s, zero]);
            let (l, outside) = if float_limit {
                // `for_limit`: floored up, ceiled down, clamped to i64 with
                // the loop skipped when the limit lies outside it on the
                // side the loop moves away from; NaN counts as too small.
                let x = any_f64(b, limit);
                let nan = b.ins(Op::FCmp(Cc::Ne), &[x, x]);
                let fl = b.ins(Op::FFloor, &[x]);
                let ce = b.ins(Op::FCeil, &[x]);
                let r = b.ins(Op::Select, &[pos, fl, ce]);
                let l = b.ins(Op::FToL, &[r]);
                let lmin = b.ins(Op::KI64(i64::MIN), &[]);
                let l = b.ins(Op::Select, &[nan, lmin, l]);
                let (min, max) = (b.kf64(-(2f64.powi(63))), b.kf64(2f64.powi(63)));
                let t = b.ins(Op::KB1(true), &[]);
                let below = b.ins(Op::FCmp(Cc::Lt), &[r, min]);
                let below = b.ins(Op::Select, &[nan, t, below]);
                let above = b.ins(Op::FCmp(Cc::Ge), &[r, max]);
                (l, Some((below, above)))
            } else {
                (b.as_i64(limit), None)
            };
            let mut up = b.ins(Op::LCmp(Cc::Gt), &[i, l]);
            let mut down = b.ins(Op::LCmp(Cc::Lt), &[i, l]);
            if let Some((below, above)) = outside {
                let t = b.ins(Op::KB1(true), &[]);
                up = b.ins(Op::Select, &[below, t, up]);
                down = b.ins(Op::Select, &[above, t, down]);
            }
            let skip = b.ins(Op::Select, &[pos, up, down]);
            let last = match const_i64(b, s) {
                Some(1 | -1) => l,
                _ => {
                    let (fwd, back) = (b.ins(Op::LSub, &[l, i]), b.ins(Op::LSub, &[i, l]));
                    let span = b.ins(Op::Select, &[pos, fwd, back]);
                    let neg = b.ins(Op::LNeg, &[s]);
                    let div = b.ins(Op::Select, &[pos, s, neg]);
                    let count = b.ins(Op::LUDiv, &[span, div]);
                    let off = b.ins(Op::LMul, &[count, s]);
                    b.ins(Op::LAdd, &[i, off])
                }
            };
            b.set(a, last);
            b.set(a + 1, s);
            b.set(a + 2, i);
            b.set(a + 3, i);
            b.flush();
            b.br(skip, skip_to, fall);
        }
        _ => b.deopt(ExitTag::Unsupported),
    }
}

/// Whether `v` is known to be a float.
fn is_float(b: &Builder<'_, '_>, v: Val) -> bool {
    let t = b.ty(v);
    t.rep == Rep::F64 || (t.within(TypeSet::FLOAT) && !t.set.is_empty())
}

fn const_i64(b: &Builder<'_, '_>, v: Val) -> Option<i64> {
    match b.f.def_op(v) {
        Some(Op::KI64(n)) => Some(n),
        Some(Op::KI32(n)) => Some(n as i64),
        Some(Op::IToL) => const_i64(b, b.f.args(b.f.def_inst(v).unwrap())[0]),
        _ => None,
    }
}

fn const_f64(b: &Builder<'_, '_>, v: Val) -> Option<f64> {
    match b.f.def_op(v) {
        Some(Op::KF64(bits)) => Some(f64::from_bits(bits)),
        Some(Op::IToF) => match b.f.def_op(b.f.args(b.f.def_inst(v).unwrap())[0]) {
            Some(Op::KI32(n)) => Some(n as f64),
            _ => None,
        },
        _ => None,
    }
}

fn emit_forloop_i(b: &mut Builder<'_, '_>, a: u8, succs: &[Block]) {
    let (exit, back) = (succs[0], succs[1]);
    let (last, step, idx) = (b.reg(a), b.reg(a + 1), b.reg(a + 2));
    let (last, step, idx) = (b.as_i32(last), b.as_i32(step), b.as_i32(idx));
    let go = b.ins(Op::ICmp(Cc::Ne), &[idx, last]);
    let edge = b.new_sealed_succ();
    b.br(go, edge, exit);
    b.switch_to(edge);
    // `idx` walks to `last` exactly, so the step stays in range.
    let next = b.ins(Op::IAddNo, &[idx, step]);
    b.set(a + 2, next);
    b.set(a + 3, next);
    b.flush();
    b.jump(back);
}

fn emit_forloop_l(b: &mut Builder<'_, '_>, a: u8, succs: &[Block]) {
    let (exit, back) = (succs[0], succs[1]);
    let (last, step, idx) = (b.reg(a), b.reg(a + 1), b.reg(a + 2));
    let (last, step, idx) = (b.as_i64(last), b.as_i64(step), b.as_i64(idx));
    let go = b.ins(Op::LCmp(Cc::Ne), &[idx, last]);
    let edge = b.new_sealed_succ();
    b.br(go, edge, exit);
    b.switch_to(edge);
    // As for `FORLOOP_I`: `idx` walks to `last` exactly.
    let next = b.ins(Op::LAdd, &[idx, step]);
    b.set(a + 2, next);
    b.set(a + 3, next);
    b.flush();
    b.jump(back);
}

fn emit_forloop_f(b: &mut Builder<'_, '_>, a: u8, succs: &[Block]) {
    let (exit, back) = (succs[0], succs[1]);
    let (lim, step, idx) = (b.reg(a), b.reg(a + 1), b.reg(a + 2));
    let (lim, step, idx) = (b.as_f64(lim), b.as_f64(step), b.as_f64(idx));
    let next = b.ins(Op::FAdd, &[idx, step]);
    let go = match const_f64(b, step) {
        Some(x) if x > 0.0 => b.ins(Op::FCmp(Cc::Le), &[next, lim]),
        Some(_) => b.ins(Op::FCmp(Cc::Le), &[lim, next]),
        None => {
            let zero = b.kf64(0.0);
            let pos = b.ins(Op::FCmp(Cc::Gt), &[step, zero]);
            let up = b.ins(Op::FCmp(Cc::Le), &[next, lim]);
            let down = b.ins(Op::FCmp(Cc::Le), &[lim, next]);
            b.ins(Op::Select, &[pos, up, down])
        }
    };
    let edge = b.new_sealed_succ();
    b.br(go, edge, exit);
    b.switch_to(edge);
    b.set(a + 2, next);
    b.set(a + 3, next);
    b.flush();
    b.jump(back);
}
