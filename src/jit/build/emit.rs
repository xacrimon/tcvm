//! Per-opcode emission (`jit-design.md` 7, Appendix D): each instruction's
//! guards from its quickened form and feedback byte, its operation, and its
//! result writes. What milestone 1 does not compile is a deopt.

use crate::env::value::Value;
use crate::instruction::{ArithKind, Family, Instruction, Op as BcOp};
use crate::jit::build::Builder;
use crate::jit::feedback as fb;
use crate::jit::ir::ops::{Cc, ExitTag, HELPER_FAIL, HelperId, Op};
use crate::jit::ir::types::{Rep, TypeSet};
use crate::jit::ir::{Block, Val};

/// How an arithmetic site compiles.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mode {
    I32,
    I64,
    /// Floats, with each operand a small integer (converted) or a float.
    F64 {
        l_small: bool,
        r_small: bool,
    },
    Deopt,
}

fn arith_mode(b: &Builder<'_, '_>, pc: u32) -> Mode {
    let insn = b.code(pc);
    let op = insn.op();
    let generic = insn.generic_op();
    let info = generic.info();
    let kind = info.kind;
    let byte = b.feedback(pc);
    let wide = byte & (fb::OVERFLOW | fb::BIGINT) != 0;
    let float_result = matches!(kind, ArithKind::Div | ArithKind::Pow);
    if matches!(op, BcOp::ARITH_MM | BcOp::ARITH_MM_R | BcOp::ARITH_MMI) {
        return Mode::Deopt;
    }
    if op != generic {
        let forms = &info.forms;
        let imm = matches!(info.family, Family::ImmArith | Family::ImmBit);
        let idx = forms.iter().position(|f| *f == Some(op));
        // A site keeps a form only while it sees the form's kinds (6.5).
        return match (imm, idx) {
            (false, Some(0)) | (true, Some(0)) if float_result => Mode::F64 {
                l_small: true,
                r_small: true,
            },
            (false, Some(0)) | (true, Some(0)) => {
                if wide {
                    Mode::I64
                } else {
                    Mode::I32
                }
            }
            (false, Some(1)) | (true, Some(1)) => Mode::F64 {
                l_small: false,
                r_small: false,
            },
            (false, Some(2)) => Mode::F64 {
                l_small: true,
                r_small: false,
            },
            (false, Some(3)) => Mode::F64 {
                l_small: false,
                r_small: true,
            },
            // A small register with an immediate that makes the result a float.
            (true, Some(2)) => {
                if info.reversed {
                    Mode::F64 {
                        l_small: false,
                        r_small: true,
                    }
                } else {
                    Mode::F64 {
                        l_small: true,
                        r_small: false,
                    }
                }
            }
            _ => Mode::Deopt,
        };
    }
    let (_, locked) = insn.adaptive();
    let ints_only = byte & fb::KINDS != 0 && byte & !(fb::INT | fb::OVERFLOW) == 0;
    if !locked && ints_only && !float_result {
        return Mode::I64;
    }
    Mode::Deopt
}

/// Whether the instruction at `pc` compiles to an unconditional deopt.
pub(super) fn always_deopts(b: &Builder<'_, '_>, pc: u32) -> bool {
    let insn = b.code(pc);
    use BcOp::*;
    match insn.op() {
        MOVE | LOAD | LOADI | LOADNIL | LFALSESKIP | GETUPVAL | NOP | FUNC | LOOP | JMP | NOT
        | ERRNNIL | JT | JF | JTSET | JFSET | RETURN0 | RETURN1 | FORLOOP_I | FORLOOP_F
        | FORLOOP | JEQS | JNEQS => false,
        VARARGPREP => b.proto.needs_vararg_table,
        RETURN => insn.b() == 0,
        CALL | CALL_R0 | CALL_R1 | CALLS | CALLS_R0 | CALLS_R1 => insn.b() == 0,
        FORPREP => false,
        UNM | BNOT => false,
        op => match op.info().family {
            Family::RegArith | Family::RegBit | Family::ImmArith | Family::ImmBit => {
                arith_mode(b, pc) == Mode::Deopt
            }
            Family::CmpReg | Family::CmpImm => false,
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
        VARARGPREP => {}
        JMP => b.jump(succs[0]),
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
        UNM => emit_unm(b, a, rb),
        BNOT => emit_bnot(b, a, rb),
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
        RETURN0 => {
            b.terminate(Op::Return { a: 0, n: 0 }, &[], &[], None);
        }
        RETURN1 => {
            store_results(b, a, 1);
            b.terminate(Op::Return { a, n: 1 }, &[], &[], None);
        }
        RETURN => {
            let s = b.snap_before();
            b.ins_snap(Op::GuardNoClose, &[], s, ExitTag::Unsupported);
            let n = rb - 1;
            store_results(b, a, n);
            b.terminate(Op::Return { a, n }, &[], &[], None);
        }
        CALL | CALL_R0 | CALL_R1 | CALLS | CALLS_R0 | CALLS_R1 => emit_call(b, pc, insn),
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
            Family::CmpImm => emit_cmp_imm(b, pc, insn, succs),
            _ => b.deopt(ExitTag::Unsupported),
        },
    }
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
    let mode = arith_mode(b, pc);
    let generic = insn.generic_op();
    let info = generic.info();
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
    use ArithKind::*;
    let res = match mode {
        Mode::Deopt => {
            b.deopt(ExitTag::Unsupported);
            return;
        }
        Mode::I32 => {
            let (x, y) = (l.i32(b), r.i32(b));
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
                _ => unreachable!("{kind:?} in i32 mode"),
            };
            if exits {
                let i = b.ins_exit(op, &[x, y], ExitTag::Overflow);
                b.f.result(i)
            } else {
                b.ins(op, &[x, y])
            }
        }
        Mode::I64 => {
            let (x, y) = (l.i64(b), r.i64(b));
            let (op, exits) = match kind {
                Add => (Op::LAdd, false),
                Sub => (Op::LSub, false),
                Mul => (Op::LMul, false),
                Mod => (Op::LModFloor, true),
                IDiv => (Op::LDivFloor, true),
                BAnd => (Op::LAnd, false),
                BOr => (Op::LOr, false),
                BXor => (Op::LXor, false),
                Shl => (Op::LShl, false),
                Shr => (Op::LShr, false),
                _ => unreachable!("{kind:?} in i64 mode"),
            };
            if exits {
                let i = b.ins_exit(op, &[x, y], ExitTag::Slow);
                b.f.result(i)
            } else {
                b.ins(op, &[x, y])
            }
        }
        Mode::F64 { l_small, r_small } => {
            let x = l.f64(b, l_small);
            let y = r.f64(b, r_small);
            match kind {
                Add => b.ins(Op::FAdd, &[x, y]),
                Sub => b.ins(Op::FSub, &[x, y]),
                Mul => b.ins(Op::FMul, &[x, y]),
                Div => b.ins(Op::FDiv, &[x, y]),
                IDiv => b.ins(Op::FIDiv, &[x, y]),
                Mod => b.ins(Op::Helper(HelperId::FMod), &[x, y]),
                Pow => b.ins(Op::Helper(HelperId::FPow), &[x, y]),
                _ => {
                    b.deopt(ExitTag::Unsupported);
                    return;
                }
            }
        }
    };
    b.set(dst, res);
}

/// An arithmetic operand: a register's value or the instruction's immediate.
#[derive(Clone, Copy)]
enum Operand {
    Reg(Val),
    Int(i64),
    Float(f64),
}

impl Operand {
    fn i32(self, b: &mut Builder<'_, '_>) -> Val {
        match self {
            Operand::Reg(v) => b.as_i32(v),
            Operand::Int(n) => b.ki32(n as i32),
            Operand::Float(x) => b.ki32(x as i32),
        }
    }

    fn i64(self, b: &mut Builder<'_, '_>) -> Val {
        match self {
            Operand::Reg(v) => b.as_i64(v),
            Operand::Int(n) => b.ins(Op::KI64(n), &[]),
            Operand::Float(x) => b.ins(Op::KI64(x as i64), &[]),
        }
    }

    fn f64(self, b: &mut Builder<'_, '_>, small: bool) -> Val {
        match self {
            Operand::Reg(v) => {
                if small {
                    let i = b.as_i32(v);
                    b.ins(Op::IToF, &[i])
                } else {
                    b.as_f64(v)
                }
            }
            Operand::Int(n) => b.kf64(n as f64),
            Operand::Float(x) => b.kf64(x),
        }
    }
}

fn emit_unm(b: &mut Builder<'_, '_>, dst: u8, src: u8) {
    let v = b.reg(src);
    let ty = b.ty(v);
    let r = if ty.rep == Rep::F64 || ty.within(TypeSet::FLOAT) {
        let x = b.as_f64(v);
        b.ins(Op::FNeg, &[x])
    } else if ty.rep == Rep::I64 {
        b.ins(Op::LNeg, &[v])
    } else if ty.rep == Rep::I32 || ty.within(TypeSet::SMALL) {
        let x = b.as_i32(v);
        let i = b.ins_exit(Op::INeg, &[x], ExitTag::Overflow);
        b.f.result(i)
    } else {
        b.deopt(ExitTag::Type);
        return;
    };
    b.set(dst, r);
}

fn emit_bnot(b: &mut Builder<'_, '_>, dst: u8, src: u8) {
    let v = b.reg(src);
    let ty = b.ty(v);
    let r = if ty.rep == Rep::I64 {
        b.ins(Op::LNot, &[v])
    } else if ty.rep == Rep::I32 || ty.within(TypeSet::SMALL) {
        let x = b.as_i32(v);
        b.ins(Op::INot, &[x])
    } else {
        b.deopt(ExitTag::Type);
        return;
    };
    b.set(dst, r);
}

// --- compares ---------------------------------------------------------------

/// What a compare's operands are compared as.
#[derive(Clone, Copy, PartialEq, Eq)]
enum CmpAs {
    I32,
    I64,
    F64,
}

/// The numeric kind a value is known or speculated to have.
fn num_kind(b: &Builder<'_, '_>, v: Val) -> Option<CmpAs> {
    let ty = b.ty(v);
    match ty.rep {
        Rep::I32 => Some(CmpAs::I32),
        Rep::I64 => Some(CmpAs::I64),
        Rep::F64 => Some(CmpAs::F64),
        _ if ty.within(TypeSet::SMALL) && !ty.set.is_empty() => Some(CmpAs::I32),
        _ if ty.within(TypeSet::FLOAT) && !ty.set.is_empty() => Some(CmpAs::F64),
        _ if ty.within(TypeSet::INT) && !ty.set.is_empty() => Some(CmpAs::I64),
        _ => None,
    }
}

/// What the site's feedback byte says its operands compare as: `I64` when
/// it saw only integers, a big one among them; `F64` when only floats.
fn byte_kind(b: &Builder<'_, '_>, pc: u32) -> Option<CmpAs> {
    let kinds = b.feedback(pc) & fb::KINDS;
    if kinds & fb::BIGINT != 0 && kinds & !fb::INT == 0 {
        Some(CmpAs::I64)
    } else if kinds == fb::FLOAT {
        Some(CmpAs::F64)
    } else {
        None
    }
}

/// Whether the site's feedback saw only kinds `how` compares.
fn byte_within(b: &Builder<'_, '_>, pc: u32, how: CmpAs) -> bool {
    let kinds = match how {
        CmpAs::I32 => fb::SMALL,
        CmpAs::I64 => fb::INT,
        CmpAs::F64 => fb::FLOAT,
    };
    b.feedback(pc) & fb::KINDS & !kinds == 0
}

fn cmp_vals(b: &mut Builder<'_, '_>, cc: Cc, x: Val, y: Val, how: CmpAs) -> Val {
    match how {
        CmpAs::I32 => {
            let (x, y) = (b.as_i32(x), b.as_i32(y));
            b.ins(Op::ICmp(cc), &[x, y])
        }
        CmpAs::I64 => {
            let (x, y) = (b.as_i64(x), b.as_i64(y));
            b.ins(Op::LCmp(cc), &[x, y])
        }
        CmpAs::F64 => {
            let (x, y) = (b.num_f64(x), b.num_f64(y));
            b.ins(Op::FCmp(cc), &[x, y])
        }
    }
}

fn emit_cmp_reg(b: &mut Builder<'_, '_>, pc: u32, insn: Instruction, succs: &[Block]) {
    let op = insn.op();
    let seen = byte_kind(b, pc);
    let wide = seen == Some(CmpAs::I64);
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
    let how = if op != generic {
        // An `_II` form, which exits saw leave i32.
        if wide { CmpAs::I64 } else { CmpAs::I32 }
    } else {
        match (num_kind(b, x), num_kind(b, y)) {
            (Some(CmpAs::I32), Some(CmpAs::I32)) => CmpAs::I32,
            (Some(CmpAs::I32 | CmpAs::I64), Some(CmpAs::I32 | CmpAs::I64)) => CmpAs::I64,
            (Some(CmpAs::F64), Some(_)) | (Some(_), Some(CmpAs::F64)) => {
                // Mixed small and float compare exactly as floats; an i64 does not.
                let wide = num_kind(b, x) == Some(CmpAs::I64) || num_kind(b, y) == Some(CmpAs::I64);
                if wide {
                    let (x, y) = (b.boxed(x), b.boxed(y));
                    match generic {
                        BcOp::JEQ | BcOp::JNEQ => generic_eq(b, x, y, sense, succs),
                        _ => generic_cmp(b, cc, x, y, sense, succs),
                    }
                    return;
                }
                CmpAs::F64
            }
            (Some(CmpAs::I32), None) | (None, Some(CmpAs::I32)) if wide => CmpAs::I64,
            // The other operand speculated to the typed one's kind, unless
            // exits saw it differ.
            (Some(k), None) | (None, Some(k))
                if generic != BcOp::JEQ && generic != BcOp::JNEQ && byte_within(b, pc, k) =>
            {
                k
            }
            (None, None) if wide => CmpAs::I64,
            _ if generic == BcOp::JEQ || generic == BcOp::JNEQ => {
                let (x, y) = (b.boxed(x), b.boxed(y));
                if eq_by_bits(b, x) || eq_by_bits(b, y) {
                    let c = b.ins(Op::SameBits, &[x, y]);
                    branch(b, c, sense, succs);
                } else {
                    generic_eq(b, x, y, sense, succs);
                }
                return;
            }
            _ => match seen {
                Some(how) => how,
                None => {
                    let (x, y) = (b.boxed(x), b.boxed(y));
                    generic_cmp(b, cc, x, y, sense, succs);
                    return;
                }
            },
        }
    };
    let c = cmp_vals(b, cc, x, y, how);
    branch(b, c, sense, succs);
}

/// Both of two `Val`s of kind `set`.
fn both(b: &mut Builder<'_, '_>, set: TypeSet, x: Val, y: Val) -> Val {
    let (cx, cy) = (b.ins(Op::IsType(set), &[x]), b.ins(Op::IsType(set), &[y]));
    let f = b.ins(Op::KB1(false), &[]);
    b.ins(Op::Select, &[cx, cy, f])
}

/// Branch on `x cc y` for `Val`s nothing typed, as the handler orders it:
/// two floats, two small integers, then `jit_lt`/`jit_le` (mixed numbers,
/// big integers, strings), whose failure (a metamethod, an error) deopts.
fn generic_cmp(b: &mut Builder<'_, '_>, cc: Cc, x: Val, y: Val, sense: bool, succs: &[Block]) {
    let ff = both(b, TypeSet::FLOAT, x, y);
    let (floats, rest) = (b.new_sealed_succ(), b.new_sealed_succ());
    b.br(ff, floats, rest);
    b.switch_to(floats);
    let (fx, fy) = (
        b.ins(Op::Unbox(Rep::F64), &[x]),
        b.ins(Op::Unbox(Rep::F64), &[y]),
    );
    let c = b.ins(Op::FCmp(cc), &[fx, fy]);
    branch(b, c, sense, succs);
    b.switch_to(rest);
    let ss = both(b, TypeSet::SMALL, x, y);
    let (smalls, slow) = (b.new_sealed_succ(), b.new_sealed_succ());
    b.br(ss, smalls, slow);
    b.switch_to(smalls);
    let (ix, iy) = (
        b.ins(Op::Unbox(Rep::I32), &[x]),
        b.ins(Op::Unbox(Rep::I32), &[y]),
    );
    let c = b.ins(Op::ICmp(cc), &[ix, iy]);
    branch(b, c, sense, succs);
    b.switch_to(slow);
    // `a > b` is `b < a`.
    let (h, x, y) = match cc {
        Cc::Lt => (HelperId::Lt, x, y),
        Cc::Le => (HelperId::Le, x, y),
        Cc::Gt => (HelperId::Lt, y, x),
        Cc::Ge => (HelperId::Le, y, x),
        Cc::Eq | Cc::Ne => unreachable!("an ordered compare"),
    };
    let c = helper_test(b, h, x, y);
    branch(b, c, sense, succs);
}

/// Branch on `x == y` for `Val`s nothing typed, as the handler orders it:
/// two floats by value (a NaN has its own bits), equal bits, two small
/// integers (unequal), then `jit_eq`, whose failure (`__eq`) deopts.
fn generic_eq(b: &mut Builder<'_, '_>, x: Val, y: Val, sense: bool, succs: &[Block]) {
    let (fall, target) = (succs[0], succs[1]);
    let (yes, no) = if sense {
        (target, fall)
    } else {
        (fall, target)
    };
    let ff = both(b, TypeSet::FLOAT, x, y);
    let (floats, rest) = (b.new_sealed_succ(), b.new_sealed_succ());
    b.br(ff, floats, rest);
    b.switch_to(floats);
    let (fx, fy) = (
        b.ins(Op::Unbox(Rep::F64), &[x]),
        b.ins(Op::Unbox(Rep::F64), &[y]),
    );
    let c = b.ins(Op::FCmp(Cc::Eq), &[fx, fy]);
    branch(b, c, sense, succs);
    b.switch_to(rest);
    let same = b.ins(Op::SameBits, &[x, y]);
    let differ = b.new_sealed_succ();
    b.br(same, yes, differ);
    b.switch_to(differ);
    let ss = both(b, TypeSet::SMALL, x, y);
    let slow = b.new_sealed_succ();
    b.br(ss, no, slow);
    b.switch_to(slow);
    let c = helper_test(b, HelperId::Eq, x, y);
    branch(b, c, sense, succs);
}

/// A compare helper's result as a `B1`, deopting `Before` when it failed.
fn helper_test(b: &mut Builder<'_, '_>, h: HelperId, x: Val, y: Val) -> Val {
    let r = b.ins(Op::Helper(h), &[x, y]);
    let fail = b.ki32(HELPER_FAIL);
    let failed = b.ins(Op::ICmp(Cc::Eq), &[r, fail]);
    let snap = b.snap_before();
    b.ins_snap(Op::GuardFalse, &[failed], snap, ExitTag::Slow);
    let zero = b.ki32(0);
    b.ins(Op::ICmp(Cc::Ne), &[r, zero])
}

/// Whether equality with `v` is identity: it is no number and no table or
/// userdata (`__eq` needs both to be one).
fn eq_by_bits(b: &Builder<'_, '_>, v: Val) -> bool {
    let ty = b.ty(v);
    ty.rep == Rep::Val
        && !ty.set.is_empty()
        && ty.within(TypeSet::NIL | TypeSet::BOOL | TypeSet::STR | TypeSet::FUN | TypeSet::THR)
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
    let how = if op != generic {
        CmpAs::F64
    } else {
        match num_kind(b, v).or(byte_kind(b, pc)) {
            Some(how) => how,
            None => {
                let (v, kk) = (b.boxed(v), b.konst(Value::small(k as i32)));
                generic_cmp(b, cc, v, kk, sense, succs);
                return;
            }
        }
    };
    let c = match how {
        CmpAs::I32 => {
            let x = b.as_i32(v);
            let kk = b.ki32(k as i32);
            b.ins(Op::ICmp(cc), &[x, kk])
        }
        CmpAs::I64 => {
            let x = b.as_i64(v);
            let kk = b.ins(Op::KI64(k), &[]);
            b.ins(Op::LCmp(cc), &[x, kk])
        }
        CmpAs::F64 => {
            let x = b.num_f64(v);
            let kk = b.kf64(k as f64);
            b.ins(Op::FCmp(cc), &[x, kk])
        }
    };
    branch(b, c, sense, succs);
}

fn emit_eqi(b: &mut Builder<'_, '_>, pc: u32, insn: Instruction, succs: &[Block]) {
    let v = b.reg(insn.a());
    let k = insn.cmp_imm_int();
    let sense = insn.op() == BcOp::JEQI;
    let c = match num_kind(b, v).or(byte_kind(b, pc)) {
        Some(CmpAs::F64) => {
            let x = b.num_f64(v);
            let kk = b.kf64(k as f64);
            b.ins(Op::FCmp(Cc::Eq), &[x, kk])
        }
        Some(CmpAs::I64) => {
            let x = b.as_i64(v);
            let kk = b.ins(Op::KI64(k), &[]);
            b.ins(Op::LCmp(Cc::Eq), &[x, kk])
        }
        Some(CmpAs::I32) => {
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
    let (a, nb, c) = (insn.a(), insn.b(), insn.c());
    let fused = matches!(op, BcOp::CALLS | BcOp::CALLS_R0 | BcOp::CALLS_R1);
    let callee = if fused {
        b.reg(insn.imm() as u8)
    } else {
        b.reg(a)
    };
    let mut live = b.cfg.live_in[pc as usize];
    live.remove(a as usize);
    let mut tys = b.store_live(live);
    let cv = b.boxed(callee);
    b.push(Op::Store(a), &[cv]);
    // What lands in the result registers is unknown.
    let nres = if c == 0 {
        b.f.meta.max_stack - a
    } else {
        c - 1
    };
    for r in a..(a as usize + (nres as usize).max(4)).min(tys.len()) as u8 {
        tys[r as usize] = TypeSet::ANY;
    }
    let resume = b.call(
        Op::Call {
            a,
            nargs: nb - 1,
            c,
            pc,
        },
        tys,
        c,
    );
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
