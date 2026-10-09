//! Conditional branches: equality, order, immediate forms, truthiness. The
//! register and immediate compares are adaptive sites: a generic
//! handler per opcode does the inline numbers and specializes the site to
//! `_II` (two small integers) or `_F` (a float register); everything else
//! is in `cmp_slow` and `eq_slow`.

use crate::env::MetamethodBits;
use crate::env::value::{Value, ValueKind};
use crate::instruction::{ADAPTIVE_AB_IMM, ADAPTIVE_AH_IMM, Instruction, Op};
use crate::jit::feedback;
use crate::vm::abi::{Slot, handler};
use crate::vm::num;
use crate::vm::ops::arith::specialize;
use crate::vm::ops::family;

/// Whether a compare site's word is locked (its adaptive bits at `$shift`).
macro_rules! locked {
    ($insn:expr, $shift:expr) => {
        ($insn.raw() >> $shift) & 4 != 0
    };
}

/// Leave the decision to `cmp_adapt`: rewrite the site for `$form` (`None`
/// counts a miss) and run the instruction again. Out of line, so the generic
/// handlers stay small and frameless.
macro_rules! adapt {
    ($form:expr) => {{
        let __f: Option<Op> = $form;
        tail!(
            cmp_adapt,
            closure = Slot::from_raw(__f.map_or(0, |o| o as u64 + 1))
        )
    }};
}
use crate::vm::ops::meta::{binop_metamethod, ret_cond_f, ret_cond_t, stage_mm};
use crate::vm::unwind::OpError;

family! {
    branch II_ROWS;
    JLT_II = op_jlt_ii (Value::both_small, op_jlt, true) |x, y| x < y,
    JNLT_II = op_jnlt_ii (Value::both_small, op_jnlt, false) |x, y| x < y,
    JLE_II = op_jle_ii (Value::both_small, op_jle, true) |x, y| x <= y,
    JNLE_II = op_jnle_ii (Value::both_small, op_jnle, false) |x, y| x <= y,
    JEQ_II = op_jeq_ii (Value::both_small, op_jeq, true) |x, y| x == y,
    JNEQ_II = op_jneq_ii (Value::both_small, op_jneq, false) |x, y| x == y,
}

family! {
    branch_imm F_ROWS;
    JLTI_F = op_jlti_f (false, op_jlti, true) |x, y| x < y,
    JNLTI_F = op_jnlti_f (false, op_jnlti, false) |x, y| x < y,
    JLEI_F = op_jlei_f (false, op_jlei, true) |x, y| x <= y,
    JNLEI_F = op_jnlei_f (false, op_jnlei, false) |x, y| x <= y,
    JGTI_F = op_jgti_f (true, op_jgti, true) |x, y| x < y,
    JNGTI_F = op_jngti_f (true, op_jngti, false) |x, y| x < y,
    JGEI_F = op_jgei_f (true, op_jgei, true) |x, y| x <= y,
    JNGEI_F = op_jngei_f (true, op_jngei, false) |x, y| x <= y,
}

/// JEQ/JNEQ: jump when `(R[a] == R[b]) == $k`. Two small integers
/// specialize the site to `$ii`.
macro_rules! eq_handler {
    ($name:ident, $g:ident, $k:literal, $ii:ident) => {
        handler! {
            bind(insn, pc, base, rt, closure, thread, nret, values);
            op fn $name {
                let (a, b) = (&reg![insn.a()], &reg![insn.b()]);
                // Floats first: a NaN has the same bits as itself.
                let eq = if a.is_float() && b.is_float() {
                    a.read_float() == b.read_float()
                } else if a.same_bits(b) {
                    true
                } else if let Some((x, y)) = Value::both_small(a, b) {
                    if insn.op() == Op::$g && !locked!(insn, ADAPTIVE_AB_IMM) {
                        adapt!(Some(Op::$ii))
                    }
                    x == y
                } else {
                    tail!(eq_slow)
                };
                branch!(eq == $k, insn.branch_offset())
            }
        }
    };
}

eq_handler!(op_jeq, JEQ, true, JEQ_II);
eq_handler!(op_jneq, JNEQ, false, JNEQ_II);

/// JLT/JNLT/JLE/JNLE: jump when `(R[a] <op> R[b]) == $k`. Two floats here;
/// two small integers specialize the site to `$ii`; a float and a small
/// integer count a miss (no form covers them); everything else is
/// `cmp_slow`.
macro_rules! cmp_handler {
    ($name:ident, $g:ident, $op:tt, $k:literal, $ii:ident) => {
        handler! {
            bind(insn, pc, base, rt, closure, thread, nret, values);
            op fn $name {
                let (a, b) = (&reg![insn.a()], &reg![insn.b()]);
                let r = if std::hint::likely(a.is_float() && b.is_float()) {
                    if insn.op() != Op::$g {
                        adapt!(Option::None)
                    }
                    a.read_float() $op b.read_float()
                } else if let Some((x, y)) = Value::both_small(a, b) {
                    if insn.op() == Op::$g && !locked!(insn, ADAPTIVE_AB_IMM) {
                        adapt!(Some(Op::$ii))
                    }
                    x $op y
                } else if let Some((x, y)) = Value::small_float(a, b) {
                    if insn.op() != Op::$g {
                        adapt!(Option::None)
                    }
                    (x as f64) $op y
                } else if let Some((y, x)) = Value::small_float(b, a) {
                    if insn.op() != Op::$g {
                        adapt!(Option::None)
                    }
                    x $op (y as f64)
                } else {
                    tail!(cmp_slow)
                };
                branch!(r == $k, insn.branch_offset())
            }
        }
    };
}

cmp_handler!(op_jlt, JLT, <, true, JLT_II);
cmp_handler!(op_jnlt, JNLT, <, false, JNLT_II);
cmp_handler!(op_jle, JLE, <=, true, JLE_II);
cmp_handler!(op_jnle, JNLE, <=, false, JNLE_II);

/// The immediate ordered compares: jump when `(R[a] <cmp> imm) == $k`, `<`
/// if `$lt` else `<=`, with the immediate on the left if `$swap`. A float
/// register specializes the site to `$f`; a small one on a specialized site
/// counts a miss.
macro_rules! cmp_imm_handler {
    ($name:ident, $g:ident, $k:literal, $lt:literal, $swap:literal, $f:ident) => {
        handler! {
            bind(insn, pc, base, rt, closure, thread, nret, values);
            op fn $name {
                let v = &reg![insn.a()];
                // A 15-bit integer, so exact as a float too.
                let k = insn.cmp_imm_int();
                let ints = |x: i64, y: i64| if $lt { x < y } else { x <= y };
                let floats = |x: f64, y: f64| if $lt { x < y } else { x <= y };
                let r = if let Some(i) = v.get_small() {
                    if insn.op() != Op::$g {
                        adapt!(Option::None)
                    }
                    let i = i as i64;
                    if $swap { ints(k, i) } else { ints(i, k) }
                } else if v.is_float() {
                    if insn.op() == Op::$g && !locked!(insn, ADAPTIVE_AH_IMM) {
                        adapt!(Some(Op::$f))
                    }
                    let f = v.read_float();
                    if $swap { floats(k as f64, f) } else { floats(f, k as f64) }
                } else if let Some(i) = v.get_integer() {
                    if $swap { ints(k, i) } else { ints(i, k) }
                } else {
                    tail!(cmp_slow)
                };
                branch!(r == $k, insn.branch_offset())
            }
        }
    };
}

cmp_imm_handler!(op_jlti, JLTI, true, true, false, JLTI_F);
cmp_imm_handler!(op_jnlti, JNLTI, false, true, false, JNLTI_F);
cmp_imm_handler!(op_jlei, JLEI, true, false, false, JLEI_F);
cmp_imm_handler!(op_jnlei, JNLEI, false, false, false, JNLEI_F);
cmp_imm_handler!(op_jgti, JGTI, true, true, true, JGTI_F);
cmp_imm_handler!(op_jngti, JNGTI, false, true, true, JNGTI_F);
cmp_imm_handler!(op_jgei, JGEI, true, false, true, JGEI_F);
cmp_imm_handler!(op_jngei, JNGEI, false, false, true, JNGEI_F);

/// JEQI/JNEQI: jump when `(R[a] == imm) == $k`. Never `__eq`: the immediate
/// is a number.
macro_rules! eqi_handler {
    ($name:ident, $k:literal) => {
        handler! {
            bind(insn, pc, base, rt, closure, thread, nret, values);
            op fn $name {
                let v = &reg![insn.a()];
                let k = insn.cmp_imm_int();
                let eq = if let Some(i) = v.get_small() {
                    i as i64 == k
                } else if v.is_float() {
                    v.read_float() == k as f64
                } else if let Some(i) = v.get_integer() {
                    i == k
                } else {
                    false
                };
                branch!(eq == $k, insn.branch_offset())
            }
        }
    };
}

eqi_handler!(op_jeqi, true);
eqi_handler!(op_jneqi, false);

/// JEQS/JNEQS: jump when `(R[a] == K[h]) == $k`, `K[h]` a string. Strings
/// are interned, so equality is identity.
macro_rules! eqs_handler {
    ($name:ident, $k:literal) => {
        handler! {
            bind(insn, pc, base, rt, closure, thread, nret, values);
            op fn $name {
                let eq = reg![insn.a()].same_bits(&k![insn.h()]);
                branch!(eq == $k, insn.branch_offset())
            }
        }
    };
}

eqs_handler!(op_jeqs, true);
eqs_handler!(op_jneqs, false);

/// JT/JF: jump when `truthy(R[a]) == $k`.
macro_rules! test_handler {
    ($name:ident, $k:literal) => {
        handler! {
            bind(insn, pc, base, rt, closure, thread, nret, values);
            op fn $name {
                let truthy = !reg![insn.a()].is_falsy();
                branch!(truthy == $k, insn.branch_offset())
            }
        }
    };
}

test_handler!(op_jt, true);
test_handler!(op_jf, false);

/// JTSET/JFSET: jump when `truthy(R[b]) == $k`, copying `R[b]` to `R[a]`
/// first.
macro_rules! testset_handler {
    ($name:ident, $k:literal) => {
        handler! {
            bind(insn, pc, base, rt, closure, thread, nret, values);
            op fn $name {
                let v = reg![insn.b()];
                let jump = !v.is_falsy() == $k;
                if jump {
                    reg![insn.a()] = v;
                }
                branch!(jump, insn.branch_offset())
            }
        }
    };
}

testset_handler!(op_jtset, true);
testset_handler!(op_jfset, false);

handler! {
    bind(insn, pc, base, rt, closure, thread, nret, values);

    /// Rewrite the compare site at `pc - 1` for the form in the `closure`
    /// slot (0: none, else the opcode plus one), then run it again.
    slow fn cmp_adapt {
        let word: Instruction = insn_at!();
        let site = unsafe { pc.sub(1).cast_mut() };
        let form = match closure.raw() {
            0 => Option::None,
            b => Some(Instruction::from_raw(b - 1).op()),
        };
        unsafe { specialize(site, word, form, true) };
        closure = Slot::closure(unsafe { crate::vm::frame::closure(base) });
        jump_by!(-1);
        next!()
    }

    /// JEQ/JNEQ past the fast path.
    slow fn eq_slow {
        let insn: Instruction = insn_at!();
        let (a, b) = (reg![insn.a()], reg![insn.b()]);
        let closure = unsafe { crate::vm::frame::closure(base) };
        feedback::record(closure, unsafe { pc.sub(1) }, feedback::kind(a) | feedback::kind(b));
        let k = insn.generic_op() == Op::JEQ;
        if num::raw_eq(a, b) {
            branch!(k, insn.branch_offset())
        }
        // Lua 5.5: `__eq` fires only when both operands are the same
        // non-primitive type (tables or userdata) and raw equality fails.
        let try_meta = (a.kind() == ValueKind::Table && b.kind() == ValueKind::Table)
            || (a.kind() == ValueKind::Userdata && b.kind() == ValueKind::Userdata);
        if try_meta {
            let mm = binop_metamethod(rt, a, b, MetamethodBits::EQ);
            if !mm.is_nil() {
                feedback::record(closure, unsafe { pc.sub(1) }, feedback::MM);
                stage_mm!(pc, base, rt, if k { ret_cond_t } else { ret_cond_f }, mm, [a, b])
            }
        }
        branch!(!k, insn.branch_offset())
    }

    /// The slow path of the ordered compares: boxed and mixed numbers,
    /// strings, and the metamethods.
    slow fn cmp_slow {
        let insn: Instruction = insn_at!();
        let op = insn.generic_op();
        let k = op.branch_sense() == Some(true);
        let offset = insn.branch_offset();
        let (a, b) = if matches!(op, Op::JLT | Op::JNLT | Op::JLE | Op::JNLE) {
            (reg![insn.a()], reg![insn.b()])
        } else {
            let (v, lit) = (reg![insn.a()], insn.cmp_imm_value(rt.mutation()));
            if matches!(op, Op::JGTI | Op::JNGTI | Op::JGEI | Op::JNGEI) { (lit, v) } else { (v, lit) }
        };
        let closure = unsafe { crate::vm::frame::closure(base) };
        feedback::record(closure, unsafe { pc.sub(1) }, feedback::kind(a) | feedback::kind(b));
        let le = matches!(op, Op::JLE | Op::JNLE | Op::JLEI | Op::JNLEI | Op::JGEI | Op::JNGEI);
        let primitive = if let Some(x) = a.get_integer()
            && let Some(y) = b.get_integer()
        {
            Some(if le { x <= y } else { x < y })
        } else if let (Some(x), Some(y)) = (a.get_integer(), b.get_float()) {
            Some(if le { num::le_int_float(x, y) } else { num::lt_int_float(x, y) })
        } else if let (Some(x), Some(y)) = (a.get_float(), b.get_integer()) {
            Some(if le { num::le_float_int(x, y) } else { num::lt_float_int(x, y) })
        } else if let (Some(x), Some(y)) = (a.get_float(), b.get_float()) {
            Some(if le { x <= y } else { x < y })
        } else if let (Some(x), Some(y)) = (a.get_string(), b.get_string()) {
            Some(if le { x <= y } else { x < y })
        } else {
            None
        };
        if let Some(r) = primitive {
            branch!(r == k, offset)
        }
        let bit = if le { MetamethodBits::LE } else { MetamethodBits::LT };
        let mm = binop_metamethod(rt, a, b, bit);
        if mm.is_nil() {
            raise!(OpError::Compare(a, b))
        }
        feedback::record(closure, unsafe { pc.sub(1) }, feedback::MM);
        stage_mm!(pc, base, rt, if k { ret_cond_t } else { ret_cond_f }, mm, [a, b])
    }
}
