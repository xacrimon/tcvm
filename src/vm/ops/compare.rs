//! Conditional branches: equality, order, immediate forms, truthiness. The
//! register and immediate compares are adaptive sites: a generic
//! handler per opcode does the inline numbers and specializes the site to
//! `_II` (two small integers) or `_F` (a float register); everything else
//! is in `cmp_slow` and `eq_slow`.

use crate::env::MetamethodBits;
use crate::env::value::{Value, ValueKind};
use crate::instruction::{ADAPTIVE_AB_IMM, ADAPTIVE_AH_IMM, Instruction, Op};
use crate::vm::abi::{Slot, handler};
use crate::vm::num;
use crate::vm::ops::arith::specialize;

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
use crate::vm::ops::meta::{binop_metamethod, ret_cond, stage_mm};
use crate::vm::unwind::OpError;

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
                branch!(eq == $k, insn.imm())
            }
        }
    };
}

eq_handler!(op_jeq, JEQ, true, JEQ_II);
eq_handler!(op_jneq, JNEQ, false, JNEQ_II);

/// `JEQ_II`/`JNEQ_II`: both small, else the generic.
macro_rules! eq_ii_handler {
    ($name:ident, $k:literal, $generic:ident) => {
        handler! {
            bind(insn, pc, base, rt, closure, thread, nret, values);
            op fn $name {
                let Some((x, y)) = Value::both_small(&reg![insn.a()], &reg![insn.b()]) else {
                    tail!($generic)
                };
                branch!((x == y) == $k, insn.imm())
            }
        }
    };
}

eq_ii_handler!(op_jeq_ii, true, op_jeq);
eq_ii_handler!(op_jneq_ii, false, op_jneq);

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
                branch!(r == $k, insn.imm())
            }
        }
    };
}

cmp_handler!(op_jlt, JLT, <, true, JLT_II);
cmp_handler!(op_jnlt, JNLT, <, false, JNLT_II);
cmp_handler!(op_jle, JLE, <=, true, JLE_II);
cmp_handler!(op_jnle, JNLE, <=, false, JNLE_II);

/// `JLT_II` and friends: both small, else the generic.
macro_rules! cmp_ii_handler {
    ($name:ident, $op:tt, $k:literal, $generic:ident) => {
        handler! {
            bind(insn, pc, base, rt, closure, thread, nret, values);
            op fn $name {
                let Some((x, y)) = Value::both_small(&reg![insn.a()], &reg![insn.b()]) else {
                    tail!($generic)
                };
                branch!((x $op y) == $k, insn.imm())
            }
        }
    };
}

cmp_ii_handler!(op_jlt_ii, <, true, op_jlt);
cmp_ii_handler!(op_jnlt_ii, <, false, op_jnlt);
cmp_ii_handler!(op_jle_ii, <=, true, op_jle);
cmp_ii_handler!(op_jnle_ii, <=, false, op_jnle);

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
                branch!(r == $k, insn.imm24())
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

/// `JLTI_F` and friends: a float register against the immediate converted
/// with one `scvtf`, else the generic.
macro_rules! cmp_imm_f_handler {
    ($name:ident, $k:literal, $lt:literal, $swap:literal, $generic:ident) => {
        handler! {
            bind(insn, pc, base, rt, closure, thread, nret, values);
            op fn $name {
                let v = &reg![insn.a()];
                if !v.is_float() {
                    tail!($generic)
                }
                let f = v.read_float();
                let k = insn.cmp_imm_int() as f64;
                let (x, y) = if $swap { (k, f) } else { (f, k) };
                let r = if $lt { x < y } else { x <= y };
                branch!(r == $k, insn.imm24())
            }
        }
    };
}

cmp_imm_f_handler!(op_jlti_f, true, true, false, op_jlti);
cmp_imm_f_handler!(op_jnlti_f, false, true, false, op_jnlti);
cmp_imm_f_handler!(op_jlei_f, true, false, false, op_jlei);
cmp_imm_f_handler!(op_jnlei_f, false, false, false, op_jnlei);
cmp_imm_f_handler!(op_jgti_f, true, true, true, op_jgti);
cmp_imm_f_handler!(op_jngti_f, false, true, true, op_jngti);
cmp_imm_f_handler!(op_jgei_f, true, false, true, op_jgei);
cmp_imm_f_handler!(op_jngei_f, false, false, true, op_jngei);

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
                branch!(eq == $k, insn.imm24())
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
                branch!(eq == $k, insn.imm24())
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
                branch!(truthy == $k, insn.imm())
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
                branch!(jump, insn.imm())
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
        let k = insn.generic_op() == Op::JEQ;
        if num::raw_eq(a, b) {
            branch!(k, insn.imm())
        }
        // Lua 5.5: `__eq` fires only when both operands are the same
        // non-primitive type (tables or userdata) and raw equality fails.
        let try_meta = (a.kind() == ValueKind::Table && b.kind() == ValueKind::Table)
            || (a.kind() == ValueKind::Userdata && b.kind() == ValueKind::Userdata);
        if try_meta {
            let mm = binop_metamethod(rt, a, b, MetamethodBits::EQ);
            if !mm.is_nil() {
                stage_mm!(pc, base, rt, ret_cond, mm, [a, b])
            }
        }
        branch!(!k, insn.imm())
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
        stage_mm!(pc, base, rt, ret_cond, mm, [a, b])
    }
}
