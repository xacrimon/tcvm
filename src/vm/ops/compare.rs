//! Conditional branches: equality, order, immediate forms, truthiness.

use crate::env::MetamethodBits;
use crate::env::value::{Value, ValueKind};
use crate::instruction::Op;
use crate::vm::abi::handler;
use crate::vm::num;
use crate::vm::ops::meta::{binop_metamethod, ret_cond, stage_mm};
use crate::vm::unwind::OpError;

/// JEQ/JNEQ: jump when `(R[a] == R[b]) == $k`.
macro_rules! eq_handler {
    ($name:ident, $k:literal) => {
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
                    x == y
                } else {
                    tail!(eq_slow)
                };
                branch!(eq == $k, insn.imm())
            }
        }
    };
}

eq_handler!(op_jeq, true);
eq_handler!(op_jneq, false);

/// JLT/JNLT/JLE/JNLE: jump when `(R[a] <op> R[b]) == $k`. Two floats or two
/// inline integers here, everything else in `cmp_slow`.
macro_rules! cmp_handler {
    ($name:ident, $op:tt, $k:literal) => {
        handler! {
            bind(insn, pc, base, rt, closure, thread, nret, values);
            op fn $name {
                let (a, b) = (&reg![insn.a()], &reg![insn.b()]);
                let r = if std::hint::likely(a.is_float() && b.is_float()) {
                    a.read_float() $op b.read_float()
                } else if let Some((x, y)) = Value::both_small(a, b) {
                    x $op y
                } else {
                    tail!(cmp_slow)
                };
                branch!(r == $k, insn.imm())
            }
        }
    };
}

cmp_handler!(op_jlt, <, true);
cmp_handler!(op_jnlt, <, false);
cmp_handler!(op_jle, <=, true);
cmp_handler!(op_jnle, <=, false);

/// The immediate ordered compares: jump when `(R[a] <cmp> imm) == $k`, `<`
/// if `$lt` else `<=`, with the immediate on the left if `$swap`.
macro_rules! cmp_imm_handler {
    ($name:ident, $k:literal, $lt:literal, $swap:literal) => {
        handler! {
            bind(insn, pc, base, rt, closure, thread, nret, values);
            op fn $name {
                let v = &reg![insn.a()];
                // A 15-bit integer, so exact as a float too.
                let k = insn.cmp_imm_int();
                let ints = |x: i64, y: i64| if $lt { x < y } else { x <= y };
                let floats = |x: f64, y: f64| if $lt { x < y } else { x <= y };
                let r = if let Some(i) = v.get_small() {
                    let i = i as i64;
                    if $swap { ints(k, i) } else { ints(i, k) }
                } else if v.is_float() {
                    let f = v.read_float();
                    if $swap { floats(k as f64, f) } else { floats(f, k as f64) }
                } else if let Some(i) = v.get_integer() {
                    if $swap { ints(k, i) } else { ints(i, k) }
                } else {
                    tail!(cmp_slow)
                };
                branch!(r == $k, insn.imm())
            }
        }
    };
}

cmp_imm_handler!(op_jlti, true, true, false);
cmp_imm_handler!(op_jnlti, false, true, false);
cmp_imm_handler!(op_jlei, true, false, false);
cmp_imm_handler!(op_jnlei, false, false, false);
cmp_imm_handler!(op_jgti, true, true, true);
cmp_imm_handler!(op_jngti, false, true, true);
cmp_imm_handler!(op_jgei, true, false, true);
cmp_imm_handler!(op_jngei, false, false, true);

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
                branch!(eq == $k, insn.imm())
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
                branch!(eq == $k, insn.imm())
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

    /// JEQ/JNEQ past the fast path.
    slow fn eq_slow {
        let insn = insn_at!();
        let (a, b) = (reg![insn.a()], reg![insn.b()]);
        let k = insn.op() == Op::JEQ;
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
        let insn = insn_at!();
        let op = insn.op();
        let k = op.branch_sense() == Some(true);
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
            branch!(r == k, insn.imm())
        }
        let bit = if le { MetamethodBits::LE } else { MetamethodBits::LT };
        let mm = binop_metamethod(rt, a, b, bit);
        if mm.is_nil() {
            raise!(OpError::Compare(a, b))
        }
        stage_mm!(pc, base, rt, ret_cond, mm, [a, b])
    }
}
