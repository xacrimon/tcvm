//! Generic `for` loops (`jit-design.md` 7.6): `TFORPREP` as the form it
//! wrote at its `TFORCALL` says, the `next` and `ipairs` steps without
//! calling the iterator, any other iterator called, `TFORLOOP` a nil test.

use crate::env::table::POS_PART_SHIFT;
use crate::env::value::Value;
use crate::instruction::{Instruction, Op as BcOp, TFOR_VARS};
use crate::jit::build::Builder;
use crate::jit::build::emit::{branch_arm, enter_join, staged_call};
use crate::jit::ir::ops::{Cc, ExitTag, HELPER_FAIL_VAL, HelperId, Op};
use crate::jit::ir::types::{Rep, TypeSet};
use crate::jit::ir::{Block, Val};
use crate::vm::ops::control::TFOR_IPAIRS;

/// `TFORPREP a`: the closing value to `R[a+2]`, the control to the first
/// variable, and the position slot `R[a+3]` the form of the `TFORCALL` it
/// jumps to says: what `op_tforprep` decided the last time it ran, guarded
/// here, since it decides again on each run.
pub(super) fn emit_tforprep(b: &mut Builder<'_, '_>, pc: u32, insn: Instruction, succs: &[Block]) {
    let a = insn.a();
    let target = (pc as i64 + 1 + insn.branch_offset() as i64) as u32;
    let (iter, state, control, closing) = (b.reg(a), b.reg(a + 1), b.reg(a + 2), b.reg(a + 3));
    // A closing value makes a to-be-closed variable: the interpreter's.
    let cl = b.boxed(closing);
    let falsy = b.ins(Op::IsFalsy, &[cl]);
    let s = b.snap_before();
    b.ins_snap(Op::GuardTrue, &[falsy], s, ExitTag::Unsupported);
    let form = b.code(target).op();
    let pos = match form {
        BcOp::TFORCALL_NEXT | BcOp::TFORCALL_IPAIRS => {
            let f = if form == BcOp::TFORCALL_NEXT {
                b.ctx.next_fn()
            } else {
                b.ctx.ipairs_iter()
            };
            let k = b.konst(Value::function(f));
            let iv = b.boxed(iter);
            b.ins_exit(Op::GuardSame, &[iv, k], ExitTag::Type);
            b.guard(state, TypeSet::TAB);
            if form == BcOp::TFORCALL_NEXT {
                b.guard(control, TypeSet::NIL);
                b.konst(Value::small(0))
            } else {
                b.konst(Value::small(TFOR_IPAIRS))
            }
        }
        _ => b.konst(Value::nil()),
    };
    b.set(a + TFOR_VARS, control);
    b.set(a + 2, closing);
    b.set(a + 3, pos);
    b.flush();
    b.jump(succs[0]);
}

/// `TFORCALL_NEXT a count`: `next`'s walk at the position in `R[a+3]`,
/// the array part and named slots inline, the other parts and the step
/// from one part to the next by helper through the slots;
/// `TFORLOOP` then tests the first variable.
pub(super) fn emit_tforcall_next(b: &mut Builder<'_, '_>, a: u8, count: u8) {
    let vars = a + TFOR_VARS;
    // Not a table, or no position: the interpreter calls the iterator.
    // Tested before the step, which moves the position.
    let state = b.reg(a + 1);
    let g = b.guard(state, TypeSet::TAB);
    let t = b.ins(Op::Unbox(Rep::Ptr), &[g]);
    let pos = b.reg(a + 3);
    let p = b.as_i32(pos);
    let zero = b.ki32(0);
    let ok = b.ins(Op::ICmp(Cc::Ge), &[p, zero]);
    let s = b.snap_before();
    b.ins_snap(Op::GuardTrue, &[ok], s, ExitTag::Slow);
    let join = b.new_block();
    let slow = b.new_block();
    // A position in the array part is its index.
    let part = b.ki32(1 << POS_PART_SHIFT);
    let in_array = b.ins(Op::ICmp(Cc::Lt), &[p, part]);
    let rest = branch_arm(b, in_array);
    let i = b.ins(Op::ArrNext, &[t, p]);
    let n = b.ins(Op::TabASize, &[t]);
    let found = b.ins(Op::ICmp(Cc::Lo), &[i, n]);
    let past = branch_arm(b, found);
    let arr = b.ins(Op::TabArray, &[t]);
    let v = b.ins(Op::ArrLoad, &[arr, i]);
    let k = b.ins(Op::Box, &[i]);
    let one = b.ki32(1);
    let after = b.ins(Op::IAddNo, &[i, one]);
    let after = b.ins(Op::Box, &[after]);
    b.set(a + 3, after);
    step_to(b, join, vars, count, k, v);
    // The named slots, part 2, at the index after the position's part.
    b.switch_to(rest);
    let named = b.ki32(2 << POS_PART_SHIFT);
    let rel = b.ins(Op::ISubNo, &[p, named]);
    let in_named = b.ins(Op::ICmp(Cc::Lo), &[rel, part]);
    let other = branch_arm(b, in_named);
    let j = b.ins(Op::NamedNext, &[t, rel]);
    let none = b.ki32(-1);
    let found = b.ins(Op::ICmp(Cc::Ne), &[j, none]);
    let gone = branch_arm(b, found);
    let k = b.ins(Op::NamedKey, &[t, j]);
    let v = b.ins(Op::NamedVal, &[t, j]);
    let next = b.ki32((2 << POS_PART_SHIFT) + 1);
    let after = b.ins(Op::IAddNo, &[j, next]);
    let after = b.ins(Op::Box, &[after]);
    b.set(a + 3, after);
    step_to(b, join, vars, count, k, v);
    for blk in [past, other, gone] {
        b.switch_to(blk);
        b.jump(slow);
    }
    enter_join(b, slow);
    b.store_regs([a as usize + 1, a as usize + 3]);
    b.push(Op::Helper(HelperId::TforNext { a, count }), &[]);
    for k in 0..count {
        b.reload(vars + k);
    }
    b.reload(a + 3);
    b.flush();
    b.jump(join);
    enter_join(b, join);
}

/// `TFORCALL_IPAIRS a count`: the array part inline, past it the helper,
/// whose failure (`__index` may answer) exits.
pub(super) fn emit_tforcall_ipairs(b: &mut Builder<'_, '_>, a: u8, count: u8) {
    let vars = a + TFOR_VARS;
    let state = b.reg(a + 1);
    let g = b.guard(state, TypeSet::TAB);
    let t = b.ins(Op::Unbox(Rep::Ptr), &[g]);
    let pos = b.reg(a + 3);
    let pos = b.boxed(pos);
    let mark = b.konst(Value::small(TFOR_IPAIRS));
    b.ins_exit(Op::GuardSame, &[pos, mark], ExitTag::Type);
    let control = b.reg(vars);
    let i = b.as_i32(control);
    let one = b.ki32(1);
    let k = b.ins_exit(Op::IAdd, &[i, one], ExitTag::Slow);
    let k = b.f.result(k);
    let kv = b.ins(Op::Box, &[k]);
    let join = b.new_block();
    let n = b.ins(Op::TabASize, &[t]);
    let inside = b.ins(Op::ICmp(Cc::Lo), &[k, n]);
    let outside = branch_arm(b, inside);
    let arr = b.ins(Op::TabArray, &[t]);
    let v = b.ins(Op::ArrLoad, &[arr, k]);
    let nil = b.ins(Op::IsType(TypeSet::NIL), &[v]);
    let found = branch_arm(b, nil);
    past_array(b, join, vars, count, g, kv);
    b.switch_to(found);
    step_to(b, join, vars, count, kv, v);
    b.switch_to(outside);
    past_array(b, join, vars, count, g, kv);
    enter_join(b, join);
}

/// The `ipairs` step at key `kv` past the array part: the value, or the
/// end with the variables nil.
fn past_array(b: &mut Builder<'_, '_>, join: Block, vars: u8, count: u8, t: Val, kv: Val) {
    let w = b.ins(Op::Helper(HelperId::TforIpairs), &[t, kv]);
    let fail = b.ins(Op::KVal(HELPER_FAIL_VAL), &[]);
    let failed = b.ins(Op::SameBits, &[w, fail]);
    let s = b.snap_before();
    b.ins_snap(Op::GuardFalse, &[failed], s, ExitTag::Slow);
    let end = b.ins(Op::IsType(TypeSet::NIL), &[w]);
    let nil = b.konst(Value::nil());
    let key = b.ins(Op::Select, &[end, nil, kv]);
    step_to(b, join, vars, count, key, w);
}

/// The loop variables `(k, v, nil...)`, then the join.
fn step_to(b: &mut Builder<'_, '_>, join: Block, vars: u8, count: u8, k: Val, v: Val) {
    b.set(vars, k);
    if count > 1 {
        b.set(vars + 1, v);
    }
    for i in 2..count {
        let nil = b.konst(Value::nil());
        b.set(vars + i, nil);
    }
    b.flush();
    b.jump(join);
}

/// `TFORCALL a count` with another iterator: `R[a](R[a+1], R[a+4])`, its
/// results the loop variables.
pub(super) fn emit_tforcall(b: &mut Builder<'_, '_>, a: u8, count: u8) {
    let vars = a + TFOR_VARS;
    let (iter, state, control) = (b.reg(a), b.reg(a + 1), b.reg(vars));
    let resume = staged_call(b, iter, &[state, control], vars, count + 1);
    let first = b.f.insts_of(resume)[0];
    let rs: Vec<Val> = b.f.results(first).collect();
    for (k, &v) in rs.iter().enumerate() {
        b.set(vars + k as u8, v);
    }
    b.flush();
    b.lazy_exit_after();
}
