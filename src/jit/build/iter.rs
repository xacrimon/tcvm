//! Generic `for` loops: `TFORPREP` as the form it wrote at its `TFORCALL` says,
//! the `next` and `ipairs` steps without calling the iterator, each arm going
//! where the `TFORLOOP` after it would (`emit_instance` emits the two as one),
//! any other iterator called, `TFORLOOP` a nil test.

use crate::env::MetamethodBits;
use crate::env::table::POS_PART_SHIFT;
use crate::env::value::Value;
use crate::instruction::{Instruction, Op as BcOp, TFOR_VARS};
use crate::jit::build::Builder;
use crate::jit::build::emit::{branch_arm, cold, enter_join, staged_call};
use crate::jit::build::tables::class_bit;
use crate::jit::ir::ops::{
    Cc, ExitTag, HASH_INTS, HASH_MISC, HASH_STRS, HELPER_FAIL_VAL, HelperId, Op,
};
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

/// `TFORCALL_NEXT a count` and its `TFORLOOP`, to `exit` or `body`:
/// `next`'s walk at the position in `R[a+3]`, the array part and named
/// slots inline, and the steps past parts that hold no key: on to the named
/// slots, or to the end; the other parts by helper through the slots, its
/// first variable then tested.
pub(super) fn emit_tforcall_next(
    b: &mut Builder<'_, '_>,
    a: u8,
    count: u8,
    exit: Block,
    body: Block,
) {
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
    let slow = b.new_block();
    // A position in the array part is its index.
    let part = b.ki32(1 << POS_PART_SHIFT);
    let in_array = b.ins(Op::ICmp(Cc::Lt), &[p, part]);
    let rest = branch_arm(b, in_array);
    let i = b.ins(Op::ArrNext, &[t, p]);
    let n = b.ins(Op::TabASize, &[t]);
    let arr = b.ins(Op::TabArray, &[t]);
    let found = b.ins(Op::ICmp(Cc::Lo), &[i, n]);
    let past = branch_arm(b, found);
    let v = b.ins(Op::ArrLoad, &[arr, i]);
    let k = b.ins(Op::Box, &[i]);
    let one = b.ki32(1);
    let after = b.ins(Op::IAddNo, &[i, one]);
    let after = b.ins(Op::Box, &[after]);
    b.set(a + 3, after);
    step_to(b, body, vars, count, k, v);
    // Past the array part: the integer keys, then the named slots.
    b.switch_to(past);
    let no_ints = b.ins(Op::TabNoKeys(HASH_INTS), &[t]);
    let ints = branch_arm(b, no_ints);
    named_step(b, a, count, t, zero, [exit, body, slow]);
    b.switch_to(ints);
    b.jump(slow);
    // The named slots, part 2, at the index after the position's part.
    b.switch_to(rest);
    let named = b.ki32(2 << POS_PART_SHIFT);
    let rel = b.ins(Op::ISubNo, &[p, named]);
    let in_named = b.ins(Op::ICmp(Cc::Lo), &[rel, part]);
    let other = branch_arm(b, in_named);
    named_step(b, a, count, t, rel, [exit, body, slow]);
    b.switch_to(other);
    b.jump(slow);
    enter_join(b, slow);
    cold(b, |b| {
        b.store_regs([a as usize + 1, a as usize + 3]);
        let at = b.slot_of(a);
        b.push(Op::Helper(HelperId::TforNext { a: at, count }), &[]);
        for k in 0..count {
            b.reload(vars + k);
        }
        b.reload(a + 3);
        b.flush();
        let k = b.reg(vars);
        let k = b.boxed(k);
        let end = b.ins(Op::IsType(TypeSet::NIL), &[k]);
        b.br(end, exit, body);
    });
}

/// `next`'s step through the named slots of table `t` (a `Ptr`) from index
/// `rel`: the next one to `body`; past the last, the end at `exit` when the
/// parts after them hold no key, else the helper at `slow`.
fn named_step(
    b: &mut Builder<'_, '_>,
    a: u8,
    count: u8,
    t: Val,
    rel: Val,
    [exit, body, slow]: [Block; 3],
) {
    let vars = a + TFOR_VARS;
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
    step_to(b, body, vars, count, k, v);
    b.switch_to(gone);
    let empty = b.ins(Op::TabNoKeys(HASH_STRS | HASH_MISC), &[t]);
    let more = branch_arm(b, empty);
    let nil = b.konst(Value::nil());
    step_to(b, exit, vars, count, nil, nil);
    b.switch_to(more);
    b.jump(slow);
}

/// `TFORCALL_IPAIRS a count` and its `TFORLOOP`, to `exit` or `body`: the
/// array part inline; a raw nil (there, or past it with no integer key in
/// the hash part) the end unless the class has `__index`, which exits; past
/// it otherwise the helper, whose failure (`__index` may answer) exits.
pub(super) fn emit_tforcall_ipairs(
    b: &mut Builder<'_, '_>,
    a: u8,
    count: u8,
    exit: Block,
    body: Block,
) {
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
    let n = b.ins(Op::TabASize, &[t]);
    // Before the branch: the first step's then dominates the loop's.
    let arr = b.ins(Op::TabArray, &[t]);
    let inside = b.ins(Op::ICmp(Cc::Lo), &[k, n]);
    let outside = branch_arm(b, inside);
    let v = b.ins(Op::ArrLoad, &[arr, k]);
    let nil = b.ins(Op::IsType(TypeSet::NIL), &[v]);
    let found = branch_arm(b, nil);
    // Inside the array part a nil is the raw value (`get_int`).
    let raw_nil = b.new_block();
    b.jump(raw_nil);
    b.switch_to(found);
    step_to(b, body, vars, count, kv, v);
    // Past the array part: the end of a list a constructor sized, or keys
    // the hash part holds.
    b.switch_to(outside);
    cold(b, |b| {
        let no_ints = b.ins(Op::TabNoKeys(HASH_INTS), &[t]);
        let hashed = branch_arm(b, no_ints);
        b.jump(raw_nil);
        b.switch_to(hashed);
        past_array(b, exit, body, vars, count, g, kv);
    });
    enter_join(b, raw_nil);
    let has = class_bit(b, t, MetamethodBits::INDEX);
    let s = b.snap_before();
    b.ins_snap(Op::GuardFalse, &[has], s, ExitTag::Slow);
    let none = b.konst(Value::nil());
    step_to(b, exit, vars, count, none, none);
}

/// The `ipairs` step at key `kv` past the array part: the value, or the
/// end with the variables nil.
fn past_array(
    b: &mut Builder<'_, '_>,
    exit: Block,
    body: Block,
    vars: u8,
    count: u8,
    t: Val,
    kv: Val,
) {
    let w = b.ins(Op::Helper(HelperId::TforIpairs), &[t, kv]);
    let fail = b.ins(Op::KVal(HELPER_FAIL_VAL), &[]);
    let failed = b.ins(Op::SameBits, &[w, fail]);
    let s = b.snap_before();
    b.ins_snap(Op::GuardFalse, &[failed], s, ExitTag::Slow);
    let end = b.ins(Op::IsType(TypeSet::NIL), &[w]);
    let more = branch_arm(b, end);
    let nil = b.konst(Value::nil());
    step_to(b, exit, vars, count, nil, w);
    b.switch_to(more);
    step_to(b, body, vars, count, kv, w);
}

/// The loop variables `(k, v, nil...)`, then `to`.
fn step_to(b: &mut Builder<'_, '_>, to: Block, vars: u8, count: u8, k: Val, v: Val) {
    b.set(vars, k);
    if count > 1 {
        b.set(vars + 1, v);
    }
    for i in 2..count {
        let nil = b.konst(Value::nil());
        b.set(vars + i, nil);
    }
    b.flush();
    b.jump(to);
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
