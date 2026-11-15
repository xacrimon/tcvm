//! Generic `for` loops: `TFORPREP` as the form it wrote at its `TFORCALL`
//! says, the `next` and `ipairs` steps without calling the iterator, each
//! arm going where the `TFORLOOP` after it would, and any other iterator
//! called.

use super::Builder;
use crate::env::MetamethodBits;
use crate::env::table::POS_PART_SHIFT;
use crate::env::value::Value;
use crate::instruction::{Instruction, Op as Bc, TFOR_VARS};
use crate::jit::ir::fields::Field;
use crate::jit::ir::ops::{Cc, ExitTag, HASH_INTS, HASH_MISC, HASH_STRS, Helper, Op};
use crate::jit::ir::types::{Rep, TypeSet};
use crate::jit::ir::{Block, BlockFlags, Val};
use crate::vm::ops::control::TFOR_IPAIRS;

impl Builder<'_, '_> {
    /// Branch on `c` and go on where it holds; the block for where it does
    /// not is returned, to `open` once this arm has ended.
    fn arm(&mut self, c: Val) -> Block {
        let (yes, no) = (self.e.new_block(), self.e.new_block());
        self.e.br(c, yes, no);
        self.open(yes, false);
        no
    }

    fn open(&mut self, b: Block, cold: bool) {
        let flags = if cold {
            BlockFlags::COLD
        } else {
            BlockFlags::empty()
        };
        let lp = self.ir_loop(self.bb);
        self.e.begin(b, lp, 0, flags);
    }

    /// The blocks the arms of a step leave through, for the end of the
    /// loop and for its body.
    fn ways(&mut self) -> [Block; 2] {
        [self.e.new_block(), self.e.new_block()]
    }

    /// The arms that left through `ways` go on to bytecode blocks `exit`
    /// and `body`, each as one edge: a loop's header takes one from here.
    fn leave(&mut self, ways: [Block; 2], exit: u32, body: u32) {
        let lp = self.ir_loop(self.bb);
        for (at, to) in ways.into_iter().zip([exit, body]) {
            if self.e.begin(at, lp, 0, BlockFlags::empty()) {
                self.join(self.cfg.blocks[to as usize].start, lp);
                self.jump(to);
            }
        }
    }

    fn i32(&mut self, x: i32) -> Val {
        self.e.k32(Rep::I32, x as u32)
    }

    /// `TFORPREP a`: the closing value to `R[a+2]`, the control to the
    /// first variable, and the position slot `R[a+3]` the form of the
    /// `TFORCALL` it jumps to says: what `op_tforprep` decided the last
    /// time it ran, guarded here, since it decides again on each run.
    pub(super) fn tforprep(&mut self, pc: u32, i: Instruction, to: u32) {
        let a = i.a();
        let target = (pc as i64 + 1 + i.branch_offset() as i64) as usize;
        let (iter, state) = (self.reg(a), self.reg(a + 1));
        let (control, closing) = (self.reg(a + 2), self.reg(a + 3));
        // A closing value makes a to-be-closed variable: the interpreter's.
        let closes = self.e.emit(Op::Truthy, Rep::B1, 0, closing.0, 0, 0);
        let s = self.state();
        let tag = ExitTag::Unsupported as u16;
        (self.e).emit(Op::GuardNot, Rep::None, tag, closes.0, s.0, u32::MAX);
        let form = self.cfg.code[target].op();
        let pos = match form {
            Bc::TFORCALL_NEXT | Bc::TFORCALL_IPAIRS => {
                let next = form == Bc::TFORCALL_NEXT;
                let f = if next {
                    self.links.next_fn
                } else {
                    self.links.ipairs_iter
                };
                // SAFETY: a function's value, which the runtime holds.
                let k = self.konst(unsafe { Value::from_raw(f) });
                let same = self.e.cmp(Op::Cmp, Cc::Eq, iter, k);
                self.guard(same, ExitTag::Type);
                self.table(state);
                if next {
                    let none = self.e.is_nil(control);
                    self.guard(none, ExitTag::Type);
                }
                self.konst(Value::small(if next { 0 } else { TFOR_IPAIRS }))
            }
            _ => self.konst(Value::nil()),
        };
        self.set(a + TFOR_VARS, control);
        self.set(a + 2, closing);
        self.set(a + 3, pos);
        self.jump(to);
    }

    /// `TFORCALL_NEXT a count` and its `TFORLOOP`, to `exit` or `body`:
    /// `next`'s walk at the position in `R[a+3]`, the array part and named
    /// slots inline, and the steps past parts that hold no key: on to the
    /// named slots, or to the end; the other parts by helper through the
    /// slots, its first variable then tested.
    pub(super) fn tforcall_next(&mut self, i: Instruction, exit: u32, body: u32) {
        let (a, count) = (i.a(), i.b());
        let vars = a + TFOR_VARS;
        // Not a table, or no position: the interpreter calls the iterator.
        let s = self.state();
        let (state, pos) = (self.reg(a + 1), self.reg(a + 3));
        let t = self.table(state);
        let p = self.e.check_type(pos, TypeSet::SMALL, ExitTag::Slow, s);
        let zero = self.i32(0);
        let ok = self.e.cmp(Op::Cmp, Cc::Ge, p, zero);
        self.guard(ok, ExitTag::Slow);
        let slow = self.e.new_block();
        let [end, go] = self.ways();
        // A position in the array part is its index.
        let part = self.i32(1 << POS_PART_SHIFT);
        let in_array = self.e.cmp(Op::Cmp, Cc::Lt, p, part);
        let rest = self.arm(in_array);
        let x = self.e.emit(Op::ArrNext, Rep::I32, 0, t.0, p.0, 0);
        let n = self.e.load(Field::ASize, t, 0);
        let found = self.e.cmp(Op::Cmp, Cc::Lo, x, n);
        let past = self.arm(found);
        let arr = self.e.load(Field::Array, t, 0);
        let at = self.e.op1(Op::ZExt32, Rep::I64, x);
        let item = Field::Item as u16;
        let v = (self.e).emit(Op::LoadIdx, Rep::Val, item, arr.0, at.0, u32::MAX);
        let k = self.e.op1(Op::BoxI32, Rep::Val, x);
        let one = self.i32(1);
        let after = self.e.op2(Op::Add, Rep::I32, x, one);
        let after = self.e.op1(Op::BoxI32, Rep::Val, after);
        self.set(a + 3, after);
        self.step_to(go, vars, count, k, v);
        // Past the array part: the integer keys, then the named slots.
        self.open(past, false);
        let ints = (self.e).emit(Op::HashKeys, Rep::I64, 0, t.0, HASH_INTS as u32, 0);
        let none = self.e.k64(Rep::I64, 0);
        let no_ints = self.e.cmp(Op::Cmp, Cc::Eq, ints, none);
        let some = self.arm(no_ints);
        self.named_step(a, count, t, zero, [end, go], slow);
        self.open(some, true);
        self.e.jump(slow);
        // The named slots, part 2, at the index after the position's part.
        self.open(rest, false);
        let named = self.i32(2 << POS_PART_SHIFT);
        let rel = self.e.op2(Op::Sub, Rep::I32, p, named);
        let in_named = self.e.cmp(Op::Cmp, Cc::Lo, rel, part);
        let other = self.arm(in_named);
        self.named_step(a, count, t, rel, [end, go], slow);
        self.open(other, true);
        self.e.jump(slow);
        self.open(slow, true);
        for (r, v) in [(a + 1, state), (a + 3, pos)] {
            self.e.emit(Op::SlotStore, Rep::None, r as u16, v.0, 0, 0);
        }
        (self.e).call(Helper::TforNext, a as u32 | (count as u32) << 8, &[]);
        for r in (vars..vars + count).chain([a + 3]) {
            let v = self.e.emit(Op::SlotLoad, Rep::Val, r as u16, 0, 0, 0);
            self.set(r, v);
        }
        let k = self.reg(vars);
        let done = self.e.is_nil(k);
        self.e.br(done, end, go);
        self.leave([end, go], exit, body);
    }

    /// `next`'s step through the named slots of table `t` (its lock) from
    /// index `rel`: the next one to `go`; past the last, to `end` when the
    /// parts after them hold no key, else the helper at `slow`.
    fn named_step(
        &mut self,
        a: u8,
        count: u8,
        t: Val,
        rel: Val,
        [end, go]: [Block; 2],
        slow: Block,
    ) {
        let vars = a + TFOR_VARS;
        let j = self.e.emit(Op::NamedNext, Rep::I32, 0, t.0, rel.0, 0);
        let none = self.i32(-1);
        let found = self.e.cmp(Op::Cmp, Cc::Ne, j, none);
        let gone = self.arm(found);
        let k = self.e.emit(Op::NamedKey, Rep::Val, 0, t.0, j.0, 0);
        let v = self.e.emit(Op::NamedVal, Rep::Val, 0, t.0, j.0, 0);
        let next = self.i32((2 << POS_PART_SHIFT) + 1);
        let after = self.e.op2(Op::Add, Rep::I32, j, next);
        let after = self.e.op1(Op::BoxI32, Rep::Val, after);
        self.set(a + 3, after);
        self.step_to(go, vars, count, k, v);
        self.open(gone, false);
        let parts = (HASH_STRS | HASH_MISC) as u32;
        let keys = self.e.emit(Op::HashKeys, Rep::I64, 0, t.0, parts, 0);
        let zero = self.e.k64(Rep::I64, 0);
        let empty = self.e.cmp(Op::Cmp, Cc::Eq, keys, zero);
        let more = self.arm(empty);
        let nil = self.konst(Value::nil());
        self.step_to(end, vars, count, nil, nil);
        self.open(more, true);
        self.e.jump(slow);
    }

    /// `TFORCALL_IPAIRS a count` and its `TFORLOOP`, to `exit` or `body`:
    /// the array part inline; a raw nil (there, or past it with no integer
    /// key in the hash part) the end unless the class has `__index`, which
    /// exits; past it otherwise the helper, whose failure (`__index` may
    /// answer) exits.
    pub(super) fn tforcall_ipairs(&mut self, i: Instruction, exit: u32, body: u32) {
        let (a, count) = (i.a(), i.b());
        let vars = a + TFOR_VARS;
        let s = self.state();
        let (state, pos, control) = (self.reg(a + 1), self.reg(a + 3), self.reg(vars));
        let t = self.table(state);
        let mark = self.konst(Value::small(TFOR_IPAIRS));
        let marked = self.e.cmp(Op::Cmp, Cc::Eq, pos, mark);
        self.guard(marked, ExitTag::Type);
        let x = self.e.check_type(control, TypeSet::SMALL, ExitTag::Type, s);
        let one = self.i32(1);
        let tag = ExitTag::Slow as u16;
        let k = self.e.emit(Op::AddOvf, Rep::I32, tag, x.0, one.0, s.0);
        let kv = self.e.op1(Op::BoxI32, Rep::Val, k);
        let n = self.e.load(Field::ASize, t, 0);
        let arr = self.e.load(Field::Array, t, 0);
        let inside = self.e.cmp(Op::Cmp, Cc::Lo, k, n);
        let outside = self.arm(inside);
        let at = self.e.op1(Op::ZExt32, Rep::I64, k);
        let item = Field::Item as u16;
        let v = (self.e).emit(Op::LoadIdx, Rep::Val, item, arr.0, at.0, u32::MAX);
        // Inside the array part a nil is the raw value (`get_int`).
        let raw_nil = self.e.new_block();
        let [end, go] = self.ways();
        let nil = self.e.is_nil(v);
        let found = self.arm(nil);
        self.e.jump(raw_nil);
        self.open(found, false);
        self.step_to(go, vars, count, kv, v);
        // Past the array part: the end of a list a constructor sized, or
        // keys the hash part holds.
        self.open(outside, true);
        let ints = (self.e).emit(Op::HashKeys, Rep::I64, 0, t.0, HASH_INTS as u32, 0);
        let zero = self.e.k64(Rep::I64, 0);
        let no_ints = self.e.cmp(Op::Cmp, Cc::Eq, ints, zero);
        let hashed = self.arm(no_ints);
        self.e.jump(raw_nil);
        self.open(hashed, true);
        let w = self.e.helped(Helper::TforIpairs, 0, &[state, kv], s);
        let last = self.e.is_nil(w);
        let more = self.arm(last);
        let none = self.konst(Value::nil());
        self.step_to(end, vars, count, none, none);
        self.open(more, true);
        self.step_to(go, vars, count, kv, w);
        self.open(raw_nil, false);
        (self.e).no_mm(t, MetamethodBits::INDEX, ExitTag::Slow, s);
        let none = self.konst(Value::nil());
        self.step_to(end, vars, count, none, none);
        self.leave([end, go], exit, body);
    }

    /// The loop variables `(k, v, nil...)`, then `to`.
    fn step_to(&mut self, to: Block, vars: u8, count: u8, k: Val, v: Val) {
        self.set(vars, k);
        if count > 1 {
            self.set(vars + 1, v);
        }
        for r in vars + 2..vars + count.max(2) {
            let nil = self.konst(Value::nil());
            self.set(r, nil);
        }
        self.e.jump(to);
    }

    /// `TFORCALL a count` with another iterator: `R[a](R[a+1], R[a+4])`,
    /// its results the loop variables.
    pub(super) fn tforcall(&mut self, pc: u32, i: Instruction) {
        let (a, count) = (i.a(), i.b());
        let vars = a + TFOR_VARS;
        let (iter, state, control) = (self.reg(a), self.reg(a + 1), self.reg(vars));
        let first = self.call_mm(pc, iter, &[state, control], vars, count + 1);
        if count == 1 {
            self.set(vars, first);
        }
        self.lazy_exit(pc + 1);
    }
}
