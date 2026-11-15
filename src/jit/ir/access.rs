//! Table access as the emitter writes it: the cold arms, the class and
//! barrier tests, and the rewrite of the generic `GetIndex`/`SetIndex`. A
//! small-integer key goes to the array part under a bounds guard; anything
//! else takes a helper, whose failure (a metamethod to call, an error)
//! leaves for the interpreter.

use super::emit::Emitter;
use super::fields::Field;
use super::ops::{Cc, ExitTag, HELPER_FAIL_VAL, Helper, Op};
use super::types::{Rep, TypeSet};
use super::{Inst, Val};
use crate::env::MetamethodBits;
use crate::env::value::NIL;

/// `aux` of a `GetIndex`/`SetIndex` whose site saw only small-integer keys
/// inside the array part.
pub(crate) const ARRAY: u16 = 1;

impl Emitter {
    pub(crate) fn load(&mut self, f: Field, base: Val, k: u32) -> Val {
        let rep = Field::info(f as u16).rep;
        self.emit(Op::Load, rep, f as u16, base.0, u32::MAX, k)
    }

    /// Run `slow` in a cold block where `c` holds, then go on. A frame
    /// state used past it must stand before it.
    pub(crate) fn unless(&mut self, c: Val, slow: impl FnOnce(&mut Self)) {
        match self.known(c) {
            Some(false) => return,
            Some(true) => return slow(self),
            None => {}
        }
        let mut on = self.new_join::<0>();
        let cold = self.fork_or_leave(c, true, &mut on, []);
        if self.arm(cold) {
            slow(self);
            if !self.cur.is_none() {
                self.leave(&mut on, []);
            }
        }
        self.join(on);
    }

    pub(crate) fn is_nil(&mut self, v: Val) -> Val {
        let nil = self.k64(Rep::Val, NIL);
        self.cmp(Op::Cmp, Cc::Eq, v, nil)
    }

    /// A helper's result, or an exit to `state` where it failed: the
    /// interpreter runs the instruction, which the helper left untouched.
    pub(crate) fn helped(&mut self, h: Helper, imm: u32, args: &[Val], state: Val) -> Val {
        let r = self.call(h, imm, args);
        let fail = self.k64(Rep::I64, HELPER_FAIL_VAL);
        let ok = self.cmp(Op::Cmp, Cc::Ne, r, fail);
        let tag = ExitTag::Mm as u16;
        self.emit(Op::Guard, Rep::None, tag, ok.0, state.0, u32::MAX);
        r
    }

    /// Leave with `tag` if the class of table `t` has a metamethod of
    /// `bits` now: a table without a class has none.
    pub(crate) fn no_mm(&mut self, t: Val, bits: MetamethodBits, tag: ExitTag, state: Val) {
        let shape = self.load(Field::Shape, t, 0);
        let class = self.load(Field::ShapeClass, shape, 0);
        let zero = self.k64(Rep::I64, 0);
        let classed = self.cmp(Op::Cmp, Cc::Ne, class, zero);
        self.unless(classed, |e| {
            // The class is one only in this arm.
            let class = e.proved(classed, true, class, TypeSet::ANY);
            let have = e.load(Field::ClassBits, class, 0);
            let mask = e.k32(Rep::I32, bits.bits());
            let hit = e.op2(Op::And, Rep::I32, have, mask);
            let zero = e.k32(Rep::I32, 0);
            let none = e.cmp(Op::Cmp, Cc::Eq, hit, zero);
            e.emit(Op::Guard, Rep::None, tag as u16, none.0, state.0, u32::MAX);
        });
    }

    /// Where `v`, read from table `t`, is nil, leave if the table's class
    /// has `bit`: the metamethod may answer there. The class is read only
    /// for a nil.
    pub(crate) fn nil_guard(
        &mut self,
        t: Val,
        v: Val,
        bit: MetamethodBits,
        tag: ExitTag,
        state: Val,
    ) {
        let nil = self.is_nil(v);
        self.unless(nil, |e| e.no_mm(t, bit, tag, state));
    }

    /// The store's barrier, unless the value is never a heap object: the
    /// helper where the table is not gray.
    pub(crate) fn barrier(&mut self, t: Val, v: Val) {
        if !self.fact(v).set.intersects(TypeSet::HEAP) {
            return;
        }
        let header = self.load(Field::GcHeader, t, 0);
        let one = self.k64(Rep::I64, 1);
        let gray = self.op2(Op::And, Rep::I64, header, one);
        let zero = self.k64(Rep::I64, 0);
        let due = self.cmp(Op::Cmp, Cc::Eq, gray, zero);
        self.unless(due, |e| {
            e.call(Helper::Barrier, 0, &[t]);
        });
    }

    /// `recv[key] = v`, generic.
    pub(crate) fn set_index(&mut self, recv: Val, key: Val, v: Val, aux: u16, state: Val) {
        let at = self.g.extra.len() as u32;
        self.g.extra.extend([recv.0, key.0, v.0]);
        self.emit(Op::SetIndex, Rep::None, aux, at, 3, state.0);
    }

    /// The rewrite of `GetIndex`/`SetIndex`, the key's fact final: a helper
    /// call, or the array part under its checks, with the cold arms a nil
    /// there and a store's barrier take.
    pub(super) fn index(&mut self, i: &Inst) -> Val {
        let get = i.op == Op::GetIndex;
        let (recv, key, v, state) = if get {
            (Val(i.a), Val(i.b), Val::NONE, Val(i.c))
        } else {
            let r = self.g.run(i.a, i.b);
            (Val(r[0]), Val(r[1]), Val(r[2]), Val(i.c))
        };
        let kf = self.fact(key);
        if kf.pending() {
            return self.append(*i);
        }
        if i.aux & ARRAY == 0 || !kf.set.intersects(TypeSet::SMALL) {
            return if get {
                self.helped(Helper::Index, 0, &[recv, key], state)
            } else {
                self.helped(Helper::NewIndex, 1, &[recv, key, v], state);
                Val::NONE
            };
        }
        let t = self.check_type(recv, TypeSet::TAB, ExitTag::Type, state);
        let k = self.check_type(key, TypeSet::SMALL, ExitTag::Type, state);
        let n = self.load(Field::ASize, t, 0);
        let inside = self.cmp(Op::Cmp, Cc::Lo, k, n);
        let tag = ExitTag::Bounds as u16;
        let k = self.emit(Op::Guard, Rep::I32, tag, inside.0, state.0, k.0);
        let arr = self.load(Field::Array, t, 0);
        let at = self.op1(Op::ZExt32, Rep::I64, k);
        let item = Field::Item as u16;
        // A nil is the answer, and a store over one plain, only where no
        // metamethod may fire.
        let old = self.emit(Op::LoadIdx, Rep::Val, item, arr.0, at.0, u32::MAX);
        if get {
            self.nil_guard(t, old, MetamethodBits::INDEX, ExitTag::Bounds, state);
            return old;
        }
        self.nil_guard(t, old, MetamethodBits::NEWINDEX, ExitTag::Bounds, state);
        self.barrier(t, v);
        self.emit(Op::StoreIdx, Rep::None, item, arr.0, at.0, v.0);
        Val::NONE
    }
}
