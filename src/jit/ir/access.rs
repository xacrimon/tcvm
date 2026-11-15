//! Table access as the emitter writes it: the cold arms, the class and
//! barrier tests, and the rewrites of the generic index and field ops. A
//! small-integer key goes to the array part under a bounds guard; a
//! constant key to its slot under a test of each shape its site record
//! names; anything else takes a helper, whose failure (a metamethod to
//! call, an error) leaves for the interpreter.

use super::emit::Emitter;
use super::fields::Field;
use super::ops::{Cc, ExitTag, HELPER_FAIL_VAL, Helper, Op};
use super::sites::{Case, FieldSite, Form};
use super::types::{Rep, TypeSet};
use super::{Inst, Val};
use crate::env::MetamethodBits;
use crate::env::shape::Kind;
use crate::env::table::SlotLoc;
use crate::env::value::NIL;
use crate::jit::bug;
use crate::jit::layout;

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

    /// The rewrite of `GetField`/`SetField`, the operands' facts final: one
    /// arm per shape of the site's record, the last under a guard that
    /// leaves for a shape the site has not seen; the helper where the
    /// record has none.
    pub(super) fn field(&mut self, i: &Inst) -> Val {
        let get = i.op == Op::GetField;
        let (recv, v, state) = if get {
            (Val(i.a), Val::NONE, Val(i.b))
        } else {
            (Val(i.a), Val(i.b), Val(i.c))
        };
        if self.fact(recv).pending() || !get && self.fact(v).pending() {
            return self.append(*i);
        }
        let site = self.sites.fields[i.aux as usize];
        let n = site.n as usize;
        if n == 0 || !self.fact(recv).set.intersects(TypeSet::TAB) {
            return self.field_helped(&site, recv, v, state);
        }
        let t = self.check_type(recv, TypeSet::TAB, ExitTag::Type, state);
        let shape = self.load(Field::Shape, t, 0);
        let mut on = None;
        for (k, case) in site.cases[..n].iter().enumerate() {
            let want = self.k64(Rep::Ptr, case.shape);
            let same = self.cmp(Op::Cmp, Cc::Eq, shape, want);
            let known = self.known(same);
            let last = k == n - 1 || known == Some(true);
            if known == Some(false) && !last {
                continue;
            }
            self.sites.keep(case.shape);
            if last {
                // Known already, this is the guard that made it so.
                let tag = ExitTag::Shape as u16;
                let base = self.emit(Op::Guard, Rep::Ptr, tag, same.0, state.0, t.0);
                let r = self.field_case(&site, case, recv, t, base, v, state);
                let Some(mut on) = on else {
                    return r;
                };
                self.leave(&mut on, [r]);
                return self
                    .join(on)
                    .map_or(Val::NONE, |[r]| if get { r } else { Val::NONE });
            }
            if on.is_none() {
                on = Some(self.new_join::<1>());
            }
            let (hit, miss) = self.fork(same, false, false);
            if self.arm(hit) {
                // The slots are this shape's only in this arm.
                let base = self.proved(same, true, t, TypeSet::ANY);
                let r = self.field_case(&site, case, recv, t, base, v, state);
                self.leave(on.as_mut().unwrap(), [r]);
            }
            self.arm(miss);
        }
        bug!("a field site without a last case")
    }

    /// One case's access through `base`, the table as its shape test left
    /// it: a load's value; for a store the table, which every arm agrees on.
    #[allow(clippy::too_many_arguments)]
    fn field_case(
        &mut self,
        site: &FieldSite,
        case: &Case,
        recv: Val,
        t: Val,
        base: Val,
        v: Val,
        state: Val,
    ) -> Val {
        if v.is_none() {
            return match case.form {
                Form::Own if !case.classed => self.slot_load(base, case.loc),
                Form::Own => {
                    // A nil is the answer only where no `__index` gives one.
                    let x = self.slot_load(base, case.loc);
                    let nil = self.is_nil(x);
                    let mut on = self.new_join::<1>();
                    let walk = self.fork_or_leave(nil, true, &mut on, [x]);
                    if self.arm(walk) {
                        let found = self.field_helped(site, recv, v, state);
                        self.leave(&mut on, [found]);
                    }
                    self.join(on).map_or(x, |[r]| r)
                }
                Form::Absent if !case.classed => self.k64(Rep::Val, NIL),
                Form::Absent | Form::Helper => self.field_helped(site, recv, v, state),
                Form::Transition => bug!("a load that adds a key"),
            };
        }
        match case.form {
            Form::Own => {
                self.kind_test(v, case, state);
                let (b, f, k) = self.slot(base, case.loc);
                // `__newindex` fires only on a nil slot.
                if case.classed {
                    let old = self.load(f, b, k);
                    self.nil_guard(t, old, MetamethodBits::NEWINDEX, ExitTag::Mm, state);
                }
                self.barrier(t, v);
                self.emit(Op::Store, Rep::None, f as u16, b.0, v.0, k);
            }
            Form::Transition => {
                if case.classed {
                    self.no_mm(t, MetamethodBits::NEWINDEX, ExitTag::Mm, state);
                }
                // Storing nil to an absent key adds nothing.
                let mut on = None;
                let add = if self.fact(v).set.intersects(TypeSet::NIL) {
                    let nil = self.k64(Rep::Val, NIL);
                    let some = self.cmp(Op::Cmp, Cc::Ne, v, nil);
                    let mut skip = self.new_join::<0>();
                    let add = self.fork_or_leave(some, false, &mut skip, []);
                    on = Some(skip);
                    self.arm(add)
                } else {
                    true
                };
                if add {
                    self.kind_test(v, case, state);
                    self.barrier(t, v);
                    let (b, f, k) = self.slot(base, case.loc);
                    self.emit(Op::Store, Rep::None, f as u16, b.0, v.0, k);
                    let to = self.k64(Rep::Ptr, case.to);
                    self.sites.keep(case.to);
                    self.emit(Op::Store, Rep::None, Field::Shape as u16, t.0, to.0, 0);
                    if let Some(skip) = &mut on {
                        self.leave(skip, []);
                    }
                }
                if let Some(skip) = on {
                    self.join(skip);
                }
            }
            Form::Helper => _ = self.field_helped(site, recv, v, state),
            Form::Absent => bug!("a store to a key its shape lacks"),
        }
        t
    }

    /// A field access by the helper, a store where `v` is one: its result,
    /// or an exit where it failed.
    fn field_helped(&mut self, site: &FieldSite, recv: Val, v: Val, state: Val) -> Val {
        let key = self.emit(Op::KObj, Rep::Val, TypeSet::STR.bits(), site.key, 0, 0);
        if v.is_none() {
            self.helped(Helper::Index, 0, &[recv, key], state)
        } else {
            self.helped(Helper::NewIndex, 0, &[recv, key, v], state);
            Val::NONE
        }
    }

    /// The named slot at `loc` of the table `base`: the access's base,
    /// field and index.
    fn slot(&mut self, base: Val, loc: SlotLoc) -> (Val, Field, u32) {
        if loc.spilled() {
            let spill = self.load(Field::Spill, base, 0);
            (spill, Field::Spilled, loc.offset() as u32 / 8)
        } else {
            let k = (loc.offset() - layout::T_INLINE) as u32 / 8;
            (base, Field::Inline, k)
        }
    }

    fn slot_load(&mut self, base: Val, loc: SlotLoc) -> Val {
        let (b, f, k) = self.slot(base, loc);
        self.load(f, b, k)
    }

    /// Leave with `Type` unless storing `v` keeps what the case's slot
    /// says: its upper word in the kind's arc, or a table of the shape the
    /// slot tracks. The interpreter's store widens.
    fn kind_test(&mut self, v: Val, case: &Case, state: Val) {
        let held = self.fact(v).set;
        let fits = TypeSet::of_classes(case.kind.classes());
        if case.kind == Kind::ANY || fits.contains(held) {
            return;
        }
        let tag = ExitTag::Type as u16;
        let tracked = |e: &mut Self, table: Val| {
            let shape = e.load(Field::Shape, table, 0);
            let want = e.k64(Rep::Ptr, case.expect);
            let same = e.cmp(Op::Cmp, Cc::Eq, shape, want);
            e.emit(Op::Guard, Rep::None, tag, same.0, state.0, u32::MAX);
        };
        if case.expect != 0 && TypeSet::TAB.contains(held) {
            let table = self.check_type(v, TypeSet::TAB, ExitTag::Type, state);
            return tracked(self, table);
        }
        let word = case.kind.to_word();
        let k = self.k64(Rep::I64, 32);
        let hi = self.op2(Op::Shr, Rep::I64, v, k);
        let hi = self.op1(Op::Trunc32, Rep::I32, hi);
        let lo = self.k32(Rep::I32, (word >> 32) as u32);
        let past = self.op2(Op::Sub, Rep::I32, hi, lo);
        let span = self.k32(Rep::I32, word as u32);
        if case.expect == 0 {
            let inside = self.cmp(Op::Cmp, Cc::Ls, past, span);
            self.emit(Op::Guard, Rep::None, tag, inside.0, state.0, u32::MAX);
            return;
        }
        let outside = self.cmp(Op::Cmp, Cc::Hi, past, span);
        let mut on = self.new_join::<0>();
        let other = self.fork_or_leave(outside, false, &mut on, []);
        if self.arm(other) {
            let table = self.check_type(v, TypeSet::TAB, ExitTag::Type, state);
            tracked(self, table);
            self.leave(&mut on, []);
        }
        self.join(on);
    }
}
