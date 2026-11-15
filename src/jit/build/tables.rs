//! Table, upvalue and global access. A constant-key site whose inline
//! cache holds the key in the receiver's own slot reads and writes that
//! slot under a shape guard; a small-integer key goes to the array part
//! under a bounds guard; everything else takes a helper, whose failure (a
//! metamethod to call, an error) leaves for the interpreter.

use super::Builder;
use crate::env::Shape;
use crate::env::function::InlineCache;
use crate::env::shape::Kind;
use crate::env::table::SlotLoc;
use crate::env::value::NIL;
use crate::instruction::{Instruction, Op as Bc};
use crate::jit::feedback as fb;
use crate::jit::ir::fields::Field;
use crate::jit::ir::ops::{Cc, ExitTag, Helper, Op};
use crate::jit::ir::types::{Rep, TypeSet};
use crate::jit::ir::{BlockFlags, Val};
use crate::jit::layout;

impl<'gc> Builder<'_, 'gc> {
    fn load(&mut self, f: Field, base: Val, k: u32) -> Val {
        let rep = Field::info(f as u16).rep;
        self.e.emit(Op::Load, rep, f as u16, base.0, u32::MAX, k)
    }

    fn guard(&mut self, c: Val, tag: ExitTag) {
        let s = self.state();
        self.e
            .emit(Op::Guard, Rep::None, tag as u16, c.0, s.0, u32::MAX);
    }

    /// The lock of the table `v` is, checked to be one.
    fn table(&mut self, v: Val) -> Val {
        let s = self.state();
        self.e.check_type(v, TypeSet::TAB, ExitTag::Type, s)
    }

    /// Run `slow` in a cold block where `c` holds, then go on.
    fn unless(&mut self, c: Val, slow: impl FnOnce(&mut Self)) {
        match self.e.known(c) {
            Some(false) => return,
            Some(true) => return slow(self),
            None => {}
        }
        let lp = self.e.g.block(self.e.cur).lp;
        let (cold, on) = (self.e.new_block(), self.e.new_block());
        self.e.br(c, cold, on);
        self.e.begin(cold, lp, 0, BlockFlags::COLD);
        slow(self);
        if !self.e.cur.is_none() {
            self.e.jump(on);
        }
        self.e.begin(on, lp, 0, BlockFlags::empty());
    }

    /// The running closure.
    fn closure(&mut self) -> Val {
        let base = self.e.emit(Op::Base, Rep::Ptr, 0, 0, 0, 0);
        let f = self.load(Field::FrameFn, base, 0);
        self.e.unboxed(f, TypeSet::FUN)
    }

    /// The value of the shared upvalue `k`, and where it is.
    fn cell(&mut self, k: u8) -> (Val, Val) {
        let c = self.closure();
        let cell = self.load(Field::UpvalCell, c, k as u32);
        let at = self.load(Field::CellAt, cell, 0);
        (cell, at)
    }

    /// The receiver of a constant-key site: a register, a by-value upvalue,
    /// or a shared cell's value.
    fn receiver(&mut self, i: Instruction) -> Val {
        match i.op().unquickened() {
            Bc::GETTABUP | Bc::SETTABUP => {
                let c = self.closure();
                self.load(Field::Upval, c, i.b() as u32)
            }
            Bc::GETTABUP_REF | Bc::SETTABUP_REF => {
                let (_, at) = self.cell(i.b());
                self.load(Field::CellValue, at, 0)
            }
            _ => self.reg(i.b()),
        }
    }

    pub(super) fn getupval(&mut self, i: Instruction) {
        let v = match i.op() {
            Bc::GETUPVAL => {
                let c = self.closure();
                self.load(Field::Upval, c, i.b() as u32)
            }
            _ => {
                let (_, at) = self.cell(i.b());
                self.load(Field::CellValue, at, 0)
            }
        };
        self.set(i.a(), v);
    }

    pub(super) fn setupval(&mut self, i: Instruction) {
        let v = self.reg(i.a());
        let (cell, at) = self.cell(i.b());
        self.e
            .emit(Op::Store, Rep::None, Field::CellValue as u16, at.0, v.0, 0);
        if self.e.fact(v).set.intersects(TypeSet::HEAP) {
            self.e.call(Helper::UpvalBarrier, 0, &[cell]);
        }
    }

    /// `t` checked to have `shape`, which the region then keeps alive.
    fn guard_shape(&mut self, t: Val, shape: Shape<'gc>) {
        if !self.shapes.iter().any(|s| s.as_word() == shape.as_word()) {
            self.shapes.push(shape);
        }
        let s = self.load(Field::Shape, t, 0);
        let k = self.e.k64(Rep::Ptr, shape.as_word());
        let same = self.e.cmp(Op::Cmp, Cc::Eq, s, k);
        self.guard(same, ExitTag::Shape);
    }

    /// The named slot at `loc` of table `t`: the access's base, field and
    /// index.
    fn slot(&mut self, t: Val, loc: SlotLoc) -> (Val, Field, u32) {
        if loc.spilled() {
            let spill = self.load(Field::Spill, t, 0);
            (spill, Field::Spilled, loc.offset() as u32 / 8)
        } else {
            (
                t,
                Field::Inline,
                (loc.offset() - layout::T_INLINE) as u32 / 8,
            )
        }
    }

    fn is_nil(&mut self, v: Val) -> Val {
        let nil = self.e.k64(Rep::Val, NIL);
        self.e.cmp(Op::Cmp, Cc::Eq, v, nil)
    }

    /// Where `v`, read from table `t`, is nil, leave if the table has a
    /// metatable: `__index` or `__newindex` may answer there. The class is
    /// read only for a nil.
    fn nil_guard(&mut self, t: Val, v: Val) {
        let nil = self.is_nil(v);
        self.unless(nil, |b| {
            let shape = b.load(Field::Shape, t, 0);
            let class = b.load(Field::ShapeClass, shape, 0);
            let zero = b.e.k64(Rep::I64, 0);
            let plain = b.e.cmp(Op::Cmp, Cc::Eq, class, zero);
            b.guard(plain, ExitTag::Bounds);
        });
    }

    /// The store's barrier, unless the value is never a heap object: the
    /// helper where the table is not gray.
    fn barrier(&mut self, t: Val, v: Val) {
        if !self.e.fact(v).set.intersects(TypeSet::HEAP) {
            return;
        }
        let header = self.load(Field::GcHeader, t, 0);
        let one = self.e.k64(Rep::I64, 1);
        let gray = self.e.op2(Op::And, Rep::I64, header, one);
        let zero = self.e.k64(Rep::I64, 0);
        let due = self.e.cmp(Op::Cmp, Cc::Eq, gray, zero);
        self.unless(due, |b| {
            b.e.call(Helper::Barrier, 0, &[t]);
        });
    }

    /// The array part's cell and the checked index of a small-integer key
    /// inside it, for a site that saw only tables with such keys in range.
    fn element(&mut self, pc: u32, recv: Val, key: Val) -> Option<(Val, Val, Val)> {
        let seen = self.proto.feedback.bits(pc as usize);
        let key_fact = self.e.fact(key);
        let small = !key_fact.pending() && TypeSet::SMALL.contains(key_fact.set);
        // `ARR_MISS` shares `BIGINT`'s bit; `OTHER` is also a receiver that
        // is no table.
        let other = fb::KINDS & !fb::SMALL;
        if seen & other != 0 || (seen & fb::SMALL == 0 && !small) {
            return None;
        }
        let state = self.state();
        let t = self.table(recv);
        let i = self.e.check_type(key, TypeSet::SMALL, ExitTag::Type, state);
        let n = self.load(Field::ASize, t, 0);
        let inside = self.e.cmp(Op::Cmp, Cc::Lo, i, n);
        let tag = ExitTag::Bounds as u16;
        let i = self
            .e
            .emit(Op::Guard, Rep::I32, tag, inside.0, state.0, i.0);
        let arr = self.load(Field::Array, t, 0);
        let at = self.e.op1(Op::ZExt32, Rep::I64, i);
        Some((t, arr, at))
    }

    /// `R[a] = R[b][R[c]]`.
    pub(super) fn gettable(&mut self, pc: u32, i: Instruction) {
        let (recv, key) = (self.reg(i.b()), self.reg(i.c()));
        let v = match self.element(pc, recv, key) {
            Some((t, arr, at)) => {
                let item = Field::Item as u16;
                let v = self
                    .e
                    .emit(Op::LoadIdx, Rep::Val, item, arr.0, at.0, u32::MAX);
                self.nil_guard(t, v);
                v
            }
            None => self.helped(Helper::Index, 0, &[recv, key]),
        };
        self.set(i.a(), v);
    }

    /// `R[b][R[c]] = R[a]`.
    pub(super) fn settable(&mut self, pc: u32, i: Instruction) {
        let (recv, key, v) = (self.reg(i.b()), self.reg(i.c()), self.reg(i.a()));
        match self.element(pc, recv, key) {
            Some((t, arr, at)) => {
                let item = Field::Item as u16;
                // `__newindex` may fire over a nil.
                let old = self
                    .e
                    .emit(Op::LoadIdx, Rep::Val, item, arr.0, at.0, u32::MAX);
                self.nil_guard(t, old);
                self.barrier(t, v);
                self.e.emit(Op::StoreIdx, Rep::None, item, arr.0, at.0, v.0);
            }
            None => {
                self.helped(Helper::NewIndex, 1, &[recv, key, v]);
            }
        }
    }

    /// `R[a] = {}` from the site's template, then the collector's check:
    /// past its threshold the code leaves after the instruction, for the
    /// interpreter to collect.
    pub(super) fn newtable(&mut self, pc: u32, i: Instruction) {
        let template = &self.proto.templates[i.d() as usize] as *const _ as u64;
        let template = self.e.k64(Rep::I64, template);
        let t = self.e.call(Helper::NewTable, 0, &[template]);
        self.set(i.a(), t);
        let metrics = self.e.k64(Rep::Ptr, self.metrics);
        let used = self.load(Field::GcUsed, metrics, 0);
        let limit = self.load(Field::GcLimit, metrics, 0);
        let below = self.e.cmp(Op::Cmp, Cc::Lo, used, limit);
        let after = self.state_at(pc + 1);
        let tag = ExitTag::Gc as u16;
        self.e
            .emit(Op::Guard, Rep::None, tag, below.0, after.0, u32::MAX);
    }

    /// The cache entry of a constant-key site, where it is an own slot the
    /// region may use: of a shape still current.
    fn own(&self, ic: u16) -> Option<(Shape<'gc>, SlotLoc)> {
        match self.proto.feedback.cache(ic as usize).get() {
            InlineCache::Own { shape, loc, .. } if !shape.is_deprecated() => Some((shape, loc)),
            _ => None,
        }
    }

    /// `R[a] = recv[K[e]]`.
    pub(super) fn getfield(&mut self, i: Instruction) {
        let (dst, _, ic, e) = i.abde();
        let recv = self.receiver(i);
        let v = match self.own(ic) {
            Some((shape, loc)) => {
                let t = self.table(recv);
                self.guard_shape(t, shape);
                let (base, f, k) = self.slot(t, loc);
                let v = self.load(f, base, k);
                // A nil is the answer only where no `__index` may give one.
                if shape.mt_cache().is_some() {
                    let nil = self.is_nil(v);
                    let s = self.state();
                    let tag = ExitTag::Slow as u16;
                    self.e
                        .emit(Op::GuardNot, Rep::None, tag, nil.0, s.0, u32::MAX);
                }
                v
            }
            None => {
                let key = self.konst(self.proto.constants[e as usize]);
                self.helped(Helper::Index, 0, &[recv, key])
            }
        };
        self.set(dst, v);
    }

    /// `recv[K[e]] = R[a]`.
    pub(super) fn setfield(&mut self, i: Instruction) {
        let (src, _, ic, e) = i.abde();
        let recv = self.receiver(i);
        let v = self.reg(src);
        let kv = self.proto.constants[e as usize];
        let held = self.e.fact(v).set;
        let own = self.own(ic).filter(|&(shape, loc)| {
            // A metatable's keys tell its class and a watched holder's its
            // dependents: the helper's store does. So does it store to a slot
            // that tracks its tables' shape, or whose kind takes more than
            // one compare to test.
            let f = shape.field(loc.slot(shape));
            let fits = TypeSet::of_classes(f.kind().classes());
            let testable =
                fits == TypeSet::ANY || fits.contains(held) || fits.unboxed() != Rep::None;
            shape.as_mt().is_none()
                && !shape.is_watched()
                && f.kind() != Kind::NONE
                && f.shape().is_none()
                && testable
        });
        let Some((shape, loc)) = own else {
            let key = self.konst(kv);
            self.helped(Helper::NewIndex, 0, &[recv, key, v]);
            return;
        };
        let t = self.table(recv);
        self.guard_shape(t, shape);
        // The slot's kind covers what it holds: a value outside it leaves,
        // and the interpreter's store widens the kind.
        let kind = shape.field(loc.slot(shape)).kind();
        let fits = TypeSet::of_classes(kind.classes());
        if kind != Kind::ANY && !fits.contains(held) {
            let s = self.state();
            self.e.check_type(v, fits, ExitTag::Type, s);
        }
        let (base, f, k) = self.slot(t, loc);
        // `__newindex` fires only on a nil slot.
        if shape.mt_cache().is_some() {
            let old = self.load(f, base, k);
            let nil = self.is_nil(old);
            let s = self.state();
            let tag = ExitTag::Slow as u16;
            self.e
                .emit(Op::GuardNot, Rep::None, tag, nil.0, s.0, u32::MAX);
        }
        self.barrier(t, v);
        self.e.emit(Op::Store, Rep::None, f as u16, base.0, v.0, k);
    }
}
