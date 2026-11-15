//! Table, upvalue and global access. A constant-key site whose inline
//! cache holds the key in the receiver's own slot reads and writes that
//! slot under a shape guard; a small-integer key goes to the array part
//! under a bounds guard; everything else takes a helper, whose failure (a
//! metamethod to call, an error) leaves for the interpreter.

use super::{Builder, DepOn};
use crate::env::function::InlineCache;
use crate::env::shape::Kind;
use crate::env::shape::MtCache;
use crate::env::table::SlotLoc;
use crate::env::value::NIL;
use crate::env::{LuaString, MetamethodBits, Shape, Table, Value};
use crate::instruction::{Instruction, Op as Bc};
use crate::jit::deps;
use crate::jit::feedback as fb;
use crate::jit::ir::access::ARRAY;
use crate::jit::ir::fields::Field;
use crate::jit::ir::ops::{Cc, ExitTag, Helper, Op};
use crate::jit::ir::types::{Rep, TypeSet};
use crate::jit::ir::{BlockFlags, Val};
use crate::jit::layout;

/// Classes and holders an access walks at compile time.
const MAX_WALK: usize = 8;

/// What a read gives where the receiver lacks the key or holds nil.
#[derive(Clone, Copy)]
enum Found<'gc> {
    /// Nil: no `__index` past the tables without the key.
    Nil,
    /// A slot of a watched holder that no store changed.
    Const(Value<'gc>),
    /// The slot at `loc` of a holder of this shape: a watched one's, or
    /// (guarded) an unwatched one's while it has the shape. Nil walks on.
    Slot(Table<'gc>, Shape<'gc>, bool, SlotLoc),
    /// Past what the region may assume: the generic walk.
    Helper,
}

impl<'gc> Builder<'_, 'gc> {
    fn load(&mut self, f: Field, base: Val, k: u32) -> Val {
        self.e.load(f, base, k)
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

    /// Whether `x == y` for values of any kind. The same word is equal
    /// unless it is a NaN; different words, in a cold block, go to the helper,
    /// which compares numbers and big integers by value, after leaving for
    /// what `__eq` may answer: either operand's, of two tables or userdata.
    pub(super) fn equal(&mut self, x: Val, y: Val) -> Val {
        let same = self.e.cmp(Op::Cmp, Cc::Eq, x, y);
        if self.e.known(same).is_some() {
            return same;
        }
        let yes = self.e.k32(Rep::B1, 1);
        self.state();
        let lp = self.e.g.block(self.e.cur).lp;
        let [word, float, differ, on] = [(); 4].map(|_| self.e.new_block());
        self.e.br(same, word, differ);
        self.e.begin(word, lp, 0, BlockFlags::empty());
        let boxes = self.e.k64(Rep::I64, crate::env::value::BOX);
        let is_float = self.e.cmp(Op::Cmp, Cc::Lo, x, boxes);
        self.e.br(is_float, float, on);
        // A float is a NaN where, its sign shifted out, it is past infinity.
        self.e.begin(float, lp, 0, BlockFlags::COLD);
        let one = self.e.k64(Rep::I64, 1);
        let unsigned = self.e.op2(Op::Shl, Rep::I64, x, one);
        let infinity = self.e.k64(Rep::I64, 0xffe0_0000_0000_0000);
        let number = self.e.cmp(Op::Cmp, Cc::Ls, unsigned, infinity);
        self.e.jump(on);
        self.e.begin(differ, lp, 0, BlockFlags::COLD);
        for v in [x, y] {
            let table = self
                .e
                .emit(Op::IsType, Rep::B1, TypeSet::TAB.bits(), v.0, 0, 0);
            let s = self.state();
            self.e.unless(table, |e| {
                let t = e.unboxed(v, TypeSet::TAB);
                e.no_mm(t, MetamethodBits::EQ, ExitTag::Slow, s);
            });
            let udata = self
                .e
                .emit(Op::IsType, Rep::B1, TypeSet::UDATA.bits(), v.0, 0, 0);
            let s = self.state();
            let tag = ExitTag::Slow as u16;
            (self.e).emit(Op::GuardNot, Rep::None, tag, udata.0, s.0, u32::MAX);
        }
        let r = self.e.call(Helper::RawEqual, 0, &[x, y]);
        let zero = self.e.k32(Rep::I32, 0);
        let equal = self.e.cmp(Op::Cmp, Cc::Ne, r, zero);
        self.e.jump(on);
        self.e.begin(on, lp, 0, BlockFlags::empty());
        self.e.param(Rep::B1, None, &[yes, number, equal])
    }

    /// `SETLIST`: the items, in their slots, into the table in `R[a]`.
    pub(super) fn setlist(&mut self, i: Instruction) {
        let (a, n) = (i.a(), i.b());
        if n == 0 {
            return self.deopt(ExitTag::Unsupported);
        }
        for r in a..=a + n {
            let v = self.reg(r);
            self.e.emit(Op::SlotStore, Rep::None, r as u16, v.0, 0, 0);
        }
        let imm = a as u32 | (n as u32) << 8 | (i.d() as u32) << 16;
        self.e.call(Helper::SetList, imm, &[]);
    }

    /// Whether the site at `pc` speculates the array part: it never left
    /// the interpreter's fast path, which is a small-integer key inside it.
    fn array_site(&self, pc: u32) -> u16 {
        let seen = self.proto.feedback.bits(pc as usize);
        if seen & fb::KINDS == 0 { ARRAY } else { 0 }
    }

    /// `R[a] = R[b][R[c]]`.
    pub(super) fn gettable(&mut self, pc: u32, i: Instruction) {
        let (recv, key) = (self.reg(i.b()), self.reg(i.c()));
        let (aux, s) = (self.array_site(pc), self.state());
        let v = self.e.emit(Op::GetIndex, Rep::Val, aux, recv.0, key.0, s.0);
        self.set(i.a(), v);
    }

    /// `R[b][R[c]] = R[a]`.
    pub(super) fn settable(&mut self, pc: u32, i: Instruction) {
        let (recv, key, v) = (self.reg(i.b()), self.reg(i.c()), self.reg(i.a()));
        let (aux, s) = (self.array_site(pc), self.state());
        self.e.set_index(recv, key, v, aux, s);
    }

    /// `R[a] = {}` from the site's template, then the collector's check:
    /// past its threshold the code leaves after the instruction, for the
    /// interpreter to collect.
    pub(super) fn newtable(&mut self, pc: u32, i: Instruction) {
        let template = &self.proto.templates[i.d() as usize] as *const _ as u64;
        let template = self.e.k64(Rep::I64, template);
        let t = self.e.call(Helper::NewTable, 0, &[template]);
        self.set(i.a(), t);
        self.gc_check(pc);
    }

    /// The cache entry of a constant-key site, where it is an own slot the
    /// region may use: of a shape still current.
    fn own(&self, ic: u16) -> Option<(Shape<'gc>, SlotLoc)> {
        match self.proto.feedback.cache(ic as usize).get() {
            InlineCache::Own { shape, loc, .. } if !shape.is_deprecated() => Some((shape, loc)),
            _ => None,
        }
    }

    /// What the `__index` walk gives for `key` from a table of `class` that
    /// lacks it or holds nil, as far as the region may assume it: through
    /// `__index` tables that are watched holders. Each assumption is a
    /// dependency.
    fn walk(&mut self, class: Option<MtCache<'gc>>, key: LuaString<'gc>) -> Found<'gc> {
        let Some(mut class) = class else {
            return Found::Nil;
        };
        // A class that changes is read live, by the helper.
        if class.is_stale() || class.is_volatile() {
            return Found::Helper;
        }
        for _ in 0..MAX_WALK {
            self.depend_class(class, MetamethodBits::INDEX);
            let index = class.mm(MetamethodBits::INDEX);
            if index.is_nil() {
                return Found::Nil;
            }
            let Some(h) = index.get_table() else {
                return Found::Helper;
            };
            let hs = h.shape();
            if hs.is_dict() {
                return Found::Helper;
            }
            let slot = hs.find_slot(key);
            if !hs.is_watched() || hs.is_volatile_holder() {
                return match slot {
                    Some(s) => Found::Slot(h, hs, true, SlotLoc::new(hs, s)),
                    None => Found::Helper,
                };
            }
            self.deps.push((DepOn::Shape(hs), deps::SHAPE_GROUP));
            if let Some(s) = slot {
                if hs.slot_mutable(s) {
                    return Found::Slot(h, hs, false, SlotLoc::new(hs, s));
                }
                self.deps.push((DepOn::Shape(hs), deps::slot_group(s)));
                let v = h.inner().borrow().named_get(s);
                if !v.is_nil() {
                    return Found::Const(v);
                }
            }
            match hs.mt_cache() {
                None => return Found::Nil,
                Some(c) if c.is_stale() || c.is_volatile() => return Found::Helper,
                Some(c) => class = c,
            }
        }
        Found::Helper
    }

    /// What `found` gives the read of constant `e` from `recv`.
    fn found(&mut self, recv: Val, e: u16, found: Found<'gc>) -> Val {
        match found {
            Found::Nil => self.e.k64(Rep::Val, NIL),
            Found::Const(v) => self.konst(v),
            Found::Slot(h, hs, guard, loc) => {
                let hv = self.konst(Value::table(h));
                let hp = self.e.unboxed(hv, TypeSet::TAB);
                if guard {
                    self.guard_shape(hp, hs);
                }
                let (base, f, k) = self.slot(hp, loc);
                let v = self.load(f, base, k);
                self.or_index(recv, v, e)
            }
            Found::Helper => {
                let key = self.konst(self.proto.constants[e as usize]);
                self.helped(Helper::Index, 0, &[recv, key])
            }
        }
    }

    /// `recv[K[e]]`, by what the site's cache entry holds.
    fn field(&mut self, recv: Val, ic: u16, e: u16) -> Val {
        let key = self.proto.constants[e as usize];
        let Some(key) = key.get_string() else {
            return self.found(recv, e, Found::Helper);
        };
        match self.proto.feedback.cache(ic as usize).get() {
            InlineCache::Own { shape, loc, .. } if !shape.is_deprecated() => {
                let t = self.table(recv);
                self.guard_shape(t, shape);
                let (base, f, k) = self.slot(t, loc);
                let v = self.load(f, base, k);
                // A nil is the answer only where no `__index` may give
                // one; else the helper walks for it.
                match self.walk(shape.mt_cache(), key) {
                    Found::Nil => v,
                    _ => self.or_index(recv, v, e),
                }
            }
            InlineCache::Absent { shape } | InlineCache::ProtoLoad { recv: shape, .. }
                if !shape.is_deprecated() && !shape.is_dict() =>
            {
                let t = self.table(recv);
                self.guard_shape(t, shape);
                let found = self.walk(shape.mt_cache(), key);
                self.found(recv, e, found)
            }
            _ => self.found(recv, e, Found::Helper),
        }
    }

    /// `R[a] = recv[K[e]]`.
    pub(super) fn getfield(&mut self, i: Instruction) {
        let (dst, _, ic, e) = i.abde();
        let recv = self.receiver(i);
        let v = self.field(recv, ic, e);
        self.set(dst, v);
    }

    /// `R[a + 4] = R[b]; R[a] = R[b][K[e]]`.
    pub(super) fn self_(&mut self, i: Instruction) {
        let (dst, obj, ic, e) = i.abde();
        let recv = self.reg(obj);
        let v = self.field(recv, ic, e);
        self.set(dst + 4, recv);
        self.set(dst, v);
    }

    /// `v`, read from an own slot of `recv`, or where it is nil what the
    /// index helper finds for constant `e`, in a cold block.
    fn or_index(&mut self, recv: Val, v: Val, e: u16) -> Val {
        let nil = self.e.is_nil(v);
        if let Some(known) = self.e.known(nil) {
            return if known {
                let key = self.konst(self.proto.constants[e as usize]);
                self.helped(Helper::Index, 0, &[recv, key])
            } else {
                v
            };
        }
        self.state();
        let lp = self.e.g.block(self.e.cur).lp;
        let (walk, on) = (self.e.new_block(), self.e.new_block());
        self.e.br(nil, walk, on);
        self.e.begin(walk, lp, 0, BlockFlags::COLD);
        let key = self.konst(self.proto.constants[e as usize]);
        let found = self.helped(Helper::Index, 0, &[recv, key]);
        self.e.jump(on);
        self.e.begin(on, lp, 0, BlockFlags::empty());
        self.e.param(Rep::Val, None, &[v, found])
    }

    /// `recv.key = v` for a table that lacks the key, where the site's
    /// cache entry is the transition the interpreter took: the value into
    /// the new slot, then the new shape. False for the helper's cases.
    fn transition(&mut self, ic: u16, recv: Val, v: Val) -> bool {
        let InlineCache::Transition { from, to, loc, .. } =
            self.proto.feedback.cache(ic as usize).get()
        else {
            return false;
        };
        let f = to.field(from.slot_count());
        let fits = TypeSet::of_classes(f.kind().classes());
        let held = self.e.fact(v).set;
        let testable = fits == TypeSet::ANY || fits.contains(held) || fits.unboxed() != Rep::None;
        // A metatable's keys tell its class and a watched holder's its
        // dependents; a full spill cell grows; and a class that has or may
        // get `__newindex` is the helper's too.
        let class = from.mt_cache();
        if from.is_deprecated()
            || to.is_deprecated()
            || from.is_stable()
            || from.as_mt().is_some()
            || from.is_watched()
            || loc.spilled()
            || f.kind() == Kind::NONE
            || f.shape().is_some()
            || !testable
            || class.is_some_and(|c| {
                c.is_stale() || c.is_volatile() || c.get().contains(MetamethodBits::NEWINDEX)
            })
        {
            return false;
        }
        if let Some(c) = class {
            self.depend_class(c, MetamethodBits::NEWINDEX);
        }
        let state = self.state();
        let t = self.table(recv);
        self.guard_shape(t, from);
        // Storing nil to an absent key adds nothing.
        let nil = self.e.is_nil(v);
        if self.e.known(nil) == Some(true) {
            return true;
        }
        let lp = self.e.g.block(self.e.cur).lp;
        let on = (self.e.known(nil).is_none()).then(|| {
            let (store, on) = (self.e.new_block(), self.e.new_block());
            self.e.br(nil, on, store);
            self.e.begin(store, lp, 0, BlockFlags::empty());
            on
        });
        if f.kind() != Kind::ANY && !fits.contains(held) {
            self.e.check_type(v, fits, ExitTag::Type, state);
        }
        // The table now refers to `to` as well as to `v`.
        self.e.barrier(t, Val::NONE);
        let (base, field, k) = self.slot(t, loc);
        self.e
            .emit(Op::Store, Rep::None, field as u16, base.0, v.0, k);
        if !self.shapes.iter().any(|s| s.as_word() == to.as_word()) {
            self.shapes.push(to);
        }
        let shape = self.e.k64(Rep::Ptr, to.as_word());
        let field = Field::Shape as u16;
        self.e.emit(Op::Store, Rep::None, field, t.0, shape.0, 0);
        if let Some(on) = on {
            self.e.jump(on);
            self.e.begin(on, lp, 0, BlockFlags::empty());
        }
        true
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
            if self.transition(ic, recv, v) {
                return;
            }
            let key = self.konst(kv);
            self.helped(Helper::NewIndex, 0, &[recv, key, v]);
            let s = self.state();
            self.e.lazy_after(s);
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
        let class = shape.mt_cache();
        if let Some(class) = class.filter(|c| !c.get().contains(MetamethodBits::NEWINDEX)) {
            self.depend_class(class, MetamethodBits::NEWINDEX);
        } else if class.is_some() {
            let old = self.load(f, base, k);
            let s = self.state();
            (self.e).nil_guard(t, old, MetamethodBits::NEWINDEX, ExitTag::Slow, s);
        }
        self.e.barrier(t, v);
        self.e.emit(Op::Store, Rep::None, f as u16, base.0, v.0, k);
    }
}
