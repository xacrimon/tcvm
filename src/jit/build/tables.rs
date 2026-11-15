//! Table, upvalue and global access. A constant-key site is a generic
//! op naming the record made here of the shapes its feedback saw; an
//! indexed one a generic op the key's kind decides. Upvalues are loads.

use super::Builder;
use crate::env::function::InlineCache;
use crate::env::shape::Kind;
use crate::env::table::SlotLoc;
use crate::env::{LuaString, MetamethodBits, Shape};
use crate::instruction::{Instruction, Op as Bc};
use crate::jit::compile::Bail;
use crate::jit::feedback as fb;
use crate::jit::feedback::MAX_SHAPES;
use crate::jit::ir::access::ARRAY;
use crate::jit::ir::fields::Field;
use crate::jit::ir::ops::{Cc, ExitTag, Helper, Op};
use crate::jit::ir::sites::{Case, FieldSite, Form};
use crate::jit::ir::types::{Rep, TypeSet};
use crate::jit::ir::{BlockFlags, Val};

impl<'gc> Builder<'_, 'gc> {
    fn load(&mut self, f: Field, base: Val, k: u32) -> Val {
        self.e.load(f, base, k)
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
        let t = self.e.op1(Op::NewTable, Rep::Val, template);
        self.set(i.a(), t);
        let after = self.state_at(pc + 1);
        let tag = ExitTag::Gc as u16;
        self.e.emit(Op::GcCheck, Rep::None, tag, after.0, 0, 0);
    }

    /// The record of the constant-key site being emitted, of key `K[e]` and
    /// cache word `ic`: a case per shape its cache entry and its shape list
    /// name, each found by looking the key up in the shape. None where the
    /// compile has more sites than an op can name.
    fn field_site(&mut self, ic: u16, e: u16, get: bool) -> Option<u16> {
        let kv = self.proto.constants[e as usize];
        let Some(key) = kv.get_string() else {
            crate::jit::bug!("a field site's key is no string");
        };
        let feedback = &self.proto.feedback;
        let cached = match feedback.cache(ic as usize).get() {
            InlineCache::Own { shape, .. } | InlineCache::Absent { shape } => Some(shape),
            InlineCache::Transition { from, .. } => Some(from),
            InlineCache::ProtoLoad { recv, .. } => Some(recv),
            InlineCache::Empty => None,
        };
        let mut shapes: Vec<Shape<'gc>> = Vec::new();
        for s in cached.into_iter().chain(feedback.shapes(self.pc as usize)) {
            if !s.is_deprecated() && !shapes.iter().any(|&x| Shape::ptr_eq(x, s)) {
                shapes.push(s);
            }
        }
        let none = Case {
            shape: 0,
            form: Form::Helper,
            classed: false,
            loc: SlotLoc::from_raw(0),
            kind: Kind::ANY,
            expect: 0,
            to: 0,
        };
        let mut site = FieldSite {
            key: self.pool.intern(kv.to_raw()),
            n: 0,
            cases: [none; MAX_SHAPES],
        };
        // More shapes than a site takes: megamorphic, the helper's.
        if shapes.len() <= MAX_SHAPES {
            for (k, &s) in shapes.iter().enumerate() {
                site.cases[k] = self.case(s, key, get, none);
                self.shapes.push(s);
            }
            site.n = shapes.len() as u8;
        }
        let at = self.e.sites.field(site);
        if at.is_none() {
            self.e.bail(Bail::TooLarge);
        }
        at
    }

    /// What an access of `key` does for tables of shape `s`.
    fn case(&mut self, s: Shape<'gc>, key: LuaString<'gc>, get: bool, none: Case) -> Case {
        let mut c = Case {
            shape: s.as_word(),
            classed: s.mt_cache().is_some(),
            ..none
        };
        if s.is_dict() {
            return c;
        }
        let slot = s.find_slot(key);
        if get {
            match slot {
                Some(k) => (c.form, c.loc) = (Form::Own, SlotLoc::new(s, k)),
                None => c.form = Form::Absent,
            }
            return c;
        }
        // A metatable's keys tell its class and a watched holder's its
        // dependents: the helper's store does.
        if s.as_mt().is_some() || s.is_watched() {
            return c;
        }
        match slot {
            Some(k) => {
                let f = s.field(k);
                if f.kind() != Kind::NONE {
                    (c.form, c.loc) = (Form::Own, SlotLoc::new(s, k));
                    (c.kind, c.expect) = (f.store_kind(), f.shape().map_or(0, Shape::as_word));
                }
            }
            // A move repeats one a slow path made from the shape, which
            // marked it as left; looked up, never made.
            None if !s.is_stable() => {
                let Some(to) = s.prop_child(self.mc, key) else {
                    return c;
                };
                let k = s.slot_count();
                let (loc, f) = (SlotLoc::new(to, k), to.field(k));
                if !to.is_dict() && !loc.spilled() && f.kind() != Kind::NONE {
                    (c.form, c.loc, c.to) = (Form::Transition, loc, to.as_word());
                    (c.kind, c.expect) = (f.store_kind(), f.shape().map_or(0, Shape::as_word));
                    self.shapes.push(to);
                }
            }
            None => {}
        }
        c
    }

    /// `R[a] = recv[K[e]]`.
    pub(super) fn getfield(&mut self, i: Instruction) {
        let (dst, _, ic, e) = i.abde();
        let recv = self.receiver(i);
        let Some(site) = self.field_site(ic, e, true) else {
            return;
        };
        let s = self.state();
        // Lowered here where the receiver's fact is final, else by copy 1.
        let v = (self.e).lower(Op::GetField, Rep::Val, site, recv.0, s.0, 0);
        self.set(dst, v);
    }

    /// `recv[K[e]] = R[a]`.
    pub(super) fn setfield(&mut self, i: Instruction) {
        let (src, _, ic, e) = i.abde();
        let recv = self.receiver(i);
        let v = self.reg(src);
        let Some(site) = self.field_site(ic, e, false) else {
            return;
        };
        let s = self.state();
        (self.e).lower(Op::SetField, Rep::None, site, recv.0, v.0, s.0);
    }
}
