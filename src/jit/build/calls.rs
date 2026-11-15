//! Calls. The callee and its arguments go to the callee's slots, the call
//! ends the block, and its results are the resume block's.

use super::Builder;
use super::cfg::RegSet;
use crate::dmm::Gc;
use crate::env::function::{Function, Prototype};
use crate::env::value::Value;
use crate::instruction::UpvalSource;
use crate::instruction::{Instruction, Op as Bc};
use crate::jit::ir::fields::Field;
use crate::jit::ir::ops::{Cc, ExitKind, ExitTag, Helper, Op};
use crate::jit::ir::table::Key;
use crate::jit::ir::types::{Rep, TypeSet};
use crate::jit::ir::{BlockFlags, HOME, Val};

impl<'gc> Builder<'_, 'gc> {
    /// `R[a] = closure(P[d])`: the helper reads what it captures from the
    /// frame's slots, and marks the frame as having open upvalues.
    pub(super) fn closure_new(&mut self, pc: u32, i: Instruction) {
        let (a, proto) = (i.a(), i.d());
        let child = self.proto.prototypes[proto as usize];
        for d in child.upvalue_desc.iter() {
            match d.source {
                UpvalSource::ParentLocal(r) if !(d.by_value && r == a) => {
                    let v = self.reg(r);
                    self.e.emit(Op::SlotStore, Rep::None, r as u16, v.0, 0, 0);
                }
                _ => {}
            }
        }
        let f = (self.e).call(Helper::Closure, proto as u32 | (a as u32) << 16, &[]);
        self.set(a, f);
        self.gc_check(pc);
    }

    /// `CLOSE`: the cells of the registers from `R[a]` up. A `__close` runs
    /// Lua: the interpreter does it.
    pub(super) fn close(&mut self, i: Instruction) {
        let tbc = self.frame_flags(crate::vm::frame::flag::HAS_TBC);
        let zero = self.e.k64(Rep::I64, 0);
        let none = self.e.cmp(Op::Cmp, Cc::Eq, tbc, zero);
        let state = self.state();
        let tag = ExitTag::Unsupported as u16;
        self.e
            .emit(Op::Guard, Rep::None, tag, none.0, state.0, u32::MAX);
        self.e.call(Helper::Close, i.a() as u32, &[]);
    }

    /// `R[a], ... = R[a](R[a + 4], ...)`; a fused form's callee is `R[e]`.
    pub(super) fn call(&mut self, pc: u32, i: Instruction) {
        let (a, nb, c) = i.abc();
        let fused = matches!(i.op(), Bc::CALLS | Bc::CALLS_R0 | Bc::CALLS_R1);
        let callee = self.reg(if fused { i.e() as u8 } else { a });
        let (callee, known) = self.known_callee(pc, callee);
        self.e
            .emit(Op::SlotStore, Rep::None, a as u16, callee.0, 0, 0);
        // Arguments up to `top` (`nb == 0`): those before the ones a call
        // landed there, which have no value here.
        let end = match nb {
            0 => self.cfg.nregs as u16,
            _ => a as u16 + 3 + nb as u16,
        };
        // A known callee takes its first parameters in registers.
        let known = known.filter(|_| nb != 0);
        let in_regs = known.map_or(0, |p| (nb - 1).min(p.num_params).min(8));
        if let Some(p) = known {
            let at = self.e.g.extra.len() as u32;
            for k in 0..in_regs {
                let v = self.reg(a + 4 + k);
                self.e.g.extra.push(v.0);
            }
            let k = self.pool.intern(p.ptr.as_ptr() as u64);
            (self.e).emit(Op::CallArgs, Rep::None, 0, at, in_regs as u32, k);
            self.protos.push(p);
        }
        for r in a as u16 + 4 + in_regs as u16..end {
            let v = self.reg(r as u8);
            if !v.is_none() {
                self.e.emit(Op::SlotStore, Rep::None, r, v.0, 0, 0);
            }
        }
        let imm = a as u32 | (nb.wrapping_sub(1) as u32) << 8 | (c as u32) << 16 | (a as u32) << 24;
        let first = self.enter(pc + 1, a as usize, self.live(pc + 1), imm);
        // Results up to `top` (`c == 0`) stay in their slots, for the
        // instruction that takes them from there.
        for r in a as usize..self.cfg.nregs {
            self.e.table.set(Key::reg(r as u32), Val::NONE.0);
        }
        if c == 2 {
            self.set(a, first);
        } else if c != 0 {
            for r in a..a + c - 1 {
                let v = self.e.emit(Op::SlotLoad, Rep::Val, r as u16, 0, 0, 0);
                self.set(r, v);
            }
        }
        // The callee may have broken what the region assumes.
        self.lazy_exit(pc + 1);
    }

    /// `callee` as the call at `pc` may take it, and its prototype where it
    /// is a Lua closure without varargs: a constant; else the one function
    /// the site called, guarded by identity, or by its prototype where that
    /// made several closures or the site's word names it. Nothing once such
    /// a guard failed here.
    fn known_callee(&mut self, pc: u32, callee: Val) -> (Val, Option<Gc<'gc, Prototype<'gc>>>) {
        use crate::env::feedback::slot_proto;
        let fixed = |f: Function<'gc>| f.as_lua().filter(|c| !c.proto.is_vararg).map(|c| c.proto);
        let d = *self.e.g.inst(callee);
        if d.op == Op::KObj {
            // SAFETY: a pool word is a value's.
            let v: Value<'gc> = unsafe { Value::from_raw(self.pool.words[d.a as usize]) };
            return (callee, v.get_function().and_then(fixed));
        }
        let feedback = &self.proto.feedback;
        let seen = feedback.callee(pc as usize).map_or(0, |w| w.get());
        if seen == 0 || feedback.bits(pc as usize) & crate::jit::feedback::CONST_MISS != 0 {
            return (callee, None);
        }
        let s = self.state();
        // Bit 0: megamorphic, or a prototype's.
        // SAFETY: else the word is the value of the function the site called.
        let f = (seen & 1 == 0).then(|| unsafe { Value::from_raw(seen) }.get_function());
        let proto = match f {
            None => slot_proto(seen).filter(|p| !p.is_vararg),
            Some(Some(f)) => match f.as_lua() {
                // The call tests such a callee anyway.
                Some(c) if c.proto.is_vararg => return (callee, None),
                Some(c) if c.proto.closures.get() > 1 => Some(c.proto),
                _ => {
                    let k = self.konst(Value::function(f));
                    let same = self.e.cmp(Op::Cmp, Cc::Eq, callee, k);
                    let tag = ExitTag::Const as u16;
                    (self.e).emit(Op::Guard, Rep::None, tag, same.0, s.0, u32::MAX);
                    return (k, fixed(f));
                }
            },
            Some(None) => None,
        };
        let Some(p) = proto else {
            return (callee, None);
        };
        let f = self.e.check_type(callee, TypeSet::FUN, ExitTag::Const, s);
        let kind = self.e.load(Field::FnKind, f, 0);
        let zero = self.e.k32(Rep::I32, 0);
        let fixed = self.e.cmp(Op::Cmp, Cc::Eq, kind, zero);
        self.guard(fixed, ExitTag::Const);
        let has = self.e.load(Field::FnProto, f, 0);
        let want = self.e.k64(Rep::Ptr, p.ptr.as_ptr() as u64);
        let same = self.e.cmp(Op::Cmp, Cc::Eq, has, want);
        self.guard(same, ExitTag::Const);
        (callee, Some(p))
    }

    /// The call whose callee and arguments are in their slots: the block
    /// ends, and the resume block's first result is returned (a value for
    /// one result wanted, `c == 2`). The instruction at `pc` goes on after
    /// it; of the registers below `below`, which the callee's frame leaves
    /// alone, those in `kept` are read after it: kept in the native frame,
    /// or in their slot where that holds them.
    fn enter(&mut self, pc: u32, below: usize, mut kept: RegSet, imm: u32) -> Val {
        // A captured register's slot is its cell's, read or not.
        let captured = self.cfg.captured;
        for r in captured.iter().filter(|&r| r < below) {
            if self.reg(r as u8).is_none() {
                let v = self.e.emit(Op::SlotLoad, Rep::Val, r as u16, 0, 0, 0);
                self.e.table.set(Key::reg(r as u32), v.0);
            }
        }
        kept.union(&captured);
        let entries: Vec<(u32, Val)> = (kept.iter())
            .filter(|&r| r < below && !self.reg(r as u8).is_none())
            .map(|r| {
                let v = self.reg(r as u8);
                let home = captured.contains(r) || self.e.slot_holds(r as u16, v);
                (r as u32 | if home { HOME } else { 0 }, v)
            })
            .collect();
        let state = (self.e).frame_state(pc, ExitKind::Before, Val::NONE, &entries);
        let lp = self.e.g.block(self.e.cur).lp;
        let resume = self.e.new_block();
        self.e
            .emit(Op::CallEntry, Rep::None, 0, state.0, resume.0, imm);
        self.e.begin(resume, lp, 0, BlockFlags::RESUME);
        // One result comes in a register; more land in their slots.
        let rep = if imm >> 16 & 0xff == 2 {
            Rep::Val
        } else {
            Rep::None
        };
        let first = self.e.emit(Op::Resume, rep, 0, imm, 0, 0);
        // The callee may have written the captured registers.
        for r in captured.iter().filter(|&r| r < below) {
            let v = self.e.emit(Op::SlotLoad, Rep::Val, r as u16, 0, 0, 0);
            self.e.table.set(Key::reg(r as u32), v.0);
        }
        self.state = Val::NONE;
        first
    }

    /// `f(args...)` from a header past the frame's registers, as the
    /// interpreter stages a metamethod's call (`call_mm!`), the instruction
    /// at `pc + 1` next: `c - 1` results, the one of `c == 2` returned, more
    /// landed from `R[land]`. No register is written but those.
    pub(super) fn call_mm(&mut self, pc: u32, f: Val, args: &[Val], land: u8, c: u8) -> Val {
        let hdr = self.proto.max_stack_size as u16;
        self.window = self.window.max(hdr + 4 + args.len() as u16);
        self.e.emit(Op::SlotStore, Rep::None, hdr, f.0, 0, 0);
        for (k, v) in args.iter().enumerate() {
            let at = hdr + 4 + k as u16;
            self.e.emit(Op::SlotStore, Rep::None, at, v.0, 0, 0);
        }
        // An instruction that fails after its metamethod ran reads its
        // operands again.
        let mut kept = self.live(pc);
        kept.union(&self.live(pc + 1));
        let imm = land as u32 | (args.len() as u32) << 8 | (c as u32) << 16 | (hdr as u32) << 24;
        let first = self.enter(pc + 1, self.cfg.nregs, kept, imm);
        if c > 2 {
            for r in land..land + c - 1 {
                let v = self.e.emit(Op::SlotLoad, Rep::Val, r as u16, 0, 0, 0);
                self.set(r, v);
            }
        }
        first
    }

    /// `ARITH_MM`, `_MM_R` and `_MMI`: the metamethod the table operand's
    /// class has, called. Operands of other kinds, or a class without it,
    /// leave as a type guard would.
    pub(super) fn arith_mm(&mut self, pc: u32, i: Instruction) {
        let orig = i.mm_orig();
        let bit = crate::vm::ops::arith::kind_mm(orig.info().kind);
        let (l, r, number_left) = match i.op() {
            Bc::ARITH_MMI => {
                let v = self.reg(i.b());
                let k = self.konst(if i.imm_is_int() {
                    Value::small(i.imm_int() as i32)
                } else {
                    Value::float(i.imm_float())
                });
                if i.c() & 1 != 0 || orig.info().reversed {
                    (k, v, true)
                } else {
                    (v, k, false)
                }
            }
            Bc::ARITH_MM_R => (self.reg(i.b()), self.reg(i.c()), true),
            _ => (self.reg(i.b()), self.reg(i.c()), false),
        };
        let zero = self.e.k64(Rep::I64, 0);
        let t = if number_left {
            let state = self.state();
            (self.e).check_type(l, TypeSet::NUM, ExitTag::Type, state);
            // The numbers' own metamethod would come first.
            let cell = self.e.k64(Rep::Ptr, self.links.number_mt);
            let mt = self.e.load(Field::TypeMt, cell, 0);
            let none = self.e.cmp(Op::Cmp, Cc::Eq, mt, zero);
            self.guard(none, ExitTag::Slow);
            self.table(r)
        } else {
            self.table(l)
        };
        let shape = self.e.load(Field::Shape, t, 0);
        let class = self.e.load(Field::ShapeClass, shape, 0);
        let classed = self.e.cmp(Op::Cmp, Cc::Ne, class, zero);
        self.guard(classed, ExitTag::Type);
        let rest = self.e.load(Field::ClassRest, class, 0);
        let any = self.e.cmp(Op::Cmp, Cc::Ne, rest, zero);
        self.guard(any, ExitTag::Type);
        let f = (self.e).load(Field::RestMm, rest, bit.bits().trailing_zeros());
        let nil = self.e.k64(Rep::Val, crate::env::value::NIL);
        let has = self.e.cmp(Op::Cmp, Cc::Ne, f, nil);
        self.guard(has, ExitTag::Type);
        let v = self.call_mm(pc, f, &[l, r], 0, 2);
        self.set(i.a(), v);
        self.lazy_exit(pc + 1);
    }
}
