//! The unit ops' lowerings: the block-free ones the rules call once the
//! graph is low-level, and the rewrites, which make arms and joins
//! (`Emitter::lower`). An arm reads its value through what its branch
//! proved.

use super::emit::Emitter;
use super::fields::Field;
use super::ops::{Cc, Helper, Op};
use super::types::{Rep, TypeSet};
use super::{Inst, Val};
use crate::jit::bug;

impl Emitter {
    /// The store's barrier on table `t`, unless the value is never a heap
    /// object: an op where a repeat is one, lowered at once where the graph
    /// is low-level.
    pub(crate) fn barrier(&mut self, t: Val, v: Val) {
        if !self.fact(v).set.intersects(TypeSet::HEAP) {
            return;
        }
        if self.lowered {
            self.lower(Op::Barrier, Rep::None, 0, t.0, 0, 0);
        } else {
            self.emit(Op::Barrier, Rep::None, 0, t.0, 0, 0);
        }
    }

    /// `FToIExact`/`FToLExact`: the conversion, and a guard that it gives
    /// the float back. A float out of range converts to something
    /// unspecified, of which only a saturated 2^63 converts back equal.
    pub(super) fn exact_lower(&mut self, i: &Inst) -> Val {
        use Op::*;
        let (x, fs) = (Val(i.a), i.b);
        let wide = i.op == FToLExact;
        let (to, back) = if wide { (FToL, LToF) } else { (FToI, IToF) };
        let n = self.op1(to, i.rep, x);
        let f = self.op1(back, Rep::F64, n);
        let mut same = self.cmp(FCmp, Cc::Eq, f, x);
        if wide {
            let top = self.kf64(2f64.powi(63));
            let below = self.cmp(FCmp, Cc::Lt, x, top);
            same = self.op2(And, Rep::B1, same, below);
        }
        self.emit(Guard, i.rep, i.aux, same.0, fs, n.0)
    }

    /// The 64-bit `x` as an `I32`, leaving with `tag` to `fs` where it does
    /// not fit.
    pub(super) fn narrow(&mut self, x: Val, tag: u16, fs: u32) -> Val {
        use Op::*;
        let t = self.op1(Trunc32, Rep::I32, x);
        let w = self.op1(SExt32, Rep::I64, t);
        let fits = self.cmp(Cmp, Cc::Eq, w, x);
        self.emit(Guard, Rep::I32, tag, fits.0, fs, t.0)
    }

    pub(super) fn abs_lower(&mut self, i: &Inst) -> Val {
        use Op::*;
        let (rep, x) = (i.rep, Val(i.a));
        let min = match rep {
            Rep::I32 => i32::MIN as i64,
            _ => i64::MIN,
        };
        let min = self.kint(rep, min);
        let has = self.cmp(Cmp, Cc::Ne, x, min);
        let x = self.emit(Guard, rep, i.aux, has.0, i.b, x.0);
        let zero = self.kint(rep, 0);
        let below = self.cmp(Cmp, Cc::Lt, x, zero);
        let neg = self.op1(Neg, rep, x);
        self.emit(Select, rep, 0, below.0, neg.0, x.0)
    }

    /// Lua's shifts: a count of 64 or more either way gives 0, a negative
    /// one shifts the other way, and a right shift is logical. The `I32`
    /// form shifts the operands as 64-bit integers and leaves where the
    /// result is no `I32`.
    pub(super) fn shift_lower(&mut self, i: &Inst) -> Val {
        use Op::*;
        let left = i.op == LuaShl;
        let (mut x, mut n) = (Val(i.a), Val(i.b));
        if i.rep == Rep::I32 {
            (x, n) = (self.op1(SExt32, Rep::I64, x), self.op1(SExt32, Rep::I64, n));
        }
        let neg = self.op1(Neg, Rep::I64, n);
        let (by_l, by_r) = if left { (n, neg) } else { (neg, n) };
        let l = self.op2(Shl, Rep::I64, x, by_l);
        let r = self.op2(Shr, Rep::I64, x, by_r);
        let (own, other) = if left { (l, r) } else { (r, l) };
        let width = self.k64(Rep::I64, 64);
        let zero = self.k64(Rep::I64, 0);
        let back = self.cmp(Cmp, Cc::Lo, neg, width);
        let rest = self.emit(Select, Rep::I64, 0, back.0, other.0, zero.0);
        let on = self.cmp(Cmp, Cc::Lo, n, width);
        let out = self.emit(Select, Rep::I64, 0, on.0, own.0, rest.0);
        if i.rep == Rep::I32 {
            self.narrow(out, i.aux, i.c)
        } else {
            out
        }
    }

    /// The collector's check: its counter below its threshold, or leave.
    pub(super) fn gc_check_lower(&mut self, i: &Inst) -> Val {
        let metrics = self.k64(Rep::Ptr, self.metrics);
        let used = self.load(Field::GcUsed, metrics, 0);
        let limit = self.load(Field::GcLimit, metrics, 0);
        let below = self.cmp(Op::Cmp, Cc::Lo, used, limit);
        self.emit(Op::Guard, Rep::None, i.aux, below.0, i.a, u32::MAX)
    }

    /// A boxed integer's payload, the value typed by its definition.
    pub(super) fn boxed_int(&mut self, v: Val) -> Val {
        let p = self.op1(Op::UnboxPtr, Rep::Ptr, v);
        self.load(Field::BoxedInt, p, 0)
    }

    // --- rewrites -----------------------------------------------------------

    /// `Barrier`: the helper, in a cold arm, where the table is not gray.
    pub(super) fn barrier_rewrite(&mut self, i: &Inst) -> Val {
        let t = Val(i.a);
        let header = self.load(Field::GcHeader, t, 0);
        let one = self.k64(Rep::I64, 1);
        let gray = self.op2(Op::And, Rep::I64, header, one);
        let zero = self.k64(Rep::I64, 0);
        let due = self.cmp(Op::Cmp, Cc::Eq, gray, zero);
        self.unless(due, |e| {
            e.call(Helper::Barrier, 0, &[t]);
        });
        Val::NONE
    }

    /// `BoxI64`: a small integer where it fits 32 bits, else the allocating
    /// helper in a cold arm.
    pub(super) fn box_rewrite(&mut self, i: &Inst) -> Val {
        use Op::*;
        let x = Val(i.a);
        let t = self.op1(Trunc32, Rep::I32, x);
        let w = self.op1(SExt32, Rep::I64, t);
        let small = self.op1(BoxI32, Rep::Val, t);
        let wide = self.cmp(Cmp, Cc::Ne, w, x);
        let mut on = self.new_join::<1>();
        let cold = self.fork_or_leave(wide, true, &mut on, [small]);
        if self.arm(cold) {
            let boxed = self.call(Helper::BoxInt, 0, &[x]);
            self.leave(&mut on, [boxed]);
        }
        self.join(on).map_or(Val::NONE, |[v]| v)
    }

    /// `UnboxI64` where more than one kind is possible: an arm for a small
    /// and one for a boxed integer, the last untested where the value is
    /// known an integer; with a frame state, anything else leaves in a last
    /// arm.
    pub(super) fn unbox_rewrite(&mut self, i: &Inst) -> Val {
        use Op::*;
        let (v, fs) = (Val(i.a), i.b);
        let has = self.fact(v).set;
        let outside = !TypeSet::INT.contains(has);
        if outside && fs == u32::MAX {
            bug!("{v:?} unboxed as an integer is not known to be one");
        }
        let kinds = [TypeSet::SMALL, TypeSet::BIGINT].map(|k| has.intersects(k).then_some(k));
        let n = kinds.iter().flatten().count();
        let mut on = self.new_join::<1>();
        // The test whose false arm is open.
        let mut failed = None;
        for (at, k) in kinds.into_iter().flatten().enumerate() {
            let last = at + 1 == n;
            let p = match failed {
                Some(c) if last && !outside => self.proved(c, false, v, k),
                None if last && !outside => v,
                _ => {
                    let c = self.emit(IsType, Rep::B1, k.bits(), v.0, 0, 0);
                    let (yes, no) = self.fork(c, false, false);
                    if self.arm(yes) {
                        let p = self.proved(c, true, v, k);
                        let x = self.as_i64(p, k);
                        self.leave(&mut on, [x]);
                    }
                    if !self.arm(no) {
                        break;
                    }
                    failed = Some(c);
                    continue;
                }
            };
            let x = self.as_i64(p, k);
            self.leave(&mut on, [x]);
        }
        if !self.cur.is_none() {
            self.emit(Deopt, Rep::None, i.aux, fs, 0, 0);
        }
        self.join(on).map_or(Val::NONE, |[v]| v)
    }

    /// The `I64` of `v`, typed by its definition as integer kind `k`.
    fn as_i64(&mut self, v: Val, k: TypeSet) -> Val {
        if k == TypeSet::SMALL {
            let x = self.op1(Op::Trunc32, Rep::I32, v);
            self.op1(Op::SExt32, Rep::I64, x)
        } else {
            self.boxed_int(v)
        }
    }

    /// `ToF64` where more than one kind is possible: an arm per kind the
    /// value's fact and the op's set leave, float first, the last untested
    /// where nothing else is possible, else leaving in a last arm.
    pub(super) fn float_rewrite(&mut self, i: &Inst) -> Val {
        use Op::*;
        let (v, fs) = (Val(i.a), i.b);
        let want = TypeSet::from_bits_retain(i.aux & TypeSet::ANY.bits());
        let tag = i.aux >> TypeSet::BITS;
        let has = self.fact(v).set;
        let kinds = [TypeSet::FLOAT, TypeSet::SMALL, TypeSet::BIGINT];
        let kinds = kinds.map(|k| (has.intersects(k) && want.contains(k)).then_some(k));
        let n = kinds.iter().flatten().count();
        let mut covered = TypeSet::empty();
        kinds.iter().flatten().for_each(|&k| covered |= k);
        let outside = !covered.contains(has);
        let mut on = self.new_join::<1>();
        // The test whose false arm is open.
        let mut failed = None;
        for (at, k) in kinds.into_iter().flatten().enumerate() {
            let last = at + 1 == n;
            let p = match failed {
                Some(c) if last && !outside => self.proved(c, false, v, k),
                None if last && !outside => v,
                _ => {
                    let c = self.emit(IsType, Rep::B1, k.bits(), v.0, 0, 0);
                    let (yes, no) = self.fork(c, false, false);
                    if self.arm(yes) {
                        let p = self.proved(c, true, v, k);
                        let x = self.as_float(p, k);
                        self.leave(&mut on, [x]);
                    }
                    if !self.arm(no) {
                        break;
                    }
                    failed = Some(c);
                    continue;
                }
            };
            let x = self.as_float(p, k);
            self.leave(&mut on, [x]);
        }
        if !self.cur.is_none() {
            self.emit(Deopt, Rep::None, tag, fs, 0, 0);
        }
        self.join(on).map_or(Val::NONE, |[v]| v)
    }

    /// The float of `v`, typed by its definition as kind `k`.
    fn as_float(&mut self, v: Val, k: TypeSet) -> Val {
        use Op::*;
        if k == TypeSet::FLOAT {
            self.op1(UnboxF64, Rep::F64, v)
        } else if k == TypeSet::SMALL {
            let x = self.op1(Trunc32, Rep::I32, v);
            self.op1(IToF, Rep::F64, x)
        } else {
            let x = self.boxed_int(v);
            self.op1(LToF, Rep::F64, x)
        }
    }
}
