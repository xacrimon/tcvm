//! The emitter's rules: each matches an instruction about to be appended
//! and either leaves it (perhaps in canonical form) or gives the value that
//! stands for it. Every rule simplifies; what a rule emits goes through
//! `emit` again and is strictly smaller or lower-level, which bounds the
//! recursion.

use super::emit::Emitter;
use super::ops::{Cc, ExitTag, Flags, Helper, Op};
use super::types::{Rep, TypeSet};
use super::{Inst, Val};
use crate::env::value::{SMALL_INT, Value};
use crate::instruction::ArithKind;
use crate::jit::build::{SPEC_SHIFT, Spec};

const NONE: u32 = u32::MAX;

/// `2^k` for `x`, with `k` at least `min`.
fn log2(x: i64, min: u32) -> Option<i64> {
    (x > 0 && x & (x - 1) == 0 && x.trailing_zeros() >= min).then(|| x.trailing_zeros() as i64)
}

impl Emitter {
    pub(super) fn rule(&mut self, i: &mut Inst) -> Option<Val> {
        use Op::*;
        if i.op.has(Flags::COMMUTES) {
            // A constant goes right; otherwise the older value goes left.
            let (ka, kb) = (self.is_const(Val(i.a)), self.is_const(Val(i.b)));
            if (ka && !kb) || (ka == kb && i.a > i.b) {
                (i.a, i.b) = (i.b, i.a);
            }
        }
        let (a, b, c) = (Val(i.a), Val(i.b), Val(i.c));
        match i.op {
            Add | Sub | Mul | And | Or | Xor | Shl | Shr | Sar | SDiv | UDiv => self.int2(i),
            AddOvf | SubOvf | MulOvf => {
                let bits = if i.rep == Rep::I32 { 32 } else { 64 };
                match (i.op, self.ki(a), self.ki(b)) {
                    (_, Some(x), Some(y)) => {
                        let r = match i.op {
                            AddOvf => x as i128 + y as i128,
                            SubOvf => x as i128 - y as i128,
                            _ => x as i128 * y as i128,
                        };
                        let fits = r >> (bits - 1) == 0 || r >> (bits - 1) == -1;
                        fits.then(|| self.kint(i.rep, r as i64))
                    }
                    (AddOvf | SubOvf, _, Some(0)) | (MulOvf, _, Some(1)) => Some(a),
                    (MulOvf, _, Some(0)) => Some(b),
                    _ => None,
                }
            }
            NegOvf => {
                let x = self.ki(a)?;
                let min = if i.rep == Rep::I32 {
                    i32::MIN as i64
                } else {
                    i64::MIN
                };
                (x != min).then(|| self.kint(i.rep, -x))
            }
            Not | Neg => {
                if let Some(x) = self.ki(a) {
                    let r = if i.op == Not { !x } else { x.wrapping_neg() };
                    return Some(self.kint(i.rep, r));
                }
                let d = self.g.inst(a);
                (d.op == i.op && d.rep == i.rep).then_some(Val(d.a))
            }
            ZExt32 => self.ki(a).map(|x| self.k64(Rep::I64, x as u32 as u64)),
            SExt32 => self.ki(a).map(|x| self.k64(Rep::I64, x as u64)),
            Trunc32 => {
                let d = *self.g.inst(a);
                match d.op {
                    K64 => Some(self.k32(Rep::I32, d.a)),
                    ZExt32 | SExt32 | BoxI32 => Some(Val(d.a)),
                    _ => None,
                }
            }
            Cmp => {
                let cc = Cc::of(i.aux);
                // An unsigned compare of `I32`s sees them zero-extended.
                let ext = |x: i64| {
                    if cc.unsigned() && self.g.rep(a) == Rep::I32 {
                        x as u32 as i64
                    } else {
                        x
                    }
                };
                match (self.ki(a), self.ki(b)) {
                    (Some(x), Some(y)) => Some(self.k32(Rep::B1, cc.eval_i(ext(x), ext(y)) as u32)),
                    (Some(_), None) => {
                        (i.a, i.b, i.aux) = (i.b, i.a, cc.swap() as u16);
                        None
                    }
                    _ if a == b => Some(self.k32(Rep::B1, cc.eval_i(0, 0) as u32)),
                    _ => None,
                }
            }
            Select => match self.known(a) {
                Some(t) => Some(if t { b } else { c }),
                None => (b == c).then_some(b),
            },
            FAdd | FSub | FMul | FDiv => {
                match (self.kf(a), self.kf(b)) {
                    (Some(x), Some(y)) => {
                        let r = match i.op {
                            FAdd => x + y,
                            FSub => x - y,
                            FMul => x * y,
                            _ => x / y,
                        };
                        // A NaN is left for the machine to make.
                        return (!r.is_nan()).then(|| self.kf64(r));
                    }
                    (_, Some(y)) if i.op == FMul && y == 2.0 => {
                        return Some(self.op2(FAdd, Rep::F64, a, a));
                    }
                    // Exact where the reciprocal is a power of two too.
                    (_, Some(y)) if i.op == FDiv => {
                        let r = 1.0 / y;
                        let pow2 = |z: f64| z.is_normal() && z.to_bits() << 12 == 0;
                        if pow2(y) && pow2(r) {
                            let r = self.kf64(r);
                            return Some(self.op2(FMul, Rep::F64, a, r));
                        }
                    }
                    _ => {}
                }
                None
            }
            FNeg | FAbs | FSqrt | FFloor | FCeil => {
                if let Some(x) = self.kf(a) {
                    let r = match i.op {
                        FNeg => -x,
                        FAbs => x.abs(),
                        FSqrt => x.sqrt(),
                        FFloor => x.floor(),
                        _ => x.ceil(),
                    };
                    return (!r.is_nan()).then(|| self.kf64(r));
                }
                let d = *self.g.inst(a);
                match (i.op, d.op) {
                    (FNeg, FNeg) => Some(Val(d.a)),
                    (FAbs, FAbs) => Some(a),
                    (FAbs, FNeg) => Some(self.op1(FAbs, Rep::F64, Val(d.a))),
                    (FFloor | FCeil, IToF | LToF | FFloor | FCeil) => Some(a),
                    _ => None,
                }
            }
            FCmp => {
                let mut cc = Cc::of(i.aux);
                if let (Some(x), Some(y)) = (self.kf(a), self.kf(b)) {
                    return Some(self.k32(Rep::B1, cc.eval_f(x, y)? as u32));
                }
                if self.kf(a).is_some() {
                    cc = cc.swap();
                    (i.a, i.b, i.aux) = (i.b, i.a, cc as u16);
                }
                let (a, b) = (Val(i.a), Val(i.b));
                // Small integers compare as floats as they do as integers.
                let x = *self.g.inst(a);
                if x.op != IToF {
                    return None;
                }
                let y = *self.g.inst(b);
                let y = match self.kf(b) {
                    _ if y.op == IToF => Val(y.a),
                    Some(k) if k == k as i32 as f64 => self.k32(Rep::I32, k as i32 as u32),
                    _ => return None,
                };
                Some(self.cmp(Cmp, cc, Val(x.a), y))
            }
            IToF | LToF => {
                if let Some(x) = self.ki(a) {
                    return Some(self.kf64(x as f64));
                }
                let d = *self.g.inst(a);
                (i.op == LToF && d.op == SExt32).then(|| self.op1(IToF, Rep::F64, Val(d.a)))
            }
            FToI | FToL => {
                let x = self.kf(a)?.trunc();
                let fits = match i.op {
                    FToI => x == x as i32 as f64,
                    _ => x >= -(2f64.powi(63)) && x < 2f64.powi(63),
                };
                fits.then(|| self.kint(i.rep, x as i64))
            }
            BoxI32 => {
                if let Some(x) = self.ki(a) {
                    return Some(self.k64(Rep::Val, SMALL_INT | x as u32 as u64));
                }
                // A box of what a check or a truncation unboxed is the value.
                let d = *self.g.inst(a);
                let v = Val(d.a);
                (d.op == CheckType
                    || (d.op == Trunc32
                        && self.g.rep(v) == Rep::Val
                        && TypeSet::SMALL.contains(self.fact(v).set)))
                .then_some(v)
            }
            BoxF64 => {
                if let Some(x) = self.kf(a) {
                    return Some(self.k64(Rep::Val, Value::float(x).to_raw()));
                }
                let d = self.g.inst(a);
                matches!(d.op, UnboxF64 | CheckType).then_some(Val(d.a))
            }
            BoxPtr => {
                let d = *self.g.inst(a);
                match d.op {
                    KObj => Some(self.emit(KObj, Rep::Val, d.aux, d.a, 0, 0)),
                    UnboxPtr | UnboxPtrPin | CheckType => Some(Val(d.a)),
                    _ => None,
                }
            }
            UnboxF64 => {
                let d = *self.g.inst(a);
                match d.op {
                    BoxF64 => Some(Val(d.a)),
                    K64 => Some(self.kf64(f64::from_bits(d.k64()))),
                    _ => None,
                }
            }
            UnboxPtr | UnboxPtrPin => {
                let d = *self.g.inst(a);
                match d.op {
                    BoxPtr => Some(Val(d.a)),
                    KObj => Some(self.emit(KObj, Rep::Ptr, d.aux, d.a, 0, 0)),
                    _ => None,
                }
            }
            Guard | GuardNot => {
                let passes = self.known(a)? == (i.op == Guard);
                // One that always leaves stays.
                if !passes {
                    return None;
                }
                if i.c == NONE || self.is_const(a) {
                    return Some(c);
                }
                // The result carries the dependency on the guard: only the
                // dominating guard's will do.
                self.number(i).ok()
            }
            Br => {
                let to = match self.known(a) {
                    Some(true) => i.b,
                    Some(false) => i.c,
                    None if i.b == i.c => i.b,
                    None => return None,
                };
                Some(self.emit(Jump, Rep::None, 0, to, 0, 0))
            }
            CheckType => {
                let set = TypeSet::from_bits_retain(i.aux & TypeSet::ANY.bits());
                let f = self.fact(a);
                if f.pending() {
                    return None;
                }
                // Lowered, a fact a check gave is that check again: its
                // pieces number to the first one's, guard and all.
                let by_def = set.contains(self.g.facts[a.idx()].set);
                if set.contains(f.set) && (by_def || !self.lowered) {
                    return Some(self.unboxed(a, set));
                }
                if !self.lowered {
                    return None;
                }
                // Today's aarch64 sequences, as values: a small integer's
                // upper word is all ones, a float is below every box.
                let Some(cond) = self.type_test(a, set) else {
                    self.bail(crate::jit::compile::Bail::Unsupported("type test"));
                    return Some(Val(0));
                };
                let tag = i.aux >> TypeSet::BITS;
                let checked = self.emit(Guard, Rep::Val, tag, cond.0, i.b, a.0);
                // The guard's result is the value, typed by its definition.
                self.g.facts[checked.idx()] = super::types::Fact::of(set);
                self.refine(a, super::types::Fact::of(set));
                Some(match i.rep {
                    Rep::I32 => self.op1(Trunc32, Rep::I32, checked),
                    Rep::F64 => self.op1(UnboxF64, Rep::F64, checked),
                    Rep::Ptr => self.op1(UnboxPtr, Rep::Ptr, checked),
                    _ => Val::NONE,
                })
            }
            IsType => {
                let set = TypeSet::from_bits_retain(i.aux);
                let f = self.fact(a);
                if f.pending() {
                    None
                } else if set.contains(f.set) {
                    Some(self.k32(Rep::B1, 1))
                } else if !set.intersects(f.set) {
                    Some(self.k32(Rep::B1, 0))
                } else if self.lowered {
                    let c = self.type_test(a, set);
                    if c.is_none() {
                        self.bail(crate::jit::compile::Bail::Unsupported("type test"));
                    }
                    Some(c.unwrap_or(Val(0)))
                } else {
                    None
                }
            }
            BoxI64 => {
                if let Some(x) = self.ki(a) {
                    return (x == x as i32 as i64)
                        .then(|| self.k64(Rep::Val, SMALL_INT | x as u32 as u64));
                }
                let d = *self.g.inst(a);
                match d.op {
                    SExt32 => Some(self.op1(BoxI32, Rep::Val, Val(d.a))),
                    UnboxI64 => Some(Val(d.a)),
                    _ => None,
                }
            }
            UnboxI64 => {
                let d = *self.g.inst(a);
                match d.op {
                    BoxI64 => Some(Val(d.a)),
                    BoxI32 => Some(self.op1(SExt32, Rep::I64, Val(d.a))),
                    K64 if d.k64() & SMALL_INT == SMALL_INT => {
                        Some(self.k64(Rep::I64, d.a as i32 as u64))
                    }
                    _ if self.fact(a).set == TypeSet::SMALL => {
                        let x = self.unboxed(a, TypeSet::SMALL);
                        Some(self.op1(SExt32, Rep::I64, x))
                    }
                    _ => None,
                }
            }
            ToF64 => {
                let d = *self.g.inst(a);
                let set = self.fact(a).set;
                match d.op {
                    BoxF64 => Some(Val(d.a)),
                    BoxI32 => Some(self.op1(IToF, Rep::F64, Val(d.a))),
                    K64 if d.k64() & SMALL_INT == SMALL_INT => Some(self.kf64(d.a as i32 as f64)),
                    K64 if set == TypeSet::FLOAT => Some(self.kf64(f64::from_bits(d.k64()))),
                    _ if self.fact(a).pending() => None,
                    _ if set == TypeSet::FLOAT => Some(self.unboxed(a, set)),
                    _ if set == TypeSet::SMALL => {
                        let x = self.unboxed(a, set);
                        Some(self.op1(IToF, Rep::F64, x))
                    }
                    _ => None,
                }
            }
            FToIExact | FToLExact => {
                if let Some(x) = self.kf(a) {
                    let fits = match i.op {
                        FToIExact => x == x as i32 as f64,
                        _ => x == x as i64 as f64 && x < 2f64.powi(63),
                    };
                    return fits.then(|| self.kint(i.rep, x as i64));
                }
                let d = *self.g.inst(a);
                match (d.op, i.op) {
                    (IToF, FToIExact) => Some(Val(d.a)),
                    (IToF, _) => Some(self.op1(SExt32, Rep::I64, Val(d.a))),
                    _ => None,
                }
            }
            LToI => {
                if let Some(x) = self.ki(a) {
                    return (x == x as i32 as i64).then(|| self.k32(Rep::I32, x as u32));
                }
                let d = self.g.inst(a);
                (d.op == SExt32).then_some(Val(d.a))
            }
            DivFloor | ModFloor => match self.floor_const(i) {
                None if self.lowered => Some(self.floor_lower(i)),
                r => r,
            },
            FIDiv => {
                let y = self.kf(b)?;
                let r = 1.0 / y;
                let pow2 = |z: f64| z.is_normal() && z.to_bits() << 12 == 0;
                (pow2(y) && pow2(r)).then(|| {
                    let r = self.kf64(r);
                    let q = self.op2(FMul, Rep::F64, a, r);
                    self.op1(FFloor, Rep::F64, q)
                })
            }
            Arith | Compare | Unm => self.specialise(i),
            K32 | K64 | KF64 | KObj | Base | Load | Store | LoadIdx | StoreIdx | SlotLoad
            | SlotStore | Deopt | LazyExit | FrameState | Param | Jump | Ret | LuaShl | LuaShr
            | AbsOvf | FMod | CallHelper => None,
            Truthy => self.truthy(a),
        }
    }

    /// The unboxed value of a `Val` known here to be in `set`, `Val::NONE`
    /// where the set has no one representation. Typed by its definition, the
    /// unboxed value is good wherever the value is; typed by a check, it is
    /// that check's result; typed by a branch, it is good from here on only.
    pub(crate) fn unboxed(&mut self, v: Val, set: TypeSet) -> Val {
        use Op::*;
        let rep = set.unboxed();
        let by_def = set.contains(self.g.facts[v.idx()].set);
        let check = Inst {
            op: CheckType,
            rep,
            aux: set.bits(),
            a: v.0,
            b: 0,
            c: 0,
        };
        if !by_def && let Ok(r) = self.number(&check) {
            return r;
        }
        match rep {
            Rep::I32 => self.op1(Trunc32, rep, v),
            Rep::F64 => self.op1(UnboxF64, rep, v),
            Rep::Ptr if by_def => self.op1(UnboxPtr, rep, v),
            Rep::Ptr => self.op1(UnboxPtrPin, rep, v),
            _ => Val::NONE,
        }
    }

    /// Whether the `Val` is in `set`, as today's aarch64 sequences written
    /// in values: a small integer's upper word is all ones, a float is below
    /// every box, a heap value's top 16 bits, sign-extended, are its tag
    /// less 8. None for a set that takes more than one compare.
    fn type_test(&mut self, v: Val, set: TypeSet) -> Option<Val> {
        use Op::*;

        use crate::env::value::{
            BOX, FUNCTION_BITS, STRING_BITS, TAG_BOXED_INT, TAG_SHIFT, TAG_TABLE, TAG_THREAD,
            TAG_USERDATA,
        };
        let tag = |bits: u64| bits >> TAG_SHIFT & 7;
        let tag = match set {
            TypeSet::SMALL => {
                let k = self.k64(Rep::I64, 32);
                let hi = self.op2(Shr, Rep::I64, v, k);
                let hi = self.op1(Trunc32, Rep::I32, hi);
                let ones = self.k32(Rep::I32, u32::MAX);
                return Some(self.cmp(Cmp, Cc::Eq, hi, ones));
            }
            TypeSet::FLOAT => {
                let k = self.k64(Rep::I64, BOX);
                return Some(self.cmp(Cmp, Cc::Lo, v, k));
            }
            TypeSet::STR => tag(STRING_BITS),
            TypeSet::FUN => tag(FUNCTION_BITS),
            TypeSet::TAB => TAG_TABLE,
            TypeSet::BIGINT => TAG_BOXED_INT,
            TypeSet::THR => TAG_THREAD,
            TypeSet::UDATA => TAG_USERDATA,
            _ => return None,
        };
        let k = self.k64(Rep::I64, TAG_SHIFT as u64);
        let top = self.op2(Sar, Rep::I64, v, k);
        let want = self.k64(Rep::I64, (tag as i64 - 8) as u64);
        Some(self.cmp(Cmp, Cc::Eq, top, want))
    }

    /// A floor division or modulo by a constant that needs no division.
    fn floor_const(&mut self, i: &Inst) -> Option<Val> {
        use Op::*;
        let (a, b) = (Val(i.a), Val(i.b));
        let (y, div) = (self.ki(b)?, i.op == DivFloor);
        let min = if i.rep == Rep::I32 {
            i32::MIN as i64
        } else {
            i64::MIN
        };
        if let Some(x) = self.ki(a) {
            // A zero divisor raises and `MIN // -1` leaves `I32`.
            if y == 0 || (x == min && y == -1 && i.rep == Rep::I32) {
                return None;
            }
            // Toward negative infinity where the signs differ.
            let (q, r) = (x.wrapping_div(y), x.wrapping_rem(y));
            let down = r != 0 && (r ^ y) < 0;
            let r = if div {
                q - down as i64
            } else {
                r + if down { y } else { 0 }
            };
            return Some(self.kint(i.rep, r));
        }
        match (y, div) {
            (1, true) => Some(a),
            (1 | -1, false) => Some(self.kint(i.rep, 0)),
            (-1, true) if i.rep == Rep::I32 => Some(self.emit(NegOvf, i.rep, i.aux, i.a, i.c, 0)),
            (-1, true) => Some(self.op1(Neg, i.rep, a)),
            _ => {
                let k = log2(y, 1)?;
                Some(if div {
                    let k = self.kint(i.rep, k);
                    self.op2(Sar, i.rep, a, k)
                } else {
                    let m = self.kint(i.rep, y - 1);
                    self.op2(And, i.rep, a, m)
                })
            }
        }
    }

    /// A floor division or modulo as its pieces: the divisor checked, the
    /// machine's division on what the checks passed, the remainder, and the
    /// step toward negative infinity where it is not zero and its sign is
    /// not the divisor's.
    fn floor_lower(&mut self, i: &Inst) -> Val {
        use Op::*;
        let (rep, fs) = (i.rep, i.c);
        let zero = self.kint(rep, 0);
        let nonzero = self.cmp(Cmp, Cc::Ne, Val(i.b), zero);
        let y = self.emit(Guard, rep, i.aux, nonzero.0, fs, i.b);
        // `MIN // -1` leaves `I32`.
        let x = if rep == Rep::I32 && i.op == DivFloor {
            let min = self.kint(rep, i32::MIN as i64);
            let ones = self.kint(rep, -1);
            let (p, q) = (
                self.cmp(Cmp, Cc::Ne, Val(i.a), min),
                self.cmp(Cmp, Cc::Ne, y, ones),
            );
            let fits = self.op2(Or, Rep::B1, p, q);
            self.emit(Guard, rep, ExitTag::Overflow as u16, fits.0, fs, i.a)
        } else {
            Val(i.a)
        };
        let q = self.op2(SDiv, rep, x, y);
        let back = self.op2(Mul, rep, q, y);
        let r = self.op2(Sub, rep, x, back);
        let left = self.cmp(Cmp, Cc::Ne, r, zero);
        let signs = self.op2(Xor, rep, r, y);
        let differ = self.cmp(Cmp, Cc::Lt, signs, zero);
        let down = self.op2(And, Rep::B1, left, differ);
        let (stepped, plain) = if i.op == DivFloor {
            let one = self.kint(rep, 1);
            (self.op2(Sub, rep, q, one), q)
        } else {
            (self.op2(Add, rep, r, y), r)
        };
        self.emit(Select, rep, 0, down.0, stepped.0, plain.0)
    }

    /// A generic op whose operands' facts are final: each operand has a
    /// kind, its fact's or the one the site speculates, which is checked.
    fn specialise(&mut self, i: &Inst) -> Option<Val> {
        use Op::*;
        let (a, b) = (Val(i.a), Val(i.b));
        let unary = i.op == Unm;
        if self.fact(a).pending() || (!unary && self.fact(b).pending()) {
            return None;
        }
        // Copy 1 goes by definitions alone, as the analysis that typed the
        // parameters did.
        let kind = |e: &Self, v: Val, k: usize| {
            let f = if e.lowered {
                e.g.facts[v.idx()]
            } else {
                e.fact(v)
            };
            Spec::by_fact(f).unwrap_or(Spec::of(i.aux >> SPEC_SHIFT[k]))
        };
        let fs = if unary { b } else { Val(i.c) };
        let small = |e: &mut Self, v: Val| e.check_type(v, TypeSet::SMALL, ExitTag::Type, fs);
        let float = |e: &mut Self, v: Val, k: Spec| {
            let x = e.check_type(v, k.set(), ExitTag::Type, fs);
            match k {
                Spec::I32 => e.op1(IToF, Rep::F64, x),
                Spec::F64 => x,
            }
        };
        let ka = kind(self, a, 0);
        if unary {
            let r = match ka {
                Spec::I32 => {
                    let x = small(self, a);
                    let r = self.emit(NegOvf, Rep::I32, ExitTag::Overflow as u16, x.0, fs.0, 0);
                    self.op1(BoxI32, Rep::Val, r)
                }
                Spec::F64 => {
                    let x = float(self, a, ka);
                    let r = self.op1(FNeg, Rep::F64, x);
                    self.op1(BoxF64, Rep::Val, r)
                }
            };
            return Some(r);
        }
        let kb = kind(self, b, 1);
        let low = i.aux & 15;
        if i.op == Compare {
            return Some(if ka == Spec::I32 && kb == Spec::I32 {
                let (x, y) = (small(self, a), small(self, b));
                self.emit(Cmp, Rep::B1, low, x.0, y.0, 0)
            } else {
                let (x, y) = (float(self, a, ka), float(self, b, kb));
                self.emit(FCmp, Rep::B1, low, x.0, y.0, 0)
            });
        }
        let is = |k: ArithKind| low == k as u16;
        if crate::jit::build::int_result(low, ka, kb) {
            let (x, y) = (small(self, a), small(self, b));
            let exits = |e: &mut Self, op: Op, tag: ExitTag| {
                e.emit(op, Rep::I32, tag as u16, x.0, y.0, fs.0)
            };
            let r = if is(ArithKind::Add) {
                exits(self, AddOvf, ExitTag::Overflow)
            } else if is(ArithKind::Sub) {
                exits(self, SubOvf, ExitTag::Overflow)
            } else if is(ArithKind::Mul) {
                exits(self, MulOvf, ExitTag::Overflow)
            } else if is(ArithKind::IDiv) {
                // A zero divisor raises, in the interpreter.
                exits(self, DivFloor, ExitTag::Slow)
            } else if is(ArithKind::Mod) {
                exits(self, ModFloor, ExitTag::Slow)
            } else if is(ArithKind::BAnd) {
                self.op2(And, Rep::I32, x, y)
            } else if is(ArithKind::BOr) {
                self.op2(Or, Rep::I32, x, y)
            } else {
                self.op2(Xor, Rep::I32, x, y)
            };
            return Some(self.op1(BoxI32, Rep::Val, r));
        }
        let (x, y) = (float(self, a, ka), float(self, b, kb));
        let r = if is(ArithKind::Add) {
            self.op2(FAdd, Rep::F64, x, y)
        } else if is(ArithKind::Sub) {
            self.op2(FSub, Rep::F64, x, y)
        } else if is(ArithKind::Mul) {
            self.op2(FMul, Rep::F64, x, y)
        } else if is(ArithKind::Div) {
            self.op2(FDiv, Rep::F64, x, y)
        } else if is(ArithKind::IDiv) {
            let q = self.op2(FDiv, Rep::F64, x, y);
            self.op1(FFloor, Rep::F64, q)
        } else if is(ArithKind::Mod) {
            self.call(Helper::FMod, 0, &[x, y])
        } else if self.kf(y) == Some(2.0) {
            self.op2(FMul, Rep::F64, x, x)
        } else {
            self.call(Helper::Pow, 0, &[x, y])
        };
        Some(self.op1(BoxF64, Rep::Val, r))
    }

    /// Whether the `Val` is neither nil nor false, once its fact is final:
    /// decided by the fact, or lowered a compare.
    fn truthy(&mut self, v: Val) -> Option<Val> {
        let f = self.fact(v);
        if f.pending() {
            None
        } else if !f.set.intersects(TypeSet::FALSY) {
            Some(self.k32(Rep::B1, 1))
        } else if TypeSet::FALSY.contains(f.set) {
            Some(self.k32(Rep::B1, 0))
        } else if self.lowered {
            // Nil and false are the two words from nil on.
            let nil = self.k64(Rep::I64, crate::env::value::NIL);
            let from = self.op2(Op::Sub, Rep::I64, v, nil);
            let two = self.k64(Rep::I64, 2);
            Some(self.cmp(Op::Cmp, Cc::Hs, from, two))
        } else {
            None
        }
    }

    fn is_const(&self, v: Val) -> bool {
        matches!(self.g.op(v), Op::K32 | Op::K64 | Op::KF64 | Op::KObj)
    }

    /// Wrapping integer arithmetic with the machine's shifts.
    fn int2(&mut self, i: &mut Inst) -> Option<Val> {
        use Op::*;
        let (a, b, rep) = (Val(i.a), Val(i.b), i.rep);
        let bits: i64 = match rep {
            Rep::I32 => 32,
            Rep::B1 => 1,
            _ => 64,
        };
        // The value as the width holds it, sign-extended.
        let wrap = |x: i64| match bits {
            32 => x as i32 as i64,
            1 => x & 1,
            _ => x,
        };
        let ones = if bits == 1 { 1 } else { -1 };
        let (ka, kb) = (self.ki(a), self.ki(b));
        if let (Some(x), Some(y)) = (ka, kb) {
            let n = (y & (bits - 1)) as u32;
            let unsigned = |x: i64| {
                if bits == 32 {
                    x as u32 as u64
                } else {
                    x as u64
                }
            };
            let r = match i.op {
                Add => x.wrapping_add(y),
                Sub => x.wrapping_sub(y),
                Mul => x.wrapping_mul(y),
                And => x & y,
                Or => x | y,
                Xor => x ^ y,
                Shl => x << n,
                Shr => (unsigned(x) >> n) as i64,
                Sar => x >> n,
                // Their guards have not run: what would trap is left.
                SDiv if y != 0 && (y != -1 || x != wrap(i64::MIN >> (64 - bits))) => x / y,
                UDiv if y != 0 => (unsigned(x) / unsigned(y)) as i64,
                _ => return None,
            };
            return Some(self.kint(rep, wrap(r)));
        }
        if a == b {
            match i.op {
                And | Or => return Some(a),
                Sub | Xor => return Some(self.kint(rep, 0)),
                _ => {}
            }
        }
        let (da, db) = (*self.g.inst(a), *self.g.inst(b));
        match i.op {
            Add if db.op == Neg => return Some(self.op2(Sub, rep, a, Val(db.a))),
            Add if da.op == Neg => return Some(self.op2(Sub, rep, b, Val(da.a))),
            Sub if db.op == Neg => return Some(self.op2(Add, rep, a, Val(db.a))),
            _ => {}
        }
        let k = kb?;
        match (i.op, k) {
            (Add | Sub | Or | Xor, 0) | (Mul | SDiv | UDiv, 1) => return Some(a),
            (Shl | Shr | Sar, _) if k & (bits - 1) == 0 => return Some(a),
            (Mul | And, 0) => return Some(b),
            (And | Or, _) if k == ones => return Some(if i.op == And { a } else { b }),
            (Xor, -1) if bits != 1 => return Some(self.op1(Not, rep, a)),
            (Mul, -1) => return Some(self.op1(Neg, rep, a)),
            (Mul, 2) => return Some(self.op2(Add, rep, a, a)),
            (Sub, _) => {
                let k = self.kint(rep, wrap(k.wrapping_neg()));
                return Some(self.op2(Add, rep, a, k));
            }
            _ => {}
        }
        if let (Mul | UDiv, Some(n)) = (i.op, log2(k, 2).filter(|_| bits == 64 || i.op == UDiv)) {
            let n = self.kint(rep, n);
            return Some(self.op2(if i.op == Mul { Shl } else { Shr }, rep, a, n));
        }
        // `(z op c1) op c2` is `z op (c1 . c2)`.
        if da.op == i.op
            && da.rep == rep
            && matches!(i.op, Add | And | Or | Xor | Sar)
            && let Some(c1) = self.ki(Val(da.b))
        {
            let c = match i.op {
                Add => c1.wrapping_add(k),
                And => c1 & k,
                Or => c1 | k,
                Xor => c1 ^ k,
                // Both counts are below the width here.
                _ => ((c1 & (bits - 1)) + (k & (bits - 1))).min(bits - 1),
            };
            let c = self.kint(rep, wrap(c));
            return Some(self.op2(i.op, rep, Val(da.a), c));
        }
        None
    }
}
