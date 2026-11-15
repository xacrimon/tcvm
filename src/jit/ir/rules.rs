//! The emitter's rules: each matches an instruction about to be appended
//! and either leaves it (perhaps in canonical form) or gives the value that
//! stands for it. Every rule simplifies; what a rule emits goes through
//! `emit` again and is strictly smaller or lower-level, which bounds the
//! recursion.

use super::emit::Emitter;
use super::ops::{Cc, Flags, Op};
use super::types::{Rep, TypeSet};
use super::{Inst, Val};
use crate::env::value::{SMALL_INT, Value};

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
                if f.pending() || !set.contains(f.set) {
                    return None;
                }
                Some(self.unboxed(a, set))
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
            DivFloor | ModFloor => {
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
                    (-1, true) if i.rep == Rep::I32 => {
                        Some(self.emit(NegOvf, i.rep, i.aux, i.a, i.c, 0))
                    }
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
            K32 | K64 | KF64 | KObj | Load | Store | LoadIdx | StoreIdx | SlotLoad | SlotStore
            | Deopt | LazyExit | FrameState | Param | Jump | Ret | LuaShl | LuaShr | AbsOvf
            | FMod | Arith | Unm | Compare | Truthy => None,
        }
    }

    /// The unboxed value of a `Val` known here to be in `set`, `Val::NONE`
    /// where the set has no one representation. Typed by its definition, the
    /// unboxed value is good wherever the value is; typed by a check, it is
    /// that check's result; typed by a branch, it is good from here on only.
    fn unboxed(&mut self, v: Val, set: TypeSet) -> Val {
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
