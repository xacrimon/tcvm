//! `Compare` by kinds. An ordered compare is an integer compare where both
//! operands are integers and a float compare otherwise, a 64-bit integer
//! against a float exactly, and two strings by their bytes in a loop. An
//! equality is the words' compare wherever that decides it, and the pairs
//! it does not decide (numbers of different kinds, boxed integers, tables
//! that may have `__eq`) past it. A site with one leaf is a block-free
//! rule; the rest is a rewrite.

use super::arith::{self, BIG, FLOAT, SMALL, STR, Side, Site};
use super::emit::{Emitter, Join};
use super::fields::Field;
use super::ops::{Cc, ExitTag, Heaps, Op};
use super::types::{Rep, TypeSet};
use super::{Inst, Val};
use crate::env::MetamethodBits;
use crate::instruction::ArithKind;
use crate::jit::bug;

/// The seen kinds past arithmetic's four.
pub(crate) const TAB: u8 = 16;
pub(crate) const OTHER: u8 = 32;

/// `aux`: the condition, the site's seen kinds (one set for both
/// operands), and `arith::MM`.
pub(crate) const SEEN_SHIFT: u32 = 4;

pub(crate) fn seen(aux: u16) -> u8 {
    (aux >> SEEN_SHIFT) as u8 & 63
}

/// How a loop parameter a site compares is best held: as the one number
/// kind the site saw, or an `I64` where it saw only integers, 64-bit ones
/// among them.
pub(crate) fn demand(aux: u16) -> Option<TypeSet> {
    match seen(aux) {
        SMALL => Some(TypeSet::SMALL),
        FLOAT => Some(TypeSet::FLOAT),
        s if s & BIG != 0 && s & !(SMALL | BIG) == 0 => Some(TypeSet::INT),
        _ => None,
    }
}

/// Whether a site with this `aux` takes an integer held as `I64` as it is.
pub(crate) fn takes_i64(aux: u16) -> bool {
    seen(aux) & BIG != 0
}

/// An operand of an equality as its kinds are tested one after another.
struct Chain {
    v: Val,
    /// What it can still be.
    rem: TypeSet,
    /// The last test that failed on the way here.
    failed: Option<Val>,
}

impl Emitter {
    fn compared(&self, v: Val) -> TypeSet {
        match self.g.rep(v) {
            Rep::I64 => TypeSet::INT,
            Rep::Val => self.deciding(v).set,
            r => bug!("a compare of a value held as {r:?}"),
        }
    }

    fn site(&self, i: &Inst, wide: bool) -> Site {
        Site {
            low: ArithKind::Add as u16,
            wide,
            tag: if i.aux & arith::MM != 0 {
                ExitTag::Mm
            } else {
                ExitTag::Type
            },
            fs: Val(i.c),
            unary: false,
            raw: false,
        }
    }

    /// The rule of `Compare`: a site one compare decides.
    pub(super) fn compare_rule(&mut self, i: &Inst) -> Option<Val> {
        let (a, b) = (Val(i.a), Val(i.b));
        if self.fact(a).pending() || self.fact(b).pending() {
            return None;
        }
        match Cc::of(i.aux & 15) {
            Cc::Eq => self.eq_rule(i, a, b),
            cc => self.order_rule(i, cc, a, b),
        }
    }

    /// The rewrite of `Compare`, the operands' facts final.
    pub(super) fn compare_rewrite(&mut self, i: &Inst) -> Val {
        let (a, b) = (Val(i.a), Val(i.b));
        if self.fact(a).pending() || self.fact(b).pending() {
            return self.append(*i);
        }
        match Cc::of(i.aux & 15) {
            Cc::Eq => self.eq_rewrite(i, a, b),
            cc => self.order_rewrite(i, cc, a, b),
        }
    }

    // --- a 64-bit integer against a float ------------------------------------

    /// `i cc f` exactly, for an `I64` and an `F64`: as floats where the
    /// integer converts exactly; past that a float in the integers' range
    /// is compared truncated (one that is not integral is smaller in
    /// magnitude than the integer, so on the same side of it and unequal),
    /// and one outside it, or a NaN, decides by its sign.
    fn int_float(&mut self, cc: Cc, i: Val, f: Val) -> Val {
        use Op::*;
        let bias = self.k64(Rep::I64, 1 << 53);
        let biased = self.op2(Add, Rep::I64, i, bias);
        let span = self.k64(Rep::I64, 1 << 54);
        let fits = self.cmp(Cmp, Cc::Ls, biased, span);
        let mut out = self.new_join::<1>();
        let (yes, no) = self.fork(fits, false, true);
        if self.arm(yes) {
            let x = self.op1(LToF, Rep::F64, i);
            let r = self.cmp(FCmp, cc, x, f);
            self.leave(&mut out, [r]);
        }
        if self.arm(no) {
            let (low, high) = (self.kf64(-(2f64.powi(63))), self.kf64(2f64.powi(63)));
            let (ge, lt) = (
                self.cmp(FCmp, Cc::Ge, f, low),
                self.cmp(FCmp, Cc::Lt, f, high),
            );
            let within = self.op2(And, Rep::B1, ge, lt);
            let (yes, no) = self.fork(within, false, false);
            if self.arm(yes) {
                let n = self.op1(FToL, Rep::I64, f);
                let r = self.cmp(Cmp, cc, i, n);
                self.leave(&mut out, [r]);
            }
            if self.arm(no) {
                let zero = self.kf64(0.0);
                let r = match cc {
                    Cc::Lt | Cc::Le => self.cmp(FCmp, Cc::Gt, f, zero),
                    Cc::Gt | Cc::Ge => self.cmp(FCmp, Cc::Lt, f, zero),
                    _ => self.k32(Rep::B1, 0),
                };
                self.leave(&mut out, [r]);
            }
        }
        self.join(out).map_or(Val::NONE, |[r]| r)
    }

    // --- order ---------------------------------------------------------------

    fn order_sides(&self, i: &Inst, a: Val, b: Val) -> (Side, Side, Site) {
        let (low, seen) = (ArithKind::Add as u16, seen(i.aux) & 15);
        let (a, b) = (self.side(a, seen, low), self.side(b, seen, low));
        let wide = (a.armed | b.armed) & BIG != 0;
        (a, b, self.site(i, wide))
    }

    fn order_rule(&mut self, i: &Inst, cc: Cc, a: Val, b: Val) -> Option<Val> {
        use Op::*;
        let (a, b, s) = self.order_sides(i, a, b);
        if a.armed == 0 || b.armed == 0 {
            self.emit(Deopt, Rep::None, s.tag as u16, s.fs.0, 0, 0);
            return Some(Val::NONE);
        }
        let one = |x: Side| x.armed.is_power_of_two();
        if s.wide || !one(a) || !one(b) || (a.armed | b.armed) & STR != 0 {
            return None;
        }
        Some(if a.armed == SMALL && b.armed == SMALL {
            let x = self.check_type(a.v, TypeSet::SMALL, s.tag, s.fs);
            let y = self.check_type(b.v, TypeSet::SMALL, s.tag, s.fs);
            self.cmp(Cmp, cc, x, y)
        } else {
            let (x, y) = (self.as_f64(a.armed, a.v, s), self.as_f64(b.armed, b.v, s));
            self.cmp(FCmp, cc, x, y)
        })
    }

    fn order_rewrite(&mut self, i: &Inst, cc: Cc, a: Val, b: Val) -> Val {
        use Op::*;
        let (a, b, s) = self.order_sides(i, a, b);
        if a.armed == 0 || b.armed == 0 {
            self.emit(Deopt, Rep::None, s.tag as u16, s.fs.0, 0, 0);
            return Val::NONE;
        }
        // A string against a number is the interpreter's (it raises).
        let numbers = Side {
            armed: b.armed & !STR,
            rest: b.rest || b.armed & STR != 0,
            ..b
        };
        let mut out = self.new_join::<1>();
        let (mut ints, mut floats) = (self.new_join::<2>(), self.new_join::<2>());
        let (mut wides, mut strs) = (self.new_join::<1>(), self.new_join::<2>());
        let mut int_a = self.new_join::<1>();
        self.by_kind(a, s, &mut |e, kind, p| match kind {
            SMALL | BIG => {
                let x = e.as_int(kind, p, s);
                e.leave(&mut int_a, [x]);
            }
            STR if b.armed & STR == 0 => {
                e.emit(Deopt, Rep::None, s.tag as u16, s.fs.0, 0, 0);
            }
            STR => {
                let x = e.check_type(p, TypeSet::STR, s.tag, s.fs);
                let y = e.check_type(b.v, TypeSet::STR, s.tag, s.fs);
                e.leave(&mut strs, [x, y]);
            }
            _ => {
                let x = e.as_f64(kind, p, s);
                e.by_kind(numbers, s, &mut |e, kind, q| match kind {
                    BIG => {
                        let y = e.as_int(kind, q, s);
                        let r = e.int_float(cc.swap(), y, x);
                        e.leave(&mut wides, [r]);
                    }
                    _ => {
                        let y = e.as_f64(kind, q, s);
                        e.leave(&mut floats, [x, y]);
                    }
                });
            }
        });
        if let Some([x]) = self.join(int_a) {
            self.by_kind(numbers, s, &mut |e, kind, q| match kind {
                SMALL | BIG => {
                    let y = e.as_int(kind, q, s);
                    e.leave(&mut ints, [x, y]);
                }
                _ if s.wide => {
                    let y = e.as_f64(kind, q, s);
                    let r = e.int_float(cc, x, y);
                    e.leave(&mut wides, [r]);
                }
                _ => {
                    let (xf, y) = (e.op1(IToF, Rep::F64, x), e.as_f64(kind, q, s));
                    e.leave(&mut floats, [xf, y]);
                }
            });
        }
        if let Some([x, y]) = self.join(ints) {
            let r = self.cmp(Cmp, cc, x, y);
            self.leave(&mut out, [r]);
        }
        if let Some([x, y]) = self.join(floats) {
            let r = self.cmp(FCmp, cc, x, y);
            self.leave(&mut out, [r]);
        }
        if let Some([r]) = self.join(wides) {
            self.leave(&mut out, [r]);
        }
        if let Some([x, y]) = self.join(strs) {
            let r = self.str_order(cc, x, y, s.fs);
            self.leave(&mut out, [r]);
        }
        self.join(out).map_or(Val::NONE, |[r]| r)
    }

    /// `x cc y` for two strings' payloads, by their bytes: the first that
    /// differ decide, else the lengths. The index counts from the bytes'
    /// offset in the payload.
    fn str_order(&mut self, cc: Cc, x: Val, y: Val, fs: Val) -> Val {
        use Op::*;
        let (cc, x, y) = match cc {
            Cc::Gt | Cc::Ge => (cc.swap(), y, x),
            _ => (cc, x, y),
        };
        let (lx, ly) = (
            self.load(Field::StrLen, x, 0),
            self.load(Field::StrLen, y, 0),
        );
        let shorter = self.cmp(Cmp, Cc::Lo, lx, ly);
        let n = self.emit(Select, Rep::I64, 0, shorter.0, lx.0, ly.0);
        let first = self.k64(Rep::I64, crate::env::LuaString::DATA_OFFSET as u64);
        let end = self.op2(Add, Rep::I64, n, first);
        let one = self.k64(Rep::I64, 1);
        let byte = Field::StrByte as u16;
        let mut out = self.new_join::<1>();
        let (lp, [at]) = self.loop_open(Heaps::empty(), fs, [first]);
        let more = self.cmp(Cmp, Cc::Lo, at, end);
        let (body, tail) = self.fork(more, false, false);
        if self.arm(body) {
            let bx = self.emit(LoadIdx, Rep::I32, byte, x.0, at.0, u32::MAX);
            let by = self.emit(LoadIdx, Rep::I32, byte, y.0, at.0, u32::MAX);
            let differ = self.cmp(Cmp, Cc::Ne, bx, by);
            let (differ, next) = self.fork(differ, false, false);
            if self.arm(differ) {
                let r = self.cmp(Cmp, Cc::Lo, bx, by);
                self.leave(&mut out, [r]);
            }
            if self.arm(next) {
                let next = self.op2(Add, Rep::I64, at, one);
                self.loop_back(lp, [next]);
            }
        }
        if self.arm(tail) {
            let ends = if cc == Cc::Lt { Cc::Lo } else { Cc::Ls };
            let r = self.cmp(Cmp, ends, lx, ly);
            self.leave(&mut out, [r]);
        }
        self.join(out).map_or(Val::NONE, |[r]| r)
    }

    // --- equality ------------------------------------------------------------

    fn eq_rule(&mut self, i: &Inst, a: Val, b: Val) -> Option<Val> {
        use Op::*;
        let (fa, fb) = (self.compared(a), self.compared(b));
        let raw = |e: &Self, v: Val| e.g.rep(v) == Rep::I64;
        let (num, float) = (TypeSet::NUM, TypeSet::FLOAT);
        // A float is equal to numbers of other kinds.
        let across = (fa.intersects(float) && fb.intersects(num))
            || (fb.intersects(float) && fa.intersects(num));
        if !fa.intersects(fb) && !across {
            return Some(self.k32(Rep::B1, 0));
        }
        if raw(self, a) && raw(self, b) {
            return Some(self.cmp(Cmp, Cc::Eq, a, b));
        }
        if raw(self, a) || raw(self, b) {
            return None;
        }
        // What is no number is equal to itself alone, but for two tables or
        // two userdata, whose `__eq` may say otherwise.
        let both = |t: TypeSet| fa.intersects(t) && fb.intersects(t);
        if (!fa.intersects(num) || !fb.intersects(num))
            && !both(TypeSet::TAB)
            && !both(TypeSet::UDATA)
        {
            return Some(self.cmp(Cmp, Cc::Eq, a, b));
        }
        let s = self.site(i, false);
        let one = |t: TypeSet| {
            [
                (TypeSet::SMALL, SMALL),
                (float, FLOAT),
                (TypeSet::BIGINT, BIG),
            ]
            .into_iter()
            .find(|k| k.0.contains(t))
            .map(|k| k.1)
        };
        match (one(fa), one(fb)) {
            (Some(SMALL), Some(SMALL)) => {}
            (Some(BIG), Some(BIG)) => {
                let (x, y) = (self.big(a, s), self.big(b, s));
                return Some(self.cmp(Cmp, Cc::Eq, x, y));
            }
            (Some(ka), Some(kb)) if ka != BIG && kb != BIG => {
                let (x, y) = (self.as_f64(ka, a, s), self.as_f64(kb, b, s));
                return Some(self.cmp(FCmp, Cc::Eq, x, y));
            }
            // A site that saw small integers alone checks for them.
            _ if seen(i.aux) != SMALL || !both(TypeSet::SMALL) => return None,
            _ => {}
        }
        let x = self.check_type(a, TypeSet::SMALL, s.tag, s.fs);
        let y = self.check_type(b, TypeSet::SMALL, s.tag, s.fs);
        Some(self.cmp(Cmp, Cc::Eq, x, y))
    }

    /// The arm where the chain's value is in `set`, tested unless nothing
    /// else is left for it to be. False where no block is left open for
    /// what else it is.
    fn kind_arm(
        &mut self,
        ch: &mut Chain,
        set: TypeSet,
        arm: &mut dyn FnMut(&mut Self, Val),
    ) -> bool {
        if !ch.rem.intersects(set) {
            return true;
        }
        if set.contains(ch.rem) {
            let p = match ch.failed {
                Some(c) => self.proved(c, false, ch.v, set),
                None => ch.v,
            };
            arm(self, p);
            ch.rem = TypeSet::empty();
            return false;
        }
        let c = self.emit(Op::IsType, Rep::B1, set.bits(), ch.v.0, 0, 0);
        let (yes, no) = self.fork(c, false, false);
        if self.arm(yes) {
            let p = self.proved(c, true, ch.v, set);
            arm(self, p);
        }
        ch.rem -= set;
        ch.failed = Some(c);
        self.arm(no)
    }

    fn chain(&self, v: Val) -> Chain {
        Chain {
            v,
            rem: self.compared(v),
            failed: None,
        }
    }

    fn eq_rewrite(&mut self, i: &Inst, a: Val, b: Val) -> Val {
        use Op::*;
        let s = self.site(i, true);
        let seen = seen(i.aux);
        let (fa, fb) = (self.compared(a), self.compared(b));
        let (small, big, float) = (TypeSet::SMALL, TypeSet::BIGINT, TypeSet::FLOAT);
        let (tab, udata) = (TypeSet::TAB, TypeSet::UDATA);
        let mut out = self.new_join::<1>();
        // Where the answer is no; two floats; a 64-bit integer and a float.
        let mut no = self.new_join::<0>();
        let (mut floats, mut mixed) = (self.new_join::<2>(), self.new_join::<2>());
        // A kind the site did not see leaves, and is recorded.
        let unseen = |e: &mut Self, tag: ExitTag| {
            e.emit(Deopt, Rep::None, tag as u16, s.fs.0, 0, 0);
        };
        let raw = [a, b].map(|v| self.g.rep(v) == Rep::I64);
        if raw[0] || raw[1] {
            // An integer held unboxed has no word to compare.
            let (n, o) = if raw[0] { (a, b) } else { (b, a) };
            let mut ch = self.chain(o);
            let mut ints = self.new_join::<1>();
            let mut open = self.kind_arm(&mut ch, small, &mut |e, p| {
                let y = e.as_int(SMALL, p, s);
                e.leave(&mut ints, [y]);
            });
            if open {
                open = self.kind_arm(&mut ch, big, &mut |e, p| {
                    let y = e.as_int(BIG, p, s);
                    e.leave(&mut ints, [y]);
                });
            }
            if open {
                open = self.kind_arm(&mut ch, float, &mut |e, p| {
                    let y = e.as_f64(FLOAT, p, s);
                    e.leave(&mut mixed, [n, y]);
                });
            }
            if open {
                self.leave(&mut no, []);
            }
            if let Some([y]) = self.join(ints) {
                let r = self.cmp(Cmp, Cc::Eq, n, y);
                self.leave(&mut out, [r]);
            }
        } else {
            let same = self.cmp(Cmp, Cc::Eq, a, b);
            let (yes, differ) = self.fork(same, false, false);
            if self.arm(yes) {
                // The same word is one value, unequal to itself as a NaN.
                let mut ch = self.chain(a);
                let mut open = true;
                if fa.intersects(float) && fb.intersects(float) {
                    open = self.kind_arm(&mut ch, float, &mut |e, p| {
                        let x = e.as_f64(FLOAT, p, s);
                        let r = e.cmp(FCmp, Cc::Eq, x, x);
                        e.leave(&mut out, [r]);
                    });
                }
                if open {
                    let yes = self.k32(Rep::B1, 1);
                    self.leave(&mut out, [yes]);
                }
            }
            if self.arm(differ) {
                let mut ch = self.chain(a);
                let mut open = true;
                if fb.intersects(TypeSet::NUM) {
                    open = self.kind_arm(&mut ch, float, &mut |e, p| {
                        let x = e.as_f64(FLOAT, p, s);
                        let mut other = e.chain(b);
                        let mut open = e.kind_arm(&mut other, float, &mut |e, q| {
                            let y = e.as_f64(FLOAT, q, s);
                            e.leave(&mut floats, [x, y]);
                        });
                        if open {
                            open = e.kind_arm(&mut other, small, &mut |e, q| {
                                let y = e.as_f64(SMALL, q, s);
                                e.leave(&mut floats, [x, y]);
                            });
                        }
                        if open {
                            open = e.kind_arm(&mut other, big, &mut |e, q| {
                                if seen & BIG == 0 {
                                    return unseen(e, ExitTag::Type);
                                }
                                let y = e.big(q, s);
                                e.leave(&mut mixed, [y, x]);
                            });
                        }
                        if open {
                            e.leave(&mut no, []);
                        }
                    });
                }
                if open && fb.intersects(float) {
                    open = self.kind_arm(&mut ch, small, &mut |e, p| {
                        let x = e.as_f64(SMALL, p, s);
                        let mut other = e.chain(b);
                        if e.kind_arm(&mut other, float, &mut |e, q| {
                            let y = e.as_f64(FLOAT, q, s);
                            e.leave(&mut floats, [x, y]);
                        }) {
                            e.leave(&mut no, []);
                        }
                    });
                }
                if open && fb.intersects(big | float) {
                    open = self.kind_arm(&mut ch, big, &mut |e, p| {
                        if seen & BIG == 0 {
                            return unseen(e, ExitTag::Type);
                        }
                        let x = e.big(p, s);
                        let mut other = e.chain(b);
                        let mut open = e.kind_arm(&mut other, big, &mut |e, q| {
                            let y = e.big(q, s);
                            let r = e.cmp(Cmp, Cc::Eq, x, y);
                            e.leave(&mut out, [r]);
                        });
                        if open {
                            open = e.kind_arm(&mut other, float, &mut |e, q| {
                                let y = e.as_f64(FLOAT, q, s);
                                e.leave(&mut mixed, [x, y]);
                            });
                        }
                        if open {
                            e.leave(&mut no, []);
                        }
                    });
                }
                if open && fb.intersects(tab) {
                    open = self.kind_arm(&mut ch, tab, &mut |e, p| {
                        if seen & TAB == 0 {
                            return unseen(e, ExitTag::Type);
                        }
                        let mut other = e.chain(b);
                        if e.kind_arm(&mut other, tab, &mut |e, q| {
                            // Either one's `__eq` would be called.
                            let x = e.check_type(p, tab, s.tag, s.fs);
                            e.no_mm(x, MetamethodBits::EQ, ExitTag::Mm, s.fs);
                            let y = e.check_type(q, tab, s.tag, s.fs);
                            e.no_mm(y, MetamethodBits::EQ, ExitTag::Mm, s.fs);
                            e.leave(&mut no, []);
                        }) {
                            e.leave(&mut no, []);
                        }
                    });
                }
                if open && fb.intersects(udata) {
                    open = self.kind_arm(&mut ch, udata, &mut |e, _| {
                        let mut other = e.chain(b);
                        if e.kind_arm(&mut other, udata, &mut |e, _| {
                            let tag = if seen & OTHER == 0 {
                                ExitTag::Type
                            } else {
                                ExitTag::Mm
                            };
                            unseen(e, tag)
                        }) {
                            e.leave(&mut no, []);
                        }
                    });
                }
                if open {
                    self.leave(&mut no, []);
                }
            }
        }
        if let Some([x, y]) = self.join(floats) {
            let r = self.cmp(FCmp, Cc::Eq, x, y);
            self.leave(&mut out, [r]);
        }
        if let Some([x, y]) = self.join(mixed) {
            let r = self.int_float(Cc::Eq, x, y);
            self.leave(&mut out, [r]);
        }
        if self.join(no).is_some() {
            let r = self.k32(Rep::B1, 0);
            self.leave(&mut out, [r]);
        }
        self.join(out).map_or(Val::NONE, |[r]| r)
    }
}
