//! `Arith` and `Unm` by kinds. An operand's kinds are the number and string
//! types its fact leaves that its site saw (a string's arithmetic is the
//! string library's metamethod, which converts it: the helper's); the op's coercion rule says
//! what code a combination takes: always a float op, always an integer op,
//! or an integer op where both are integers and a float op otherwise. A
//! site with one leaf is a block-free rule; the rest is a rewrite.

use super::emit::{Emitter, Join};
use super::ops::{ExitTag, Helper, Op};
use super::types::{Rep, TypeSet};
use super::{Inst, Val};
use crate::instruction::ArithKind;
use crate::jit::bug;

/// The kinds an arm can be made for, as bits of a set.
pub(crate) const SMALL: u8 = 1;
pub(crate) const BIG: u8 = 2;
pub(crate) const FLOAT: u8 = 4;
pub(crate) const STR: u8 = 8;
const INTS: u8 = SMALL | BIG;

/// `aux`: the `ArithKind`, each operand's seen kinds, then these.
pub(crate) const SEEN_SHIFT: [u32; 2] = [4, 8];
/// The site overflowed 32 bits.
pub(crate) const OVERFLOWED: u16 = 1 << 12;
/// A metamethod ran at the site: what is no kind leaves with `Mm`.
pub(crate) const MM: u16 = 1 << 13;

const KINDS: [(u8, TypeSet); 4] = [
    (SMALL, TypeSet::SMALL),
    (BIG, TypeSet::BIGINT),
    (FLOAT, TypeSet::FLOAT),
    (STR, TypeSet::STR),
];

pub(crate) fn kinds_of(set: TypeSet) -> u8 {
    (KINDS.iter()).fold(0, |k, &(bit, t)| {
        k | if set.intersects(t) { bit } else { 0 }
    })
}

pub(crate) fn set_of(kinds: u8) -> TypeSet {
    (KINDS.iter())
        .filter(|&&(bit, _)| kinds & bit != 0)
        .fold(TypeSet::empty(), |s, &(_, t)| s | t)
}

/// How an op coerces its operands.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Class {
    /// `/`, `^`: each operand a float.
    Float,
    /// `&`, `|`, `~`, `<<`, `>>`: each operand an integer.
    Int,
    /// `+`, `-`, `*`, `//`, `%`, unary minus: integers stay integers.
    Mixed,
}

fn class(low: u16) -> Class {
    let is = |k: ArithKind| low == k as u16;
    if is(ArithKind::Div) || is(ArithKind::Pow) {
        Class::Float
    } else if [
        ArithKind::BAnd,
        ArithKind::BOr,
        ArithKind::BXor,
        ArithKind::Shl,
        ArithKind::Shr,
    ]
    .iter()
    .any(|&k| is(k))
    {
        Class::Int
    } else {
        Class::Mixed
    }
}

/// The kinds of a type an arm is made for at a site that saw `seen`: no
/// bitwise op converts a string.
fn armed(set: TypeSet, seen: u8, low: u16) -> u8 {
    let own = kinds_of(set);
    // A type known to be numbers or strings the site never saw as such is
    // what it is: its feedback is stale, and leaving would say nothing.
    let known = !set.is_empty() && (TypeSet::NUM | TypeSet::STR).contains(set);
    let all = if known && own & seen == 0 {
        own
    } else {
        own & seen
    };
    if class(low) == Class::Int {
        all & !STR
    } else {
        all
    }
}

/// Whether operand `k` of a site with this `aux` takes an integer held as
/// `I64` as it is: one of its armed kinds is the 64-bit integer.
pub(crate) fn takes_i64(aux: u16, k: usize) -> bool {
    armed(TypeSet::INT, (aux >> SEEN_SHIFT[k]) as u8 & 15, aux & 15) & BIG != 0
}

/// The types an `Arith` (`b`: its right operand's type) or `Unm` gives,
/// from its operands' types: the union of its leaves' results. The rule,
/// the rewrite and the analysis all go by this.
pub(crate) fn result(aux: u16, a: TypeSet, b: Option<TypeSet>) -> TypeSet {
    let low = aux & 15;
    let ka = armed(a, (aux >> SEEN_SHIFT[0]) as u8 & 15, low);
    let kb = b.map(|b| armed(b, (aux >> SEEN_SHIFT[1]) as u8 & 15, low));
    let both = ka | kb.unwrap_or(0);
    if ka == 0 || kb == Some(0) {
        return TypeSet::empty();
    }
    let wide = aux & OVERFLOWED != 0 || both & BIG != 0;
    let ints = if wide { TypeSet::INT } else { TypeSet::SMALL };
    let kb = kb.unwrap_or(ka);
    let mut out = TypeSet::empty();
    // A string's number is of either kind.
    if both & STR != 0 {
        out |= TypeSet::NUM;
    }
    let nums = SMALL | BIG | FLOAT;
    match class(low) {
        Class::Float if ka & nums != 0 && kb & nums != 0 => out |= TypeSet::FLOAT,
        Class::Float => {}
        Class::Int if ka == SMALL && kb == SMALL && !wide => out |= TypeSet::SMALL,
        Class::Int => out |= TypeSet::INT,
        Class::Mixed => {
            if ka & INTS != 0 && kb & INTS != 0 {
                out |= ints;
            }
            if (ka & FLOAT != 0 && kb & nums != 0) || (kb & FLOAT != 0 && ka & nums != 0) {
                out |= TypeSet::FLOAT;
            }
        }
    }
    out
}

/// One operand at a site.
#[derive(Clone, Copy)]
pub(super) struct Side {
    pub(super) v: Val,
    /// The kinds it gets arms for.
    pub(super) armed: u8,
    /// Whether its fact allows anything else: its last test then fails to
    /// an exit.
    pub(super) rest: bool,
}

/// What a site's leaves share.
#[derive(Clone, Copy)]
pub(super) struct Site {
    pub(super) low: u16,
    pub(super) wide: bool,
    /// The exit of a value that is no kind.
    pub(super) tag: ExitTag,
    pub(super) fs: Val,
    pub(super) unary: bool,
    /// Every leaf gives a 64-bit integer: the result stays an `I64`, which
    /// what takes it boxed boxes, and the next integer op takes as it is.
    pub(super) raw: bool,
}

impl Emitter {
    pub(super) fn side(&self, v: Val, seen: u8, low: u16) -> Side {
        // An integer the copy hands over unboxed.
        if self.g.rep(v) == Rep::I64 {
            return Side {
                v,
                armed: BIG,
                rest: false,
            };
        }
        let set = self.deciding(v).set;
        let armed = armed(set, seen, low);
        Side {
            v,
            armed,
            rest: !set_of(armed).contains(set),
        }
    }

    /// The operands and what the site's leaves share; none while an
    /// operand's type is not final.
    fn arith_site(&self, i: &Inst) -> Option<(Side, Side, Site)> {
        let unary = i.op == Op::Unm;
        let (a, b, fs) = if unary {
            (Val(i.a), Val(i.a), Val(i.b))
        } else {
            (Val(i.a), Val(i.b), Val(i.c))
        };
        if self.fact(a).pending() || self.fact(b).pending() {
            return None;
        }
        let low = i.aux & 15;
        let seen = |k: usize| (i.aux >> SEEN_SHIFT[k]) as u8 & 15;
        let (a, b) = (
            self.side(a, seen(0), low),
            self.side(b, seen(if unary { 0 } else { 1 }), low),
        );
        let site = Site {
            low,
            wide: i.aux & OVERFLOWED != 0 || (a.armed | b.armed) & BIG != 0,
            tag: if i.aux & MM != 0 {
                ExitTag::Mm
            } else {
                ExitTag::Type
            },
            fs,
            unary,
            raw: false,
        };
        Some((a, b, site))
    }

    /// The fact of an `Arith` or `Unm` that stays generic.
    pub(super) fn arith_fact(&self, i: &Inst) -> TypeSet {
        let of = |w: u32| self.fact(Val(w)).set;
        if i.op == Op::Unm {
            result(i.aux, of(i.a), None)
        } else {
            result(i.aux, of(i.a), Some(of(i.b)))
        }
    }

    /// The rule of `Arith` and `Unm`: a site with one leaf whose integers
    /// are small. A site no arm can be made for leaves.
    pub(super) fn arith_rule(&mut self, i: &Inst) -> Option<Val> {
        use Op::*;
        let (a, b, s) = self.arith_site(i)?;
        if a.armed == 0 || b.armed == 0 {
            self.emit(Deopt, Rep::None, s.tag as u16, s.fs.0, 0, 0);
            return Some(Val::NONE);
        }
        let one = |x: Side| x.armed.is_power_of_two();
        if s.wide || !one(a) || !one(b) {
            return None;
        }
        if (a.armed | b.armed) & STR != 0 {
            return Some(self.coerced(a.v, b.v, s));
        }
        let small = |e: &mut Self, x: Side| e.check_type(x.v, TypeSet::SMALL, s.tag, s.fs);
        let float = |e: &mut Self, x: Side| match x.armed {
            SMALL => {
                let n = e.check_type(x.v, TypeSet::SMALL, s.tag, s.fs);
                e.op1(IToF, Rep::F64, n)
            }
            _ => e.check_type(x.v, TypeSet::FLOAT, s.tag, s.fs),
        };
        let ints = a.armed == SMALL && b.armed == SMALL;
        Some(match class(s.low) {
            // A float as an integer is 64 bits wide: the rewrite's.
            Class::Int if !ints => return None,
            Class::Int | Class::Mixed if ints => {
                let (x, y) = (small(self, a), small(self, b));
                let r = self.int_op(x, y, s);
                self.op1(BoxI32, Rep::Val, r)
            }
            _ => {
                let (x, y) = (float(self, a), float(self, b));
                let r = self.float_op(x, y, s);
                self.op1(BoxF64, Rep::Val, r)
            }
        })
    }

    /// The whole op by the helper that converts strings: a boxed number,
    /// or an exit where it fails (the interpreter raises, or calls).
    fn coerced(&mut self, a: Val, b: Val, s: Site) -> Val {
        let boxed = |e: &mut Self, v: Val| match e.g.rep(v) {
            Rep::I64 => e.lower(Op::BoxI64, Rep::Val, 0, v.0, 0, 0),
            _ => v,
        };
        let a = boxed(self, a);
        if s.unary {
            return self.helped(Helper::Unm, 0, &[a], s.fs);
        }
        let b = boxed(self, b);
        self.helped(Helper::ArithCoerce, s.low as u32, &[a, b], s.fs)
    }

    /// The integer op on two `I32`s, or two `I64`s at a wide site (which
    /// wraps); unary minus takes `x`.
    fn int_op(&mut self, x: Val, y: Val, s: Site) -> Val {
        use Op::*;
        let rep = if s.wide { Rep::I64 } else { Rep::I32 };
        let exits =
            |e: &mut Self, op: Op, tag: ExitTag| e.emit(op, rep, tag as u16, x.0, y.0, s.fs.0);
        if s.unary {
            return match s.wide {
                true => self.op1(Neg, rep, x),
                false => self.emit(NegOvf, rep, ExitTag::Overflow as u16, x.0, s.fs.0, 0),
            };
        }
        let is = |k: ArithKind| s.low == k as u16;
        let checked = [
            (ArithKind::Add, AddOvf, Add),
            (ArithKind::Sub, SubOvf, Sub),
            (ArithKind::Mul, MulOvf, Mul),
        ];
        if let Some(&(_, narrow, wrapping)) = checked.iter().find(|c| is(c.0)) {
            return match s.wide {
                true => self.op2(wrapping, rep, x, y),
                false => exits(self, narrow, ExitTag::Overflow),
            };
        }
        if is(ArithKind::IDiv) {
            // A zero divisor raises, in the interpreter.
            exits(self, DivFloor, ExitTag::Slow)
        } else if is(ArithKind::Mod) {
            exits(self, ModFloor, ExitTag::Slow)
        } else if is(ArithKind::BAnd) {
            self.op2(And, rep, x, y)
        } else if is(ArithKind::BOr) {
            self.op2(Or, rep, x, y)
        } else if is(ArithKind::BXor) {
            self.op2(Xor, rep, x, y)
        } else {
            // A small shift's result may not be small: it leaves then.
            let op = if is(ArithKind::Shl) { LuaShl } else { LuaShr };
            let fs = if s.wide { u32::MAX } else { s.fs.0 };
            self.emit(op, rep, ExitTag::Overflow as u16, x.0, y.0, fs)
        }
    }

    fn float_op(&mut self, x: Val, y: Val, s: Site) -> Val {
        use Op::*;
        if s.unary {
            return self.op1(FNeg, Rep::F64, x);
        }
        let is = |k: ArithKind| s.low == k as u16;
        let ops = [
            (ArithKind::Add, FAdd),
            (ArithKind::Sub, FSub),
            (ArithKind::Mul, FMul),
            (ArithKind::Div, FDiv),
            (ArithKind::IDiv, FIDiv),
            (ArithKind::Mod, FMod),
        ];
        match ops.iter().find(|o| is(o.0)) {
            Some(&(_, op)) => self.op2(op, Rep::F64, x, y),
            None if is(ArithKind::Pow) => self.call(Helper::Pow, 0, &[x, y]),
            None => bug!("a float op of arithmetic kind {}", s.low),
        }
    }

    /// An arm per armed kind of `x`, integers first, each tested but a last
    /// one nothing else is possible past: `arm` gets the kind and the value
    /// typed as it, and ends its block. Where the fact allows more, what is
    /// no armed kind leaves.
    pub(super) fn by_kind(&mut self, x: Side, s: Site, arm: &mut dyn FnMut(&mut Self, u8, Val)) {
        if self.g.rep(x.v) == Rep::I64 {
            return arm(self, BIG, x.v);
        }
        let n = x.armed.count_ones();
        // The test whose false arm is open.
        let mut failed = None;
        let kinds = KINDS.iter().filter(|k| x.armed & k.0 != 0);
        for (at, &(kind, set)) in kinds.enumerate() {
            let last = at as u32 + 1 == n && !x.rest;
            match failed {
                Some(c) if last => {
                    let p = self.proved(c, false, x.v, set);
                    return arm(self, kind, p);
                }
                None if last => return arm(self, kind, x.v),
                _ => {}
            }
            let c = self.emit(Op::IsType, Rep::B1, set.bits(), x.v.0, 0, 0);
            let (yes, no) = self.fork(c, false, false);
            if self.arm(yes) {
                let p = self.proved(c, true, x.v, set);
                arm(self, kind, p);
            }
            if !self.arm(no) {
                return;
            }
            failed = Some(c);
        }
        self.emit(Op::Deopt, Rep::None, s.tag as u16, s.fs.0, 0, 0);
    }

    /// The integer of `p`, typed as integer kind `kind` (or an `I64`
    /// already): `I64` at a wide site, else `I32`.
    pub(super) fn as_int(&mut self, kind: u8, p: Val, s: Site) -> Val {
        use Op::*;
        if self.g.rep(p) == Rep::I64 {
            return p;
        }
        // Read through what types the value: where a fact alone does, the
        // check again.
        if kind == BIG {
            return self.big(p, s);
        }
        let x = self.check_type(p, TypeSet::SMALL, s.tag, s.fs);
        match s.wide {
            true => self.op1(SExt32, Rep::I64, x),
            false => x,
        }
    }

    /// The payload of `p`, known here a boxed integer: read through what
    /// types it, which is the check again where a fact alone does.
    pub(super) fn big(&mut self, p: Val, s: Site) -> Val {
        self.check_type(p, TypeSet::BIGINT, s.tag, s.fs);
        match self.typed(p, TypeSet::BIGINT) {
            Some(t) => self.boxed_int(t),
            None => bug!("nothing types {p:?} a boxed integer past its check"),
        }
    }

    /// The float of `p`, typed as number kind `kind`.
    pub(super) fn as_f64(&mut self, kind: u8, p: Val, s: Site) -> Val {
        use Op::*;
        if self.g.rep(p) == Rep::I64 {
            return self.op1(LToF, Rep::F64, p);
        }
        match kind {
            SMALL => {
                let x = self.check_type(p, TypeSet::SMALL, s.tag, s.fs);
                self.op1(IToF, Rep::F64, x)
            }
            BIG => {
                let x = self.big(p, s);
                self.op1(LToF, Rep::F64, x)
            }
            _ => self.check_type(p, TypeSet::FLOAT, s.tag, s.fs),
        }
    }

    /// An integer result: boxed, a small one or at a wide site whichever
    /// it fits; left an `I64` where the site gives nothing else.
    fn int_result(&mut self, r: Val, s: Site) -> Val {
        match (s.wide, s.raw) {
            (true, true) => r,
            (true, false) => self.lower(Op::BoxI64, Rep::Val, 0, r.0, 0, 0),
            (false, _) => self.op1(Op::BoxI32, Rep::Val, r),
        }
    }

    /// The rewrite of `Arith` and `Unm`, the operands' facts final: the
    /// tests their kinds need, then the one integer op and the one float op
    /// the paths join in. The result is boxed, or an `I64` where a 64-bit
    /// integer is all the site gives.
    pub(super) fn arith_rewrite(&mut self, i: &Inst) -> Val {
        let Some((a, b, s)) = self.arith_site(i) else {
            return self.append(*i);
        };
        if a.armed == 0 || b.armed == 0 {
            self.emit(Op::Deopt, Rep::None, s.tag as u16, s.fs.0, 0, 0);
            return Val::NONE;
        }
        // A float's result and the helper's are boxed.
        let boxes = (a.armed | b.armed) & (FLOAT | STR) != 0;
        let s = Site {
            raw: s.wide && !boxes,
            ..s
        };
        let mut out = self.new_join::<1>();
        // Where a string was: the whole op is the helper's.
        let mut strs = self.new_join::<0>();
        match (class(s.low), s.unary) {
            (_, true) => self.unary(a, s, &mut out, &mut strs),
            (Class::Mixed, _) => self.mixed(a, b, s, &mut out, &mut strs),
            (c, _) => self.each(a, b, s, c == Class::Float, &mut out, &mut strs),
        }
        if self.join(strs).is_some() {
            let r = self.coerced(a.v, b.v, s);
            self.leave(&mut out, [r]);
        }
        self.join(out).map_or(Val::NONE, |[r]| r)
    }

    /// Unary minus: the integer negated where the kinds join, the float in
    /// its arm.
    fn unary(&mut self, a: Side, s: Site, out: &mut Join<1>, strs: &mut Join<0>) {
        let mut ints = self.new_join::<1>();
        self.by_kind(a, s, &mut |e, kind, p| match kind {
            SMALL | BIG => {
                let x = e.as_int(kind, p, s);
                e.leave(&mut ints, [x]);
            }
            STR => e.leave(strs, []),
            _ => {
                let x = e.as_f64(kind, p, s);
                let r = e.float_op(x, x, s);
                let r = e.op1(Op::BoxF64, Rep::Val, r);
                e.leave(out, [r]);
            }
        });
        if let Some([x]) = self.join(ints) {
            let r = self.int_op(x, x, s);
            let r = self.int_result(r, s);
            self.leave(out, [r]);
        }
    }

    /// An op that converts each operand on its own: to a float (`/`, `^`),
    /// or to an integer, which a float is only where it holds one exactly.
    fn each(
        &mut self,
        a: Side,
        b: Side,
        s: Site,
        float: bool,
        out: &mut Join<1>,
        strs: &mut Join<0>,
    ) {
        // An integer of this class is 64 bits wide: a float converts to one.
        let s = Site {
            wide: true,
            raw: !float,
            ..s
        };
        let value = |e: &mut Self, kind: u8, p: Val| match (float, kind) {
            (true, _) => e.as_f64(kind, p, s),
            (false, FLOAT) => {
                let x = e.as_f64(kind, p, s);
                // No integer value: the interpreter raises.
                let tag = ExitTag::Slow as u16;
                e.emit(Op::FToLExact, Rep::I64, tag, x.0, s.fs.0, 0)
            }
            (false, _) => e.as_int(kind, p, s),
        };
        let (mut xs, mut ys) = (self.new_join::<1>(), self.new_join::<1>());
        self.by_kind(a, s, &mut |e, kind, p| match kind {
            STR => e.leave(strs, []),
            _ => {
                let x = value(e, kind, p);
                e.leave(&mut xs, [x]);
            }
        });
        let Some([x]) = self.join(xs) else {
            return;
        };
        self.by_kind(b, s, &mut |e, kind, p| match kind {
            STR => e.leave(strs, []),
            _ => {
                let y = value(e, kind, p);
                e.leave(&mut ys, [y]);
            }
        });
        if let Some([y]) = self.join(ys) {
            let r = match float {
                true => {
                    let r = self.float_op(x, y, s);
                    self.op1(Op::BoxF64, Rep::Val, r)
                }
                false => {
                    let r = self.int_op(x, y, s);
                    self.int_result(r, s)
                }
            };
            self.leave(out, [r]);
        }
    }

    /// An op that is an integer's where both operands are integers, else a
    /// float's: `a`'s class first, then `b`'s under an integer `a`; under a
    /// float `a`, `b` only converts.
    fn mixed(&mut self, a: Side, b: Side, s: Site, out: &mut Join<1>, strs: &mut Join<0>) {
        let (mut ints, mut floats) = (self.new_join::<2>(), self.new_join::<2>());
        let mut int_a = self.new_join::<1>();
        self.by_kind(a, s, &mut |e, kind, p| match kind {
            SMALL | BIG => {
                let x = e.as_int(kind, p, s);
                e.leave(&mut int_a, [x]);
            }
            STR => e.leave(strs, []),
            _ => {
                let x = e.as_f64(kind, p, s);
                e.by_kind(b, s, &mut |e, kind, q| match kind {
                    STR => e.leave(strs, []),
                    _ => {
                        let y = e.as_f64(kind, q, s);
                        e.leave(&mut floats, [x, y]);
                    }
                });
            }
        });
        if let Some([x]) = self.join(int_a) {
            self.by_kind(b, s, &mut |e, kind, q| match kind {
                SMALL | BIG => {
                    let y = e.as_int(kind, q, s);
                    e.leave(&mut ints, [x, y]);
                }
                STR => e.leave(strs, []),
                _ => {
                    let op = if s.wide { Op::LToF } else { Op::IToF };
                    let (xf, y) = (e.op1(op, Rep::F64, x), e.as_f64(kind, q, s));
                    e.leave(&mut floats, [xf, y]);
                }
            });
        }
        if let Some([x, y]) = self.join(ints) {
            let r = self.int_op(x, y, s);
            let r = self.int_result(r, s);
            self.leave(out, [r]);
        }
        if let Some([x, y]) = self.join(floats) {
            let r = self.float_op(x, y, s);
            let r = self.op1(Op::BoxF64, Rep::Val, r);
            self.leave(out, [r]);
        }
    }
}
