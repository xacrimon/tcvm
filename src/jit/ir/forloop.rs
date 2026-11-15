//! The numeric `for`: generic ops that share one decision of the loop's
//! kind, each lowered by a block-free rule once that is decided.

use super::emit::Emitter;
use super::ops::{Cc, ExitTag, Op};
use super::types::{Fact, Rep, TypeSet};
use super::{Inst, Val};
use crate::jit::build::Spec;

/// The kind of a loop from its start's and step's kinds, where both are
/// decided: an integer loop when both are integers, else a float loop.
pub(crate) fn kind_of(start: Option<Spec>, step: Option<Spec>, speculated: Spec) -> Spec {
    match (start, step) {
        (Some(Spec::I32), Some(Spec::I32)) => Spec::I32,
        (Some(_), Some(_)) => Spec::F64,
        _ => speculated,
    }
}

impl Emitter {
    /// What decides a kind: copy 1 goes by definitions alone, as the
    /// analysis that typed the parameters did.
    fn deciding(&self, v: Val) -> Fact {
        if self.lowered {
            self.g.facts[v.idx()]
        } else {
            self.fact(v)
        }
    }

    /// `v` as a value of a loop of `kind`: a small integer, or a float (a
    /// known small integer converts). Anything else leaves.
    fn loop_value(&mut self, v: Val, kind: Spec, fs: Val) -> Val {
        match (kind, Spec::by_fact(self.deciding(v))) {
            (Spec::I32, _) => self.check_type(v, TypeSet::SMALL, ExitTag::Type, fs),
            (Spec::F64, Some(Spec::I32)) => {
                let x = self.check_type(v, TypeSet::SMALL, ExitTag::Type, fs);
                self.op1(Op::IToF, Rep::F64, x)
            }
            (Spec::F64, _) => self.check_type(v, TypeSet::FLOAT, ExitTag::Type, fs),
        }
    }

    /// The step as the loop's kind, which is not zero: a zero step raises,
    /// in the interpreter.
    fn loop_step(&mut self, s: Val, kind: Spec, fs: Val) -> Val {
        let st = self.loop_value(s, kind, fs);
        let moves = match kind {
            Spec::I32 => {
                let zero = self.k32(Rep::I32, 0);
                self.cmp(Op::Cmp, Cc::Ne, st, zero)
            }
            Spec::F64 => {
                let zero = self.kf64(0.0);
                self.cmp(Op::FCmp, Cc::Ne, st, zero)
            }
        };
        let (rep, tag) = (self.g.rep(st), ExitTag::Slow as u16);
        self.emit(Op::Guard, rep, tag, moves.0, fs.0, st.0)
    }

    fn boxed(&mut self, x: Val, kind: Spec) -> Val {
        match kind {
            Spec::I32 => self.op1(Op::BoxI32, Rep::Val, x),
            Spec::F64 => self.op1(Op::BoxF64, Rep::Val, x),
        }
    }

    /// `ForStart`, `ForStep`, `ForLimit`, `ForSkip`, the start's and the
    /// step's facts final.
    pub(super) fn for_prep(&mut self, i: &Inst) -> Option<Val> {
        use Op::*;
        let (x, n, s, fs) = match i.op {
            ForStart | ForStep => (Val(i.a), Val::NONE, Val(i.b), Val(i.c)),
            _ => {
                let r = self.g.run(i.a, i.b);
                (Val(r[0]), Val(r[1]), Val(r[2]), Val(i.c))
            }
        };
        let pending = |e: &Self, v: Val| !v.is_none() && e.fact(v).pending();
        if pending(self, x) || pending(self, n) || pending(self, s) {
            return None;
        }
        let by = |e: &Self, v: Val| Spec::by_fact(e.deciding(v));
        let kind = kind_of(by(self, x), by(self, s), Spec::of(i.aux));
        Some(match (i.op, kind) {
            (ForStart, _) => {
                let x = self.loop_value(x, kind, fs);
                self.boxed(x, kind)
            }
            (ForStep, _) => {
                let s = self.loop_step(s, kind, fs);
                self.boxed(s, kind)
            }
            (ForLimit, Spec::F64) => {
                let n = self.loop_value(n, kind, fs);
                self.boxed(n, kind)
            }
            (ForLimit, Spec::I32) => {
                let n = self.loop_value(n, kind, fs);
                let s = self.loop_step(s, kind, fs);
                // A unit step walks to the limit itself.
                if matches!(self.ki(s), Some(1 | -1)) {
                    return Some(self.boxed(n, kind));
                }
                // last = x + (n - x) / s * s: exact where the loop runs,
                // the quotient not negative; unused where it does not.
                let x = self.loop_value(x, kind, fs);
                let wide = |e: &mut Self, v: Val| e.op1(SExt32, Rep::I64, v);
                let (x, n, s) = (wide(self, x), wide(self, n), wide(self, s));
                let span = self.op2(Sub, Rep::I64, n, x);
                let count = self.op2(SDiv, Rep::I64, span, s);
                let off = self.op2(Mul, Rep::I64, count, s);
                let last = self.op2(Add, Rep::I64, x, off);
                let last = self.op1(Trunc32, Rep::I32, last);
                self.boxed(last, kind)
            }
            (_, Spec::I32) => {
                let (x, n) = (self.loop_value(x, kind, fs), self.loop_value(n, kind, fs));
                let s = self.loop_step(s, kind, fs);
                let zero = self.k32(Rep::I32, 0);
                let up = self.cmp(Cmp, Cc::Gt, s, zero);
                let (over, under) = (self.cmp(Cmp, Cc::Gt, x, n), self.cmp(Cmp, Cc::Lt, x, n));
                self.emit(Select, Rep::B1, 0, up.0, over.0, under.0)
            }
            (_, Spec::F64) => {
                let (x, n) = (self.loop_value(x, kind, fs), self.loop_value(n, kind, fs));
                let s = self.loop_step(s, kind, fs);
                let zero = self.kf64(0.0);
                let up = self.cmp(FCmp, Cc::Gt, s, zero);
                let (over, under) = (self.cmp(FCmp, Cc::Lt, n, x), self.cmp(FCmp, Cc::Lt, x, n));
                self.emit(Select, Rep::B1, 0, up.0, over.0, under.0)
            }
        })
    }

    /// `ForMore` and `ForNext`, their operands' facts final. The step's
    /// kind is the loop's (it is what `ForStep` made); in a region entered
    /// at the loop's body nothing types it, and the speculation is checked.
    pub(super) fn for_step(&mut self, i: &Inst) -> Option<Val> {
        use Op::*;
        let more = i.op == ForMore;
        let (idx, last, s, fs) = if more {
            let r = self.g.run(i.a, i.b);
            (Val(r[0]), Val(r[1]), Val(r[2]), Val(i.c))
        } else {
            (Val(i.a), Val::NONE, Val(i.b), Val::NONE)
        };
        let pending = |e: &Self, v: Val| !v.is_none() && e.fact(v).pending();
        if pending(self, idx) || pending(self, last) || pending(self, s) {
            return None;
        }
        let kind = Spec::by_fact(self.deciding(s)).unwrap_or(Spec::of(i.aux));
        if !more {
            // What `ForMore` checked, read as the numbers they are.
            let read = |e: &mut Self, v: Val| match (kind, Spec::by_fact(e.deciding(v))) {
                (Spec::F64, Some(Spec::I32)) => {
                    let x = e.unboxed(v, TypeSet::SMALL);
                    e.op1(IToF, Rep::F64, x)
                }
                _ => e.unboxed(v, kind.set()),
            };
            let (x, st) = (read(self, idx), read(self, s));
            let next = match kind {
                // It cannot overflow: the counter walks to the last value.
                Spec::I32 => self.op2(Add, Rep::I32, x, st),
                Spec::F64 => self.op2(FAdd, Rep::F64, x, st),
            };
            return Some(self.boxed(next, kind));
        }
        let value = |e: &mut Self, v: Val| e.loop_value(v, kind, fs);
        let (x, n, st) = (value(self, idx), value(self, last), value(self, s));
        Some(match kind {
            Spec::I32 => self.cmp(Cmp, Cc::Ne, x, n),
            Spec::F64 => {
                let next = self.op2(FAdd, Rep::F64, x, st);
                let zero = self.kf64(0.0);
                let up = self.cmp(FCmp, Cc::Gt, st, zero);
                let (below, above) = (
                    self.cmp(FCmp, Cc::Le, next, n),
                    self.cmp(FCmp, Cc::Le, n, next),
                );
                self.emit(Select, Rep::B1, 0, up.0, below.0, above.0)
            }
        })
    }
}
