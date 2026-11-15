//! Builtins compiled in place: a `CALL` whose callee is, at compile time,
//! a builtin with an intrinsic guards the callee by identity and computes
//! the result; anything the code does not cover exits before the `CALL`,
//! and the interpreter runs the builtin.

use super::Builder;
use crate::env::MetamethodBits;
use crate::env::function::{Function, NativeFn, NativeKind};
use crate::env::value::Value;
use crate::instruction::Instruction;
use crate::jit::feedback as fb;
use crate::jit::ir::Val;
use crate::jit::ir::fields::Field;
use crate::jit::ir::ops::{Cc, ExitTag, Helper, Op};
use crate::jit::ir::types::{Rep, TypeSet};
use crate::vm::abi::Handler;
use crate::vm::ff;
use crate::vm::native::ContFn;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Intrinsic {
    Sqrt,
    /// A one-argument libm function.
    Unary(Helper),
    /// `atan(y [, x])`: `atan2`.
    Atan,
    /// `log(x [, base])`.
    Log,
    Floor,
    Ceil,
    Abs,
    Assert,
    Error,
    Pairs,
    Ipairs,
    StrLen,
    /// `min(...)`, `max(...)` (true).
    MinMax(bool),
    /// A builtin computed by its helper.
    Builtin(Helper),
}

/// The intrinsic of a builtin, known by its fast entry (each builtin with
/// one has its own), else by its function.
fn intrinsic_of(f: Function<'_>) -> Option<Intrinsic> {
    use crate::builtin::basic as base;
    let nc = f.as_native()?;
    let plain: &[(NativeFn, Intrinsic)] = &[
        (base::lua_type, Intrinsic::Builtin(Helper::Type)),
        (base::lua_error, Intrinsic::Error),
        (
            base::lua_getmetatable,
            Intrinsic::Builtin(Helper::GetMetatable),
        ),
        (base::lua_rawget, Intrinsic::Builtin(Helper::RawGet)),
        (base::lua_rawset, Intrinsic::Builtin(Helper::RawSet)),
        (base::lua_rawequal, Intrinsic::Builtin(Helper::RawEqual)),
        (base::lua_rawlen, Intrinsic::Builtin(Helper::RawLen)),
        (base::lua_tonumber, Intrinsic::Builtin(Helper::ToNumber)),
        #[cfg(feature = "lib-string")]
        (crate::builtin::string::lua_len, Intrinsic::StrLen),
        #[cfg(feature = "lib-string")]
        (
            crate::builtin::string::lua_byte,
            Intrinsic::Builtin(Helper::StrByte),
        ),
    ];
    match nc.function {
        NativeKind::Plain(p) => {
            if let Some(&(_, i)) = plain.iter().find(|(f, _)| *f as usize == p as usize) {
                return Some(i);
            }
        }
        NativeKind::Cont(p) => {
            let cont: &[(ContFn, Intrinsic)] = &[
                (base::lua_tostring, Intrinsic::Builtin(Helper::ToString)),
                #[cfg(feature = "lib-table")]
                (
                    crate::builtin::table::lua_insert,
                    Intrinsic::Builtin(Helper::TInsert),
                ),
                #[cfg(feature = "lib-table")]
                (
                    crate::builtin::table::lua_remove,
                    Intrinsic::Builtin(Helper::TRemove),
                ),
                #[cfg(feature = "lib-math")]
                (crate::builtin::math::lua_min, Intrinsic::MinMax(false)),
                #[cfg(feature = "lib-math")]
                (crate::builtin::math::lua_max, Intrinsic::MinMax(true)),
            ];
            if let Some(&(_, i)) = cont.iter().find(|(f, _)| *f as usize == p as usize) {
                return Some(i);
            }
        }
        _ => {}
    }
    let entry = nc.entry as usize;
    let table: [(Handler, Intrinsic); 17] = [
        (ff::ff_sqrt, Intrinsic::Sqrt),
        (ff::ff_sin, Intrinsic::Unary(Helper::Sin)),
        (ff::ff_cos, Intrinsic::Unary(Helper::Cos)),
        (ff::ff_tan, Intrinsic::Unary(Helper::Tan)),
        (ff::ff_asin, Intrinsic::Unary(Helper::Asin)),
        (ff::ff_acos, Intrinsic::Unary(Helper::Acos)),
        (ff::ff_exp, Intrinsic::Unary(Helper::Exp)),
        (ff::ff_atan, Intrinsic::Atan),
        (ff::ff_log, Intrinsic::Log),
        (ff::ff_floor, Intrinsic::Floor),
        (ff::ff_ceil, Intrinsic::Ceil),
        (ff::ff_abs, Intrinsic::Abs),
        (ff::ff_assert, Intrinsic::Assert),
        (crate::vm::native::ff_pairs, Intrinsic::Pairs),
        (crate::vm::native::ff_ipairs, Intrinsic::Ipairs),
        (
            ff::ff_setmetatable,
            Intrinsic::Builtin(Helper::SetMetatable),
        ),
        (ff::ff_sub, Intrinsic::Builtin(Helper::StrSub)),
    ];
    table
        .iter()
        .find(|(h, _)| *h as usize == entry)
        .map(|&(_, i)| i)
}

impl<'gc> Builder<'_, 'gc> {
    /// The function the call at `pc` calls when the builder may take it for
    /// a constant, which its user guards by identity (`pin`): a constant
    /// callee, else the one function the site has called. Neither once such
    /// a guard failed.
    fn seen_callee(&self, pc: u32, callee: Val) -> Option<Function<'gc>> {
        let d = self.e.g.inst(callee);
        if d.op == Op::KObj {
            // SAFETY: a pool word is a value's.
            let v: Value<'gc> = unsafe { Value::from_raw(self.pool.words[d.a as usize]) };
            return v.get_function();
        }
        let feedback = &self.proto.feedback;
        let seen = feedback.callee(pc as usize)?.get();
        // Bit 0: megamorphic, or a prototype's.
        if seen == 0 || seen & 1 != 0 || feedback.bits(pc as usize) & fb::CONST_MISS != 0 {
            return None;
        }
        // SAFETY: the value of the function the site called.
        unsafe { Value::<'gc>::from_raw(seen) }.get_function()
    }

    /// Guard `callee` to be the builtin `f`.
    fn pin(&mut self, f: Function<'gc>, callee: Val) {
        let k = self.konst(Value::function(f));
        let same = self.e.cmp(Op::Cmp, Cc::Eq, callee, k);
        self.guard(same, ExitTag::Const);
    }

    /// `x` as a float: a small integer converted, a float as it is; any
    /// other kind leaves, and the builtin converts or raises.
    fn float_arg(&mut self, x: Val) -> Val {
        let s = self.state();
        let set = self.e.fact(x).set;
        let known = |k: TypeSet| !set.is_empty() && k.contains(set);
        if known(TypeSet::FLOAT) {
            return self.e.check_type(x, TypeSet::FLOAT, ExitTag::Type, s);
        }
        let small = |b: &mut Self| {
            let i = b.e.check_type(x, TypeSet::SMALL, ExitTag::Type, s);
            b.e.op1(Op::IToF, Rep::F64, i)
        };
        if known(TypeSet::SMALL) {
            return small(self);
        }
        let is = (self.e).emit(Op::IsType, Rep::B1, TypeSet::SMALL.bits(), x.0, 0, 0);
        // Decided: one arm, whose check leaves for a kind that is neither.
        if self.e.known(is) == Some(false) {
            return self.e.check_type(x, TypeSet::FLOAT, ExitTag::Type, s);
        }
        let other = self.arm(is);
        let fi = small(self);
        let join = self.e.new_block();
        self.e.jump(join);
        self.open(other, false);
        let ff = self.e.check_type(x, TypeSet::FLOAT, ExitTag::Type, s);
        self.e.jump(join);
        self.open(join, false);
        self.e.param(Rep::F64, None, &[fi, ff])
    }

    /// The value one of two arms makes of the number `x`, by its kind; any
    /// other kind leaves.
    fn by_kind(
        &mut self,
        x: Val,
        small: impl Fn(&mut Self, Val) -> Val,
        float: impl Fn(&mut Self, Val) -> Val,
    ) -> Val {
        let s = self.state();
        let is = (self.e).emit(Op::IsType, Rep::B1, TypeSet::SMALL.bits(), x.0, 0, 0);
        // Decided: one arm, whose check leaves for a kind that is neither.
        match self.e.known(is) {
            Some(true) => {
                let i = self.e.check_type(x, TypeSet::SMALL, ExitTag::Type, s);
                return small(self, i);
            }
            Some(false) => {
                let f = self.e.check_type(x, TypeSet::FLOAT, ExitTag::Type, s);
                return float(self, f);
            }
            None => {}
        }
        let other = self.arm(is);
        let i = self.e.check_type(x, TypeSet::SMALL, ExitTag::Type, s);
        let ri = small(self, i);
        let join = self.e.new_block();
        self.e.jump(join);
        self.open(other, false);
        let f = self.e.check_type(x, TypeSet::FLOAT, ExitTag::Type, s);
        let rf = float(self, f);
        self.e.jump(join);
        self.open(join, false);
        self.e.param(Rep::Val, None, &[ri, rf])
    }

    /// Compile the `CALL` at `pc` as its callee's intrinsic, if it has one
    /// that covers the call's shape: a static count of arguments and at
    /// most one result (the iterators and `assert`: those wanted).
    pub(super) fn intrinsic(&mut self, pc: u32, i: Instruction, callee: Val) -> bool {
        let (a, nb, c) = i.abc();
        let Some(f) = self.seen_callee(pc, callee) else {
            return false;
        };
        let Some(intr) = intrinsic_of(f) else {
            return false;
        };
        let nargs = nb - 1;
        let arg = |b: &Self, k: u8| b.reg(a + 4 + k);
        let nil = |b: &mut Self| b.konst(Value::nil());
        let s = self.state();
        let bits = self.proto.feedback.bits(pc as usize);
        let one = c == 1 || c == 2;
        let r = match intr {
            Intrinsic::Error => {
                // The interpreter's builtin raises; nothing follows.
                self.pin(f, callee);
                self.deopt(ExitTag::Slow);
                return true;
            }
            // Without an argument the builtin raises.
            Intrinsic::Assert if nargs > 0 && c != 0 => {
                self.pin(f, callee);
                let x = arg(self, 0);
                let holds = self.e.emit(Op::Truthy, Rep::B1, 0, x.0, 0, 0);
                self.guard(holds, ExitTag::Slow);
                let vals: Vec<Val> = (0..c - 1)
                    .map(|k| if k < nargs { arg(self, k) } else { nil(self) })
                    .collect();
                for (k, v) in vals.into_iter().enumerate() {
                    self.set(a + k as u8, v);
                }
                return true;
            }
            Intrinsic::Pairs | Intrinsic::Ipairs if nargs == 1 && (1..=5).contains(&c) => {
                self.pin(f, callee);
                let x = arg(self, 0);
                let (iter, control) = if intr == Intrinsic::Pairs {
                    // A table without `__pairs` (`ff_pairs`).
                    let t = self.table(x);
                    (self.e).no_mm(t, MetamethodBits::PAIRS, ExitTag::Mm, s);
                    (self.links.next_fn, nil(self))
                } else {
                    (self.links.ipairs_iter, self.konst(Value::small(0)))
                };
                // SAFETY: a function's value, which the runtime holds.
                let iter = self.konst(unsafe { Value::from_raw(iter) });
                let results = [iter, x, control, nil(self)];
                for (k, &r) in results.iter().take(c as usize - 1).enumerate() {
                    self.set(a + k as u8, r);
                }
                return true;
            }
            Intrinsic::MinMax(max) if nargs >= 2 && one => {
                self.pin(f, callee);
                // As the builtin: the first argument unless a later one
                // compares past it.
                let mut best = arg(self, 0);
                let mut bf = self.float_arg(best);
                for k in 1..nargs {
                    let x = arg(self, k);
                    let xf = self.float_arg(x);
                    let past = match max {
                        true => self.e.cmp(Op::FCmp, Cc::Lt, bf, xf),
                        false => self.e.cmp(Op::FCmp, Cc::Lt, xf, bf),
                    };
                    best = (self.e).emit(Op::Select, Rep::Val, 0, past.0, x.0, best.0);
                    bf = (self.e).emit(Op::Select, Rep::F64, 0, past.0, xf.0, bf.0);
                }
                best
            }
            Intrinsic::StrLen if nargs == 1 && one => {
                self.pin(f, callee);
                let p = (self.e).check_type(arg(self, 0), TypeSet::STR, ExitTag::Type, s);
                let n = self.e.load(Field::StrLen, p, 0);
                let n = self.e.op1(Op::Trunc32, Rep::I32, n);
                self.e.op1(Op::BoxI32, Rep::Val, n)
            }
            Intrinsic::Builtin(h @ (Helper::StrByte | Helper::StrSub)) if one => {
                let sub = h == Helper::StrSub;
                let (min, max) = if sub { (2, 3) } else { (1, 2) };
                if !(min..=max).contains(&nargs) {
                    return false;
                }
                self.pin(f, callee);
                let p = (self.e).check_type(arg(self, 0), TypeSet::STR, ExitTag::Type, s);
                // `sub`'s `j` defaults to -1, `byte`'s `i` to 1.
                let mut pos = vec![p];
                for k in 1..max {
                    pos.push(if k < nargs {
                        (self.e).check_type(arg(self, k), TypeSet::SMALL, ExitTag::Type, s)
                    } else {
                        (self.e).k32(Rep::I32, if sub { -1i32 as u32 } else { 1 })
                    });
                }
                let r = self.e.helped(h, 0, &pos, s);
                if sub {
                    self.gc_after(pc, a, c, r);
                    return true;
                }
                r
            }
            Intrinsic::Builtin(h) if one => {
                let arity = match h {
                    Helper::SetMetatable | Helper::RawGet | Helper::RawEqual | Helper::TInsert => 2,
                    Helper::RawSet => 3,
                    _ => 1,
                };
                if nargs != arity {
                    return false;
                }
                self.pin(f, callee);
                let args: Vec<Val> = (0..arity).map(|k| arg(self, k)).collect();
                let total = matches!(
                    h,
                    Helper::Type | Helper::GetMetatable | Helper::RawEqual | Helper::ToNumber
                );
                let r = match total {
                    true => self.e.call(h, 0, &args),
                    false => self.e.helped(h, 0, &args, s),
                };
                let r = if h == Helper::RawEqual {
                    let zero = self.e.k32(Rep::I32, 0);
                    let eq = self.e.cmp(Op::Cmp, Cc::Ne, r, zero);
                    let (yes, no) = (
                        self.konst(Value::boolean(true)),
                        self.konst(Value::boolean(false)),
                    );
                    (self.e).emit(Op::Select, Rep::Val, 0, eq.0, yes.0, no.0)
                } else {
                    r
                };
                // They allocate; and `rawset` may change a class or a
                // watched holder, `setmetatable` a holder's shape.
                let writes = matches!(
                    h,
                    Helper::RawSet | Helper::SetMetatable | Helper::TInsert | Helper::TRemove
                );
                if writes || h == Helper::ToString {
                    self.gc_after(pc, a, c, r);
                    if writes {
                        self.lazy_exit(pc + 1);
                    }
                    return true;
                }
                r
            }
            Intrinsic::Sqrt | Intrinsic::Unary(_) if nargs == 1 && one => {
                self.pin(f, callee);
                let x = self.float_arg(arg(self, 0));
                let r = match intr {
                    Intrinsic::Unary(h) => self.e.call(h, 0, &[x]),
                    _ => self.e.op1(Op::FSqrt, Rep::F64, x),
                };
                self.e.op1(Op::BoxF64, Rep::Val, r)
            }
            Intrinsic::Atan if (nargs == 1 || nargs == 2) && one => {
                self.pin(f, callee);
                let y = self.float_arg(arg(self, 0));
                let x = match nargs {
                    2 => self.float_arg(arg(self, 1)),
                    _ => self.e.kf64(1.0),
                };
                let r = self.e.call(Helper::Atan2, 0, &[y, x]);
                self.e.op1(Op::BoxF64, Rep::Val, r)
            }
            Intrinsic::Log if nargs == 1 && one => {
                self.pin(f, callee);
                let x = self.float_arg(arg(self, 0));
                let r = self.e.call(Helper::Log, 0, &[x]);
                self.e.op1(Op::BoxF64, Rep::Val, r)
            }
            // An integer is its own; a float's that leaves i32 is the
            // builtin's from then on.
            Intrinsic::Floor | Intrinsic::Ceil if nargs == 1 && one && bits & fb::OVERFLOW == 0 => {
                self.pin(f, callee);
                let op = if intr == Intrinsic::Floor {
                    Op::FFloor
                } else {
                    Op::FCeil
                };
                let x = arg(self, 0);
                self.by_kind(
                    x,
                    |_, _| x,
                    |b, f| {
                        let r = b.e.op1(op, Rep::F64, f);
                        let tag = ExitTag::Overflow as u16;
                        let i = b.e.emit(Op::FToIExact, Rep::I32, tag, r.0, s.0, 0);
                        b.e.op1(Op::BoxI32, Rep::Val, i)
                    },
                )
            }
            Intrinsic::Abs if nargs == 1 && one && bits & fb::OVERFLOW == 0 => {
                self.pin(f, callee);
                let x = arg(self, 0);
                self.by_kind(
                    x,
                    |b, i| {
                        let tag = ExitTag::Overflow as u16;
                        let r = b.e.emit(Op::AbsOvf, Rep::I32, tag, i.0, s.0, 0);
                        b.e.op1(Op::BoxI32, Rep::Val, r)
                    },
                    |b, f| {
                        let r = b.e.op1(Op::FAbs, Rep::F64, f);
                        b.e.op1(Op::BoxF64, Rep::Val, r)
                    },
                )
            }
            _ => return false,
        };
        if c == 2 {
            self.set(a, r);
        }
        true
    }

    /// The result of a builtin that allocated, then the collector's check
    /// past the call.
    fn gc_after(&mut self, pc: u32, a: u8, c: u8, r: Val) {
        if c == 2 {
            self.set(a, r);
        }
        self.gc_check(pc);
    }
}
