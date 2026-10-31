//! Builtins compiled inline (`jit-design.md` 12): a CALL whose callee is,
//! at compile time, a builtin with an intrinsic guards the callee by
//! identity and computes the result in place; anything the inline code does
//! not cover exits before the CALL, and the interpreter runs the builtin.

use crate::dmm::Gc;
use crate::env::function::{Function, LuaFn, NativeFn, NativeKind, Prototype};
use crate::env::string::LuaString;
use crate::env::value::Value;
use crate::instruction::Instruction;
use crate::jit::build::Builder;
use crate::jit::build::emit::{arm_end, branch_arm, enter_join};
use crate::jit::feedback as fb;
use crate::jit::ir::Block;
use crate::jit::ir::Val;
use crate::jit::ir::ops::{Cc, ExitTag, HELPER_FAIL_VAL, HelperId, Op};
use crate::jit::ir::types::{Rep, TypeSet};
use crate::vm::abi::Handler;
use crate::vm::ff;
use crate::vm::native::ContFn;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Intrinsic {
    Sqrt,
    Sin,
    Cos,
    Floor,
    Ceil,
    Abs,
    Assert,
    Error,
    Pairs,
    Ipairs,
    Select,
    /// A builtin computed by its helper.
    Builtin(HelperId),
}

/// The intrinsic of a builtin, known by its fast entry (each builtin with
/// one has its own), else by its function.
fn intrinsic_of(f: Function<'_>) -> Option<Intrinsic> {
    use crate::builtin::basic as base;
    let nc = f.as_native()?;
    let plain: [(NativeFn, Intrinsic); 13] = [
        (base::lua_type, Intrinsic::Builtin(HelperId::Type)),
        (base::lua_error, Intrinsic::Error),
        (
            base::lua_getmetatable,
            Intrinsic::Builtin(HelperId::GetMetatable),
        ),
        (base::lua_rawget, Intrinsic::Builtin(HelperId::RawGet)),
        (base::lua_rawset, Intrinsic::Builtin(HelperId::RawSet)),
        (base::lua_rawequal, Intrinsic::Builtin(HelperId::RawEqual)),
        (base::lua_rawlen, Intrinsic::Builtin(HelperId::RawLen)),
        (base::lua_tonumber, Intrinsic::Builtin(HelperId::ToNumber)),
        (base::lua_select, Intrinsic::Select),
        (
            crate::builtin::string::lua_len,
            Intrinsic::Builtin(HelperId::StrLen),
        ),
        (
            crate::builtin::string::lua_byte,
            Intrinsic::Builtin(HelperId::StrByte),
        ),
        (
            crate::builtin::math::lua_min,
            Intrinsic::Builtin(HelperId::Min),
        ),
        (
            crate::builtin::math::lua_max,
            Intrinsic::Builtin(HelperId::Max),
        ),
    ];
    match nc.function {
        NativeKind::Plain(p) => {
            if let Some(&(_, i)) = plain.iter().find(|(f, _)| *f as usize == p as usize) {
                return Some(i);
            }
        }
        NativeKind::Cont(p) => {
            let cont: [(ContFn, HelperId); 3] = [
                (base::lua_tostring, HelperId::ToString),
                (crate::builtin::table::lua_insert, HelperId::TInsert),
                (crate::builtin::table::lua_remove, HelperId::TRemove),
            ];
            if let Some(&(_, h)) = cont.iter().find(|(f, _)| *f as usize == p as usize) {
                return Some(Intrinsic::Builtin(h));
            }
        }
        _ => {}
    }
    let entry = nc.entry as usize;
    let table: [(Handler, Intrinsic); 11] = [
        (ff::ff_sqrt, Intrinsic::Sqrt),
        (ff::ff_sin, Intrinsic::Sin),
        (ff::ff_cos, Intrinsic::Cos),
        (ff::ff_floor, Intrinsic::Floor),
        (ff::ff_ceil, Intrinsic::Ceil),
        (ff::ff_abs, Intrinsic::Abs),
        (ff::ff_assert, Intrinsic::Assert),
        (crate::vm::native::ff_pairs, Intrinsic::Pairs),
        (crate::vm::native::ff_ipairs, Intrinsic::Ipairs),
        (
            ff::ff_setmetatable,
            Intrinsic::Builtin(HelperId::SetMetatable),
        ),
        (ff::ff_sub, Intrinsic::Builtin(HelperId::StrSub)),
    ];
    table
        .iter()
        .find(|(h, _)| *h as usize == entry)
        .map(|&(_, i)| i)
}

/// The value `v` has now, when it is a load the builder can redo: a
/// constant, an upvalue of the closure being compiled, a field read of such
/// a table, or a join of one value. With `guess`, a loop header's value of a
/// register is taken to be the loop entry's, and a load of the entry's frame
/// its value there. The region guards what it uses of it.
pub(super) fn value_now<'gc>(b: &Builder<'_, 'gc>, v: Val, guess: bool) -> Option<Value<'gc>> {
    let Some(i) = b.f.def_inst(v) else {
        return match b.param_source(v)? {
            Ok(a) => value_now(b, a, guess),
            Err(r) if guess => b.entry_frame.get(r as usize).copied(),
            Err(_) => None,
        };
    };
    match b.f.op(i) {
        Op::Load(r) if guess && b.f.def_block(v) == Some(b.f.entry) => {
            b.entry_frame.get(r as usize).copied()
        }
        Op::KObj(p) => Some(b.f.pool[p as usize]),
        Op::Guard(_) => value_now(b, b.f.args(i)[0], guess),
        Op::UpvalValue(k) => Some(unsafe { (*b.closure.upvalue_ptr().add(k as usize)).value }),
        Op::CellLoad => {
            let c = b.f.def_inst(b.f.args(i)[0])?;
            let Op::UpvalCell(k) = b.f.op(c) else {
                return None;
            };
            Some(unsafe { (*b.closure.upvalue_ptr().add(k as usize)).cell }.get())
        }
        _ => {
            let &(recv, key) = b.field_loads.get(&v)?;
            Some(value_now(b, recv, guess)?.get_table()?.raw_get(key))
        }
    }
}

/// The function the call at `pc` calls when the builder may take it for a
/// constant, which its user guards by identity: what `callee` holds now
/// (`value_now`), else the one function the site has called (its callee
/// word, I2). Neither once such a guard failed.
pub(super) fn known_callee<'gc>(
    b: &Builder<'_, 'gc>,
    pc: u32,
    callee: Val,
) -> Option<Function<'gc>> {
    if b.feedback(pc) & fb::CONST_MISS != 0 {
        return value_now(b, callee, false).and_then(|v| v.get_function());
    }
    if let Some(f) = value_now(b, callee, true).and_then(|v| v.get_function()) {
        return Some(f);
    }
    let seen = b.proto.feedback.callee(pc as usize)?.get();
    // Bit 0: megamorphic, or a prototype's (`proto_slot`).
    if seen == 0 || seen & 1 != 0 {
        return None;
    }
    unsafe { Value::from_raw(seen) }.get_function()
}

/// What the call at `pc` may take its callee for (`callee_of`).
pub(super) enum Callee<'gc> {
    Unknown,
    /// The callee value is this constant.
    Const(Function<'gc>),
    /// This function, guarded by identity.
    Same(Function<'gc>),
    /// A closure without varargs of this prototype, guarded as such: one
    /// that made several closures (a function's local functions are new
    /// ones on each call), or that a site's slot names (`proto_slot`).
    /// With a closure of it `known_callee` names.
    Proto(Gc<'gc, Prototype<'gc>>, Option<LuaFn<'gc>>),
}

/// The callee of the call at `pc` as the region may take it: a constant;
/// what `known_callee` names, guarded by identity unless a Lua closure of a
/// prototype that made several closures; else the prototype the site's slot
/// names. Nothing once such a guard failed here, even for a value of the
/// compiled closure's own upvalues, which may differ per closure.
pub(super) fn callee_of<'gc>(b: &Builder<'_, 'gc>, pc: u32, callee: Val) -> Callee<'gc> {
    use crate::env::feedback::slot_proto;
    if let Some(Op::KObj(k)) = b.f.def_op(callee) {
        return match b.f.pool[k as usize].get_function() {
            Some(f) => Callee::Const(f),
            None => Callee::Unknown,
        };
    }
    if b.feedback(pc) & fb::CONST_MISS != 0 {
        return Callee::Unknown;
    }
    match known_callee(b, pc, callee) {
        Some(f) => match f.as_lua() {
            // The generic call tests such a callee anyway.
            Some(c) if c.proto.is_vararg => Callee::Unknown,
            Some(c) if c.proto.closures.get() > 1 => Callee::Proto(c.proto, Some(c)),
            _ => Callee::Same(f),
        },
        None => {
            let slot = b.proto.feedback.callee(pc as usize);
            match slot.and_then(|s| slot_proto(s.get())) {
                Some(p) if !p.is_vararg => Callee::Proto(p, None),
                _ => Callee::Unknown,
            }
        }
    }
}

/// What `emit` made of a CALL.
pub(super) enum Emitted {
    /// Nothing: the CALL is a call.
    No,
    /// Its results.
    Done,
    /// Its results, and a block for the call itself, where the inline code
    /// found the builtin must run, with the builtin as a constant.
    Fallback(Block, Val),
}

/// Compile the CALL at `pc` as its callee's intrinsic, if it has one that
/// covers the call's shape.
pub(super) fn emit(b: &mut Builder<'_, '_>, pc: u32, insn: Instruction, callee: Val) -> Emitted {
    // A tail call returns the one result: not for a builtin that may
    // return more.
    let tail = insn.op() == crate::instruction::Op::TAILCALL;
    let (a, nb, c) = if tail {
        (insn.a(), insn.b(), 2)
    } else {
        (insn.a(), insn.b(), insn.c())
    };
    let Some(f) = known_callee(b, pc, callee) else {
        return Emitted::No;
    };
    let Some(intr) = intrinsic_of(f) else {
        return Emitted::No;
    };
    match intr {
        Intrinsic::Select => return select(b, pc, a, nb, c, f, callee, tail),
        Intrinsic::Assert if nb != 0 && c != 0 && !(tail && nb > 2) => {
            return assert(b, a, nb - 1, c, f, callee);
        }
        Intrinsic::Assert => return Emitted::No,
        Intrinsic::Error => {
            // The interpreter's builtin raises; nothing follows.
            pin(b, f, callee);
            b.deopt(ExitTag::Slow);
            return Emitted::Done;
        }
        Intrinsic::Builtin(h) if nb != 0 => return builtin(b, h, a, nb - 1, c, f, callee),
        _ if nb == 0 => return Emitted::No,
        _ => {}
    }
    emit_math(b, pc, intr, a, nb - 1, c, f, callee)
}

/// Guard `callee` to be the builtin `f`; the constant.
fn pin<'gc>(b: &mut Builder<'_, 'gc>, f: Function<'gc>, callee: Val) -> Val {
    let k = b.konst(Value::function(f));
    let cv = b.boxed(callee);
    b.ins_exit(Op::GuardSame, &[cv, k], ExitTag::Const);
    k
}

/// `assert(v, ...)`: its `nargs` arguments, a falsy `v` an exit, where the
/// interpreter's builtin raises (LuaJIT's `recff_assert`).
fn assert<'gc>(
    b: &mut Builder<'_, 'gc>,
    a: u8,
    nargs: u8,
    c: u8,
    f: Function<'gc>,
    callee: Val,
) -> Emitted {
    pin(b, f, callee);
    let x = b.reg(a + 4);
    let xv = b.boxed(x);
    let falsy = b.ins(Op::IsFalsy, &[xv]);
    b.ins_exit(Op::GuardFalse, &[falsy], ExitTag::Slow);
    for k in 0..c - 1 {
        let v = if k < nargs {
            b.reg(a + 4 + k)
        } else {
            b.konst(Value::nil())
        };
        b.set(a + k, v);
    }
    Emitted::Done
}

/// A builtin by its helper; `type` of a known kind is a constant.
fn builtin<'gc>(
    b: &mut Builder<'_, 'gc>,
    h: HelperId,
    a: u8,
    nargs: u8,
    c: u8,
    f: Function<'gc>,
    callee: Val,
) -> Emitted {
    // The helper's operand count, and how many of them the call may leave
    // out (passed as nil).
    let (arity, optional) = match h {
        HelperId::SetMetatable | HelperId::RawGet | HelperId::RawEqual => (2, 0),
        HelperId::RawSet => (3, 0),
        HelperId::StrSub => (3, 1),
        HelperId::StrByte => (2, 1),
        HelperId::TInsert | HelperId::Min | HelperId::Max => (2, 0),
        _ => (1, 0),
    };
    if nargs > arity || nargs + optional < arity || c == 0 || c > 2 {
        return Emitted::No;
    }
    let k = b.konst(Value::function(f));
    let cv = b.boxed(callee);
    b.ins_exit(Op::GuardSame, &[cv, k], ExitTag::Const);
    let args: Vec<Val> = (0..arity)
        .map(|i| {
            if i < nargs {
                let v = b.reg(a + 4 + i);
                b.boxed(v)
            } else {
                b.konst(Value::nil())
            }
        })
        .collect();
    if matches!(h, HelperId::Min | HelperId::Max)
        && let Some(r) = min_max_inline(b, h == HelperId::Max, args[0], args[1])
    {
        if c == 2 {
            b.set(a, r);
        }
        return Emitted::Done;
    }
    if h == HelperId::Type
        && let Some(name) = type_name(b.ty(args[0]).set)
    {
        if c == 2 {
            let s = b.konst(Value::string(LuaString::new(b.ctx, name.as_bytes())));
            b.set(a, s);
        }
        return Emitted::Done;
    }
    let r = b.ins(Op::Helper(h), &args);
    let mut out = Emitted::Done;
    if !matches!(
        h,
        HelperId::Type | HelperId::GetMetatable | HelperId::RawEqual | HelperId::ToNumber
    ) {
        let fail = b.ins(Op::KVal(HELPER_FAIL_VAL), &[]);
        let failed = b.ins(Op::SameBits, &[r, fail]);
        let (slow, ok) = (b.new_sealed_succ(), b.new_sealed_succ());
        b.br(failed, slow, ok);
        b.switch_to(ok);
        out = Emitted::Fallback(slow, k);
    }
    if c == 2 {
        let r = if h == HelperId::RawEqual {
            let zero = b.ki32(0);
            let t = b.ins(Op::ICmp(Cc::Ne), &[r, zero]);
            b.ins(Op::Box, &[t])
        } else {
            r
        };
        b.set(a, r);
    }
    out
}

/// `math.max(x, y)` (or `min`) of two small integers or two floats: the
/// builtin's pick by `<`, which keeps `x` for a NaN.
fn min_max_inline(b: &mut Builder<'_, '_>, max: bool, x: Val, y: Val) -> Option<Val> {
    let (tx, ty) = (b.ty(x), b.ty(y));
    let is = |t: crate::jit::ir::types::Ty, rep, set| {
        t.rep == rep || (!t.set.is_empty() && t.within(set))
    };
    let (x, y, cmp) = if is(tx, Rep::I32, TypeSet::SMALL) && is(ty, Rep::I32, TypeSet::SMALL) {
        (b.as_i32(x), b.as_i32(y), Op::ICmp(Cc::Lt))
    } else if is(tx, Rep::F64, TypeSet::FLOAT) && is(ty, Rep::F64, TypeSet::FLOAT) {
        (b.as_f64(x), b.as_f64(y), Op::FCmp(Cc::Lt))
    } else {
        return None;
    };
    let lt = if max {
        b.ins(cmp, &[x, y])
    } else {
        b.ins(cmp, &[y, x])
    };
    Some(b.ins(Op::Select, &[lt, y, x]))
}

/// What `type` answers for values of `set`, when they share it.
fn type_name(set: TypeSet) -> Option<&'static str> {
    let names = [
        (TypeSet::NIL, "nil"),
        (TypeSet::BOOL, "boolean"),
        (TypeSet::NUM, "number"),
        (TypeSet::STR, "string"),
        (TypeSet::TAB, "table"),
        (TypeSet::FUN, "function"),
        (TypeSet::THR, "thread"),
        (TypeSet::UDATA, "userdata"),
    ];
    if set.is_empty() {
        return None;
    }
    names.iter().find(|(s, _)| s.contains(set)).map(|&(_, n)| n)
}

/// `select('#', ...)` and `select(k, ...)` with a constant selector and one
/// result, from the arguments, or from the frame's extras when they are a
/// `VARARG` up to `top` just before; a `tail` call only of the count, the
/// one select that returns one value.
#[allow(clippy::too_many_arguments)]
fn select<'gc>(
    b: &mut Builder<'_, 'gc>,
    pc: u32,
    a: u8,
    nb: u8,
    c: u8,
    f: Function<'gc>,
    callee: Val,
    tail: bool,
) -> Emitted {
    if c != 2 || nb == 1 {
        return Emitted::No;
    }
    let sel = match b.callee_hint {
        Some((at, _, sel)) if at == pc => sel,
        _ => b.reg(a + 4),
    };
    let Some(sv) = const_value(b, sel) else {
        return Emitted::No;
    };
    let count = sv.get_string().is_some_and(|s| s.as_bytes() == b"#");
    let k = sv.get_integer();
    if !count && (k.is_none() || tail) {
        return Emitted::No;
    }
    let extras = nb == 0;
    if extras {
        let p = pc.checked_sub(1).map(|p| b.code(p));
        let vararg = p.is_some_and(|p| {
            p.op() == crate::instruction::Op::VARARG && p.a() == a + 5 && p.b() == 0
        });
        let same_block = vararg && b.cfg.block_of[pc as usize] == b.cfg.block_of[pc as usize - 1];
        if !same_block || (!count && !k.is_some_and(|k| (1..=256).contains(&k))) {
            return Emitted::No;
        }
    }
    let m = nb as i64 - 2;
    let pos = k.map(|k| if k < 0 { m + k + 1 } else { k });
    if !extras && !count && pos.is_none_or(|p| p < 1) {
        // An index out of range raises in the builtin.
        return Emitted::No;
    }
    let kf = b.konst(Value::function(f));
    let cv = b.boxed(callee);
    b.ins_exit(Op::GuardSame, &[cv, kf], ExitTag::Const);
    let r = match (count, extras) {
        (true, false) => b.konst(Value::small(m as i32)),
        (true, true) => {
            let nv = b.ins(Op::NumVarargs, &[]);
            b.ins(Op::Box, &[nv])
        }
        (false, false) => {
            let p = pos.unwrap();
            if p > m {
                b.konst(Value::nil())
            } else {
                b.reg(a + 4 + p as u8)
            }
        }
        (false, true) => {
            let nv = b.ins(Op::NumVarargs, &[]);
            b.ins(Op::VarargAt((k.unwrap() - 1) as u8), &[nv])
        }
    };
    b.set(a, r);
    Emitted::Done
}

/// The value of a constant.
fn const_value<'gc>(b: &Builder<'_, 'gc>, v: Val) -> Option<Value<'gc>> {
    match b.f.def_op(v)? {
        Op::KVal(bits) => Some(unsafe { Value::from_raw(bits) }),
        Op::KObj(p) => Some(b.f.pool[p as usize]),
        _ => None,
    }
}

/// The math intrinsics, `pairs` and `ipairs`. An argument their
/// inline code does not take goes to the builtin (I1), but where an
/// operation's own exit stops it (an overflow, a result outside i32).
#[allow(clippy::too_many_arguments)]
fn emit_math<'gc>(
    b: &mut Builder<'_, 'gc>,
    pc: u32,
    intr: Intrinsic,
    a: u8,
    nargs: u8,
    c: u8,
    f: Function<'gc>,
    callee: Val,
) -> Emitted {
    let insn = b.code(pc);
    let iterators = matches!(intr, Intrinsic::Pairs | Intrinsic::Ipairs);
    // An exit saw an argument the inline code does not take.
    let seen = b.feedback(pc) & fb::KINDS;
    if !iterators && seen & !(fb::SMALL | fb::BIGINT | fb::FLOAT) != 0 {
        return Emitted::No;
    }
    // A tail call or MULTRET wants all results; the iterators make up to
    // four, the others one.
    let tail = insn.op() == crate::instruction::Op::TAILCALL;
    if c == 0 || (c > 2 && !iterators) || (iterators && (tail || c > 5)) || nargs != 1 {
        return Emitted::No;
    }
    let k = pin(b, f, callee);
    let x = b.reg(a + 4);
    // Where the builtin runs instead.
    let mut slow = None;
    if iterators {
        let x = b.boxed(x);
        let (iter, control) = if intr == Intrinsic::Pairs {
            // A table without `__pairs` (`ff_pairs`).
            let t = b.guard(x, TypeSet::TAB);
            let mm = b.ins(
                Op::Helper(HelperId::Mm(crate::env::MetamethodBits::PAIRS.bits())),
                &[t, t],
            );
            let none = b.ins(Op::IsType(TypeSet::NIL), &[mm]);
            slow = Some(fallback(b, none, true));
            (b.ctx.next_fn(), Value::nil())
        } else {
            (b.ctx.ipairs_iter(), Value::small(0))
        };
        let results = [
            b.konst(Value::function(iter)),
            x,
            b.konst(control),
            b.konst(Value::nil()),
        ];
        for (k, &r) in results.iter().take(c as usize - 1).enumerate() {
            b.set(a + k as u8, r);
        }
        return done(slow, k);
    }
    let r = match intr {
        Intrinsic::Sqrt | Intrinsic::Sin | Intrinsic::Cos => {
            let x = number_f64(b, x, seen);
            match intr {
                Intrinsic::Sqrt => b.ins(Op::FSqrt, &[x]),
                Intrinsic::Sin => b.ins(Op::Helper(HelperId::Sin), &[x]),
                _ => b.ins(Op::Helper(HelperId::Cos), &[x]),
            }
        }
        Intrinsic::Floor | Intrinsic::Ceil => {
            let ty = b.ty(x);
            let int = ty.rep == Rep::I32
                || ty.rep == Rep::I64
                || (!ty.set.is_empty() && ty.within(TypeSet::INT));
            if int {
                // An integer is its own floor.
                x
            } else {
                let x = number_f64(b, x, seen);
                let op = if intr == Intrinsic::Floor {
                    Op::FFloor
                } else {
                    Op::FCeil
                };
                let y = b.ins(op, &[x]);
                // Outside i32 the builtin makes a big integer or a float.
                let i = b.ins_exit(Op::FToIExact, &[y], ExitTag::Slow);
                b.f.result(i)
            }
        }
        Intrinsic::Abs => {
            let ty = b.ty(x);
            let known = |set| !ty.set.is_empty() && ty.within(set);
            if ty.rep == Rep::F64 || known(TypeSet::FLOAT) {
                let x = b.as_f64(x);
                b.ins(Op::FAbs, &[x])
            } else if ty.rep == Rep::I32 || known(TypeSet::SMALL) {
                let x = b.as_i32(x);
                abs_i32(b, x)
            } else {
                // A float, a small integer, else the builtin.
                let xv = b.boxed(x);
                let join = b.new_block();
                let is_float = b.ins(Op::IsType(TypeSet::FLOAT), &[xv]);
                let rest = branch_arm(b, is_float);
                let f = b.ins(Op::Unbox(Rep::F64), &[xv]);
                let r = b.ins(Op::FAbs, &[f]);
                arm_end(b, join, a, r);
                b.switch_to(rest);
                let is_small = b.ins(Op::IsType(TypeSet::SMALL), &[xv]);
                let other = branch_arm(b, is_small);
                let i = b.ins(Op::Unbox(Rep::I32), &[xv]);
                let r = abs_i32(b, i);
                arm_end(b, join, a, r);
                enter_join(b, join);
                return Emitted::Fallback(other, k);
            }
        }
        Intrinsic::Pairs | Intrinsic::Ipairs => unreachable!("returned above"),
        Intrinsic::Select | Intrinsic::Builtin(_) | Intrinsic::Assert | Intrinsic::Error => {
            unreachable!("not math")
        }
    };
    if c == 2 {
        b.set(a, r);
    }
    done(slow, k)
}

/// Go on where `cond` is `holds`, the returned block taking the other way.
fn fallback(b: &mut Builder<'_, '_>, cond: Val, holds: bool) -> Block {
    let (slow, ok) = (b.new_sealed_succ(), b.new_sealed_succ());
    if holds {
        b.br(cond, ok, slow);
    } else {
        b.br(cond, slow, ok);
    }
    b.switch_to(ok);
    slow
}

fn done(slow: Option<Block>, k: Val) -> Emitted {
    match slow {
        Some(s) => Emitted::Fallback(s, k),
        None => Emitted::Done,
    }
}

/// `|x|` of an i32; `abs(i32::MIN)` leaves i32 and exits to the builtin.
fn abs_i32(b: &mut Builder<'_, '_>, x: Val) -> Val {
    let i = b.ins_exit(Op::INeg, &[x], ExitTag::Slow);
    let neg = b.f.result(i);
    let zero = b.ki32(0);
    let below = b.ins(Op::ICmp(Cc::Lt), &[x, zero]);
    b.ins(Op::Select, &[below, neg, x])
}

/// The f64 of a number argument as the builtin converts it (an integer
/// rounds): typed ones converted, else a small integer or a float, or an
/// integer where the site's feedback (`seen`) saw big ones and no floats;
/// anything else exits to the builtin, for good where it saw both.
fn number_f64(b: &mut Builder<'_, '_>, x: Val, seen: fb::Bits) -> Val {
    let ty = b.ty(x);
    let known = |set| !ty.set.is_empty() && ty.within(set);
    if matches!(ty.rep, Rep::F64 | Rep::I32 | Rep::I64)
        || known(TypeSet::FLOAT)
        || known(TypeSet::SMALL)
    {
        return b.num_f64(x);
    }
    if known(TypeSet::INT) || (seen & fb::BIGINT != 0 && seen & fb::FLOAT == 0) {
        let i = b.as_i64(x);
        return b.ins(Op::LToF, &[i]);
    }
    let xv = b.boxed(x);
    let tag = if seen & fb::BIGINT == 0 {
        ExitTag::Type
    } else {
        ExitTag::Slow
    };
    let i = b.ins_exit(Op::ToF64, &[xv], tag);
    b.f.result(i)
}
