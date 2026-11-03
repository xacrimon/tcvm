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
use crate::jit::build::emit::{all_in, arm_end, branch_arm, const_f64, enter_join};
use crate::jit::feedback as fb;
use crate::jit::ir::Block;
use crate::jit::ir::Val;
use crate::jit::ir::ops::{Cc, ExitTag, HELPER_FAIL_VAL, HelperId, Libm, Op};
use crate::jit::ir::types::{Rep, TypeSet};
use crate::vm::abi::Handler;
use crate::vm::ff;
use crate::vm::native::ContFn;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Intrinsic {
    Sqrt,
    /// A one-argument libm function.
    Unary(Libm),
    /// `atan(y [, x])`: `atan2`.
    Atan,
    /// `log(x [, base])`.
    Log,
    /// `fmod(x, y)` with a float `x`.
    FMod,
    Floor,
    Ceil,
    Abs,
    Assert,
    Error,
    Pairs,
    Ipairs,
    Select,
    /// `string.len(s)`.
    StrLen,
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
            Intrinsic::StrLen,
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
    let table: [(Handler, Intrinsic); 18] = [
        (ff::ff_sqrt, Intrinsic::Sqrt),
        (ff::ff_sin, Intrinsic::Unary(Libm::Sin)),
        (ff::ff_cos, Intrinsic::Unary(Libm::Cos)),
        (ff::ff_tan, Intrinsic::Unary(Libm::Tan)),
        (ff::ff_asin, Intrinsic::Unary(Libm::Asin)),
        (ff::ff_acos, Intrinsic::Unary(Libm::Acos)),
        (ff::ff_exp, Intrinsic::Unary(Libm::Exp)),
        (ff::ff_atan, Intrinsic::Atan),
        (ff::ff_log, Intrinsic::Log),
        (ff::ff_fmod, Intrinsic::FMod),
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
        Op::Guard(_) | Op::Refine(_) => value_now(b, b.f.args(i)[0], guess),
        Op::UpvalValue(k) if !b.region.proto.feedback.upvalue_varies(k as usize) => {
            Some(unsafe { (*b.region.upvalue_ptr().add(k as usize)).value })
        }
        Op::CellLoad => {
            let c = b.f.def_inst(b.f.args(i)[0])?;
            let Op::UpvalCell(k) = b.f.op(c) else {
                return None;
            };
            Some(unsafe { (*b.region.upvalue_ptr().add(k as usize)).cell }.get())
        }
        _ => {
            let &(recv, key) = b.known.field_loads.get(&v)?;
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
    /// A closure a `CLOSURE` of this compile made (W3 §10): no guard.
    Made(Gc<'gc, Prototype<'gc>>),
}

/// The callee of the call at `pc` as the region may take it: a constant;
/// what `known_callee` names, guarded by identity unless a Lua closure of a
/// prototype that made several closures; else the prototype the site's slot
/// names. Nothing once such a guard failed here, even for a value of the
/// compiled closure's own upvalues, which may differ per closure.
pub(super) fn callee_of<'gc>(b: &Builder<'_, 'gc>, pc: u32, callee: Val) -> Callee<'gc> {
    use crate::env::feedback::slot_proto;
    if let Some(&p) = b.known.made.get(&b.root_of(callee)) {
        return Callee::Made(p);
    }
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
            // An upvalue that differs between closures, holding a closure of
            // a prototype that made several: guarded by that prototype, its
            // value now still names the code to inline.
            if let Some(c) = varying_closure(b, callee)
                && c.proto.closures.get() > 1
                && !c.proto.is_vararg
            {
                return Callee::Proto(c.proto, Some(c));
            }
            let slot = b.proto.feedback.callee(pc as usize);
            match slot.and_then(|s| slot_proto(s.get())) {
                Some(p) if !p.is_vararg => Callee::Proto(p, None),
                _ => Callee::Unknown,
            }
        }
    }
}

/// The Lua closure `v` holds now when it is a load of an upvalue that
/// differs between closures, which `value_now` does not take as known.
fn varying_closure<'gc>(b: &Builder<'_, 'gc>, v: Val) -> Option<LuaFn<'gc>> {
    let i = b.f.def_inst(v)?;
    let Op::UpvalValue(k) = b.f.op(i) else {
        return None;
    };
    b.region
        .proto
        .feedback
        .upvalue_varies(k as usize)
        .then_some(())?;
    unsafe { (*b.region.upvalue_ptr().add(k as usize)).value }
        .get_function()?
        .as_lua()
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

/// Whether the callee the CALL at `pc` is known to call has an intrinsic
/// ([`emit`] may still decline the call's shape).
pub(super) fn has_intrinsic(b: &Builder<'_, '_>, pc: u32, callee: Val) -> bool {
    known_callee(b, pc, callee).and_then(intrinsic_of).is_some()
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
        Intrinsic::Builtin(h @ (HelperId::Min | HelperId::Max)) if nb != 0 => {
            return min_max(b, pc, h, a, nb - 1, c, f, callee);
        }
        Intrinsic::StrLen | Intrinsic::Builtin(HelperId::StrSub | HelperId::StrByte) if nb != 0 => {
            return string(b, pc, intr, a, nb - 1, c, f, callee);
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
    if h == HelperId::RawLen && TypeSet::STR.contains(b.ty(args[0]).set) && !b.ty(args[0]).set.is_empty() {
        if c == 2 {
            let p = b.ins(Op::Unbox(Rep::Ptr), &[args[0]]);
            let n = b.ins(Op::StrLenOf, &[p]);
            b.set(a, n);
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
    // A tail call's after-snapshot would resume at the `RETURN` after it.
    let class = (h == HelperId::SetMetatable
        && c == 2
        && b.code(b.pc).op() != crate::instruction::Op::TAILCALL)
        .then(|| moved_shape(b, &args))
        .flatten();
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
        b.moved_shape = class;
    }
    // `rawset` may change a class (`MtCache::member_write`) or a watched
    // holder, `setmetatable` a holder's shape or adopt one; a `TAILCALL` only
    // returns after them.
    if matches!(h, HelperId::RawSet | HelperId::SetMetatable)
        && b.code(b.pc).op() != crate::instruction::Op::TAILCALL
    {
        b.flush();
        b.lazy_exit_after();
    }
    out
}

/// The shape `setmetatable(t, mt)` gives `t`, known before the call, when
/// the constant `mt`'s last move took `t`'s shape there: the result is
/// guarded to have it after the call (`emit_call`), and its class's
/// metamethods are then constants (W3 §9).
fn moved_shape(b: &mut Builder<'_, '_>, args: &[Val]) -> Option<u32> {
    let Some(Op::KObj(k)) = b.f.def_op(args[1]) else {
        return None;
    };
    let (from, to) = b.f.pool[k as usize].get_table()?.shape().mt_move()?;
    let from = b.intern_shape(from);
    (b.known_shapes(args[0]) == Some(crate::jit::build::Shapes::one(from)))
        .then(|| b.intern_shape(to))
}

/// `min(...)`/`max(...)` of `nargs` arguments by the kinds of them all,
/// folded as the builtin's loop picks: the next argument when below the pick
/// (`max`: the pick below it), so the first of equals stays and a NaN follows
/// `<`. Integers compare as i32 or, with a big one, i64; floats as floats;
/// small integers with floats as floats (exact for i32), picking the
/// original values; strings by their bytes. Big integers with floats take
/// the exact helper (two arguments; more call the builtin), as do kinds
/// `<` raises on or runs a metamethod for.
#[allow(clippy::too_many_arguments)]
fn min_max<'gc>(
    b: &mut Builder<'_, 'gc>,
    pc: u32,
    h: HelperId,
    a: u8,
    nargs: u8,
    c: u8,
    f: Function<'gc>,
    callee: Val,
) -> Emitted {
    if nargs == 0 || c == 0 || c > 2 {
        return Emitted::No;
    }
    let max = h == HelperId::Max;
    let bits = b.feedback(pc);
    let args: Vec<Val> = (0..nargs).map(|i| b.reg(a + 4 + i)).collect();
    let all = args.iter().enumerate().fold(TypeSet::empty(), |s, (i, &v)| {
        let group = if i == 0 { bits } else { bits >> fb::ARGS_SHIFT };
        s | arg_kinds(b, v, group & fb::KINDS)
    });
    let within = |set: TypeSet| set.contains(all);
    if nargs > 1 && !within(TypeSet::SMALL | TypeSet::FLOAT) && !within(TypeSet::INT) {
        if !within(TypeSet::STR) {
            return if nargs == 2 && within(TypeSet::NUM) {
                builtin(b, h, a, nargs, c, f, callee)
            } else {
                Emitted::No
            };
        }
    }
    pin(b, f, callee);
    // The pick so far and the next argument, `x < y` deciding.
    let below = |b: &mut Builder<'_, 'gc>, op: Op, pick: Val, v: Val| {
        if max { b.ins(op, &[pick, v]) } else { b.ins(op, &[v, pick]) }
    };
    let r = if nargs == 1 {
        args[0]
    } else if within(TypeSet::SMALL) || within(TypeSet::INT) || within(TypeSet::FLOAT) {
        let (conv, op): (fn(&mut Builder<'_, 'gc>, Val) -> Val, Op) = if within(TypeSet::SMALL) {
            (|b, v| b.as_i32(v), Op::ICmp(Cc::Lt))
        } else if within(TypeSet::INT) {
            (|b, v| b.as_i64(v), Op::LCmp(Cc::Lt))
        } else {
            (|b, v| b.as_f64(v), Op::FCmp(Cc::Lt))
        };
        let mut pick = conv(b, args[0]);
        for &v in &args[1..] {
            let v = conv(b, v);
            let lt = below(b, op, pick, v);
            pick = b.ins(Op::Select, &[lt, v, pick]);
        }
        pick
    } else if within(TypeSet::SMALL | TypeSet::FLOAT) {
        let num = |b: &mut Builder<'_, 'gc>, v: Val| {
            let t = b.ty(v);
            if t.rep != Rep::Val || (!t.set.is_empty() && t.within(TypeSet::SMALL | TypeSet::FLOAT)) {
                return (b.num_f64(v), b.boxed(v));
            }
            let bv = b.boxed(v);
            let i = b.ins_exit(Op::ToF64(TypeSet::SMALL | TypeSet::FLOAT), &[bv], ExitTag::Type);
            (b.f.result(i), bv)
        };
        let (mut at, mut pick) = num(b, args[0]);
        for &v in &args[1..] {
            let (x, v) = num(b, v);
            let lt = below(b, Op::FCmp(Cc::Lt), at, x);
            at = b.ins(Op::Select, &[lt, x, at]);
            pick = b.ins(Op::Select, &[lt, v, pick]);
        }
        pick
    } else {
        let str = |b: &mut Builder<'_, 'gc>, v: Val| {
            let v = b.boxed(v);
            let g = b.guard(v, TypeSet::STR);
            (b.ins(Op::Unbox(Rep::Ptr), &[g]), g)
        };
        let (mut at, mut pick) = str(b, args[0]);
        for &v in &args[1..] {
            let (p, v) = str(b, v);
            let lt = below(b, Op::StrLt, at, p);
            at = b.ins(Op::Select, &[lt, p, at]);
            pick = b.ins(Op::Select, &[lt, v, pick]);
        }
        pick
    };
    if c == 2 {
        b.set(a, r);
    }
    Emitted::Done
}

/// `string.len(s)`, `string.sub(s, i [, j])` and `string.byte(s [, i])`'s
/// one result, of a string `s` (a number converts in the builtin, so a call
/// that saw one calls it): the length inline, `byte` inline until a position
/// out of range is seen, `sub` (and `byte` then) by a helper given the
/// payload and i64 positions (`position`; one that may be a string calls
/// the builtin).
#[allow(clippy::too_many_arguments)]
fn string<'gc>(
    b: &mut Builder<'_, 'gc>,
    pc: u32,
    intr: Intrinsic,
    a: u8,
    nargs: u8,
    c: u8,
    f: Function<'gc>,
    callee: Val,
) -> Emitted {
    let (min, max) = match intr {
        Intrinsic::StrLen => (1, 1),
        Intrinsic::Builtin(HelperId::StrSub) => (2, 3),
        _ => (1, 2),
    };
    if !(min..=max).contains(&nargs) || c == 0 || c > 2 {
        return Emitted::No;
    }
    let bits = b.feedback(pc);
    // Nothing records these calls' kinds but exits: until one does, the
    // string is a string and the positions integers, guarded.
    let kinds = |b: &Builder<'_, 'gc>, v: Val, seen: fb::Bits, guess: TypeSet| {
        let k = arg_kinds(b, v, seen);
        if seen == 0 && k.contains(guess) { guess } else { k }
    };
    let s = b.reg(a + 4);
    if kinds(b, s, bits & fb::KINDS, TypeSet::STR) != TypeSet::STR {
        return Emitted::No;
    }
    let rest = bits >> fb::ARGS_SHIFT & fb::KINDS;
    let mut pos = Vec::new();
    for k in 1..nargs {
        let v = b.reg(a + 4 + k);
        let kv = kinds(b, v, rest, TypeSet::INT);
        if kv.contains(TypeSet::STR) {
            return Emitted::No;
        }
        pos.push((v, kv));
    }
    pin(b, f, callee);
    let s = b.guard(s, TypeSet::STR);
    let p = b.ins(Op::Unbox(Rep::Ptr), &[s]);
    let mut pos: Vec<Val> = pos.into_iter().map(|(v, k)| position(b, v, k)).collect();
    let r = match intr {
        Intrinsic::StrLen => b.ins(Op::StrLenOf, &[p]),
        Intrinsic::Builtin(HelperId::StrSub) => {
            if pos.len() == 1 {
                pos.push(b.ins(Op::KI64(-1), &[]));
            }
            b.ins(Op::Helper(HelperId::StrSub), &[p, pos[0], pos[1]])
        }
        _ => {
            let i = pos.first().copied().unwrap_or_else(|| b.ins(Op::KI64(1), &[]));
            if bits & fb::OTHER_RESULT == 0 {
                let r = b.ins_exit(Op::StrByteAt, &[p, i], ExitTag::OtherResult);
                b.f.result(r)
            } else {
                b.ins(Op::Helper(HelperId::StrByte), &[p, i])
            }
        }
    };
    if c == 2 {
        b.set(a, r);
    }
    Emitted::Done
}

/// A string position of kinds `k` as the i64 the builtin takes
/// (`luaL_checkinteger`): an integer, or a float with an integer value
/// (otherwise the builtin raises, to which the conversion exits). One that
/// may be either converts through a float, exact where it matters: past
/// ±2^31 only its sign counts.
fn position(b: &mut Builder<'_, '_>, v: Val, k: TypeSet) -> Val {
    let exact = |b: &mut Builder<'_, '_>, f: Val| {
        let i = b.ins_exit(Op::FToLExact, &[f], ExitTag::Slow);
        b.f.result(i)
    };
    match b.ty(v).rep {
        Rep::I32 => return b.ins(Op::IToL, &[v]),
        Rep::I64 => return v,
        Rep::F64 => return exact(b, v),
        _ => {}
    }
    if TypeSet::INT.contains(k) {
        b.as_i64(v)
    } else if TypeSet::FLOAT.contains(k) {
        let f = b.as_f64(v);
        exact(b, f)
    } else {
        let v = b.boxed(v);
        let f = b.ins_exit(Op::ToF64(TypeSet::NUM), &[v], ExitTag::Slow);
        let f = b.f.result(f);
        exact(b, f)
    }
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
    let bits = b.feedback(pc);
    // A tail call or MULTRET wants all results; the iterators make up to
    // four, the others one.
    let tail = insn.op() == crate::instruction::Op::TAILCALL;
    let arity = match intr {
        Intrinsic::Atan | Intrinsic::Log => nargs == 1 || nargs == 2,
        Intrinsic::FMod => nargs == 2,
        _ => nargs == 1,
    };
    if c == 0 || (c > 2 && !iterators) || (iterators && (tail || c > 5)) || !arity {
        return Emitted::No;
    }
    let x = b.reg(a + 4);
    let k = pin(b, f, callee);
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
        Intrinsic::Sqrt => {
            let x = float_arg(b, x);
            b.ins(Op::FSqrt, &[x])
        }
        Intrinsic::Unary(l) => {
            let x = float_arg(b, x);
            b.ins(Op::Helper(HelperId::Libm(l)), &[x])
        }
        Intrinsic::Atan => {
            let y = float_arg(b, x);
            let x = if nargs == 2 {
                let x = b.reg(a + 5);
                float_arg(b, x)
            } else {
                b.kf64(1.0)
            };
            b.ins(Op::Helper(HelperId::Libm(Libm::Atan2)), &[y, x])
        }
        Intrinsic::Log => {
            let x = float_arg(b, x);
            let libm = |b: &mut Builder<'_, '_>, l, x| b.ins(Op::Helper(HelperId::Libm(l)), &[x]);
            if nargs == 1 {
                libm(b, Libm::Log, x)
            } else {
                let base = b.reg(a + 5);
                match const_f64(b, base) {
                    Some(2.0) => libm(b, Libm::Log2, x),
                    Some(10.0) => libm(b, Libm::Log10, x),
                    // `log(base)` by the same libm the builtin calls.
                    Some(k) => {
                        let ln = libm(b, Libm::Log, x);
                        let d = b.kf64(k.ln());
                        b.ins(Op::FDiv, &[ln, d])
                    }
                    None => {
                        log_arms(b, a, x, base);
                        return Emitted::Done;
                    }
                }
            }
        }
        Intrinsic::FMod => {
            let y = b.reg(a + 5);
            match fmod(b, a, x, y, bits) {
                Some(r) => r,
                None => return Emitted::Done,
            }
        }
        Intrinsic::Floor | Intrinsic::Ceil => match round(b, a, x, bits, intr == Intrinsic::Ceil) {
            Some(r) => r,
            None => return Emitted::Done,
        },
        Intrinsic::Abs => match abs(b, a, x, bits) {
            Some(r) => r,
            None => return Emitted::Done,
        },
        Intrinsic::Pairs | Intrinsic::Ipairs => unreachable!("returned above"),
        Intrinsic::Select
        | Intrinsic::StrLen
        | Intrinsic::Builtin(_)
        | Intrinsic::Assert
        | Intrinsic::Error => {
            unreachable!("not math")
        }
    };
    if c == 2 {
        b.set(a, r);
    }
    done(slow, k)
}

/// `R[a] = log(x, base)` of a base that varies, compared as the builtin
/// does after converting it: 2 and 10 by their own functions, any other
/// as `log(x) / log(base)`. The arms' floats join unboxed.
fn log_arms(b: &mut Builder<'_, '_>, a: u8, x: Val, base: Val) {
    let libm = |b: &mut Builder<'_, '_>, l, x| b.ins(Op::Helper(HelperId::Libm(l)), &[x]);
    let rep = b.ty(base).rep;
    let base = match rep {
        Rep::I32 | Rep::I64 => base,
        _ => float_arg(b, base),
    };
    let is = |b: &mut Builder<'_, '_>, n: i32| match rep {
        Rep::I32 => {
            let k = b.ki32(n);
            b.ins(Op::ICmp(Cc::Eq), &[base, k])
        }
        Rep::I64 => {
            let k = b.ins(Op::KI64(n as i64), &[]);
            b.ins(Op::LCmp(Cc::Eq), &[base, k])
        }
        _ => {
            let k = b.kf64(n as f64);
            b.ins(Op::FCmp(Cc::Eq), &[base, k])
        }
    };
    let join = b.new_block();
    for (n, l) in [(2, Libm::Log2), (10, Libm::Log10)] {
        let c = is(b, n);
        let other = branch_arm(b, c);
        let r = libm(b, l, x);
        arm_end(b, join, a, r);
        b.switch_to(other);
    }
    let ln = libm(b, Libm::Log, x);
    let f = b.as_f64(base);
    let lb = libm(b, Libm::Log, f);
    let r = b.ins(Op::FDiv, &[ln, lb]);
    arm_end(b, join, a, r);
    enter_join(b, join);
}

/// The f64 of an argument of a builtin whose result is a float whatever it
/// gets: converted by its representation, else in an arm per kind its type
/// allows (strings by a call); only what the builtin raises on exits.
fn float_arg(b: &mut Builder<'_, '_>, x: Val) -> Val {
    let ty = b.ty(x);
    if matches!(ty.rep, Rep::F64 | Rep::I32 | Rep::I64)
        || (!ty.set.is_empty() && ty.within(TypeSet::SMALL | TypeSet::FLOAT))
    {
        return b.num_f64(x);
    }
    let xv = b.boxed(x);
    let i = b.ins_exit(Op::ToF64(TypeSet::NUM | TypeSet::STR), &[xv], ExitTag::Slow);
    b.f.result(i)
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

/// The kinds a type-following builtin's argument `v` takes arms for: its
/// representation's, else its type narrowed to what the call's feedback saw
/// of it (`seen`, the argument's group of bits; empty for a call the
/// interpreter has not run), which may speculate (the arms' tests exit for
/// the rest).
fn arg_kinds(b: &Builder<'_, '_>, v: Val, seen: fb::Bits) -> TypeSet {
    let t = b.ty(v);
    let own = match t.rep {
        Rep::I32 => return TypeSet::SMALL,
        Rep::I64 => return TypeSet::INT,
        Rep::F64 => return TypeSet::FLOAT,
        _ if t.set.is_empty() => TypeSet::ANY,
        _ => t.set,
    };
    let mut saw = TypeSet::empty();
    for (bit, set) in [
        (fb::SMALL, TypeSet::SMALL),
        (fb::BIGINT, TypeSet::BIGINT),
        (fb::FLOAT, TypeSet::FLOAT),
        (fb::STR, TypeSet::STR),
        (fb::TAB, TypeSet::TAB),
        (fb::OTHER, TypeSet::ANY - TypeSet::NUM - TypeSet::STR - TypeSet::TAB),
    ] {
        if seen & bit != 0 {
            saw |= set;
        }
    }
    match own & saw {
        k if k.is_empty() => own,
        k => k,
    }
}

/// One arm of [`dispatch`]: the kinds it takes, its result's
/// representation, and the code making the result of the argument.
type Arm<'f, 'gc> = (TypeSet, Rep, &'f dyn Fn(&mut Builder<'_, 'gc>, Val) -> Val);

/// `R[a] =` the arm of `x`'s kind: each tested in turn, the last guarded,
/// exiting with `rest` for any other kind. Their results join as they are
/// when of one representation, else boxed. Without arms (the call saw only
/// what the builtin raises on) it exits.
fn dispatch<'gc>(b: &mut Builder<'_, 'gc>, a: u8, x: Val, arms: &[Arm<'_, 'gc>], rest: ExitTag) {
    let x = b.boxed(x);
    match arms {
        [] => {
            b.deopt(ExitTag::Slow);
            return;
        }
        // No join to go through.
        &[(set, _, f)] => {
            let g = b.ins_exit(Op::Guard(set), &[x], rest);
            let v = b.f.result(g);
            let r = f(b, v);
            b.set(a, r);
            return;
        }
        _ => {}
    }
    let boxed = arms.iter().any(|arm| arm.1 != arms[0].1 || arm.1 == Rep::Val);
    let join = b.new_block();
    for (n, &(set, _, f)) in arms.iter().enumerate() {
        let other = if n + 1 < arms.len() {
            let c = b.ins(Op::IsType(set), &[x]);
            Some(branch_arm(b, c))
        } else {
            None
        };
        let v = match other {
            Some(_) => b.ins(Op::Refine(set), &[x]),
            None => {
                let g = b.ins_exit(Op::Guard(set), &[x], rest);
                b.f.result(g)
            }
        };
        let r = f(b, v);
        let r = if boxed { b.boxed(r) } else { r };
        arm_end(b, join, a, r);
        if let Some(o) = other {
            b.switch_to(o);
        }
    }
    enter_join(b, join);
}

/// The exit of a dispatch whose arms take `covered`: one that may learn a
/// number or string kind they miss, or only what the builtin raises on.
fn rest_tag(covered: TypeSet) -> ExitTag {
    if covered.contains(TypeSet::NUM | TypeSet::STR) {
        ExitTag::Slow
    } else {
        ExitTag::Type
    }
}

/// Whether a type-following builtin's integers are i64 (no small | big
/// union): the argument may be a big integer, or a result left i32.
fn wide_ints(kinds: TypeSet, bits: fb::Bits) -> bool {
    kinds.contains(TypeSet::BIGINT) || bits & (fb::OVERFLOW | fb::OTHER_RESULT) != 0
}

/// `abs(x)`: `fabs` of a float or a string's number, an integer's
/// magnitude in i32 (`i32::MIN` exits to learn the overflow) or, once wide,
/// in i64, wrapping as Lua's.
fn abs<'gc>(b: &mut Builder<'_, 'gc>, a: u8, x: Val, bits: fb::Bits) -> Option<Val> {
    let k = arg_kinds(b, x, bits & fb::KINDS);
    let wide = wide_ints(k, bits);
    let fabs = |b: &mut Builder<'_, 'gc>, v: Val| {
        let f = b.ins(Op::Unbox(Rep::F64), &[v]);
        b.ins(Op::FAbs, &[f])
    };
    let sabs = |b: &mut Builder<'_, 'gc>, v: Val| {
        let f = b.ins_exit(Op::ToF64(TypeSet::STR), &[v], ExitTag::Slow);
        let f = b.f.result(f);
        b.ins(Op::FAbs, &[f])
    };
    let iabs = |b: &mut Builder<'_, 'gc>, v: Val| {
        let i = b.as_i32(v);
        abs_i32(b, i)
    };
    let labs = |b: &mut Builder<'_, 'gc>, v: Val| {
        let i = b.as_i64(v);
        abs_i64(b, i)
    };
    match b.ty(x).rep {
        Rep::F64 => return Some(b.ins(Op::FAbs, &[x])),
        Rep::I32 if !wide => return Some(abs_i32(b, x)),
        Rep::I32 | Rep::I64 => return Some(labs(b, x)),
        _ => {}
    }
    let mut arms: Vec<Arm<'_, 'gc>> = Vec::new();
    let mut covered = TypeSet::empty();
    if k.contains(TypeSet::FLOAT) {
        arms.push((TypeSet::FLOAT, Rep::F64, &fabs));
        covered |= TypeSet::FLOAT;
    }
    if k.intersects(TypeSet::INT) {
        let (set, rep, f): Arm<'_, 'gc> = if wide {
            (TypeSet::INT, Rep::I64, &labs)
        } else {
            (TypeSet::SMALL, Rep::I32, &iabs)
        };
        arms.push((set, rep, f));
        covered |= set;
    }
    if k.contains(TypeSet::STR) {
        arms.push((TypeSet::STR, Rep::F64, &sabs));
        covered |= TypeSet::STR;
    }
    dispatch(b, a, x, &arms, rest_tag(covered));
    None
}

/// `floor(x)` (`ceil`): an integer is its own; a float or a string's
/// number rounded, then its integer: in i32 (beyond exits to learn the
/// overflow), once wide in i64 (no integer representation exits to learn
/// it), and once a float result was seen either, boxed.
fn round<'gc>(b: &mut Builder<'_, 'gc>, a: u8, x: Val, bits: fb::Bits, ceil: bool) -> Option<Val> {
    let k = arg_kinds(b, x, bits & fb::KINDS);
    let wide = wide_ints(k, bits);
    let floats = bits & fb::OTHER_RESULT != 0;
    let to_int = move |b: &mut Builder<'_, 'gc>, f: Val| {
        let r = b.ins(if ceil { Op::FCeil } else { Op::FFloor }, &[f]);
        if floats {
            // An integer where it has one, else the float: both boxed.
            let i = b.ins(Op::FToL, &[r]);
            let back = b.ins(Op::LToF, &[i]);
            let same = b.ins(Op::FCmp(Cc::Eq), &[r, back]);
            let max = b.ins(Op::KI64(i64::MAX), &[]);
            let sat = b.ins(Op::LCmp(Cc::Ne), &[i, max]);
            let join = b.new_block();
            let no = branch_arm(b, same);
            let no2 = branch_arm(b, sat);
            let v = b.ins(Op::Box, &[i]);
            arm_end(b, join, a, v);
            for o in [no, no2] {
                b.switch_to(o);
                let v = b.ins(Op::Box, &[r]);
                arm_end(b, join, a, v);
            }
            enter_join(b, join);
            b.reg(a)
        } else if wide {
            let i = b.ins_exit(Op::FToLExact, &[r], ExitTag::OtherResult);
            b.f.result(i)
        } else {
            let i = b.ins_exit(Op::FToIExact, &[r], ExitTag::Overflow);
            b.f.result(i)
        }
    };
    let rep = if floats {
        Rep::Val
    } else if wide {
        Rep::I64
    } else {
        Rep::I32
    };
    match b.ty(x).rep {
        Rep::I32 | Rep::I64 => return Some(x),
        Rep::F64 => return Some(to_int(b, x)),
        _ => {}
    }
    let fround = |b: &mut Builder<'_, 'gc>, v: Val| {
        let f = b.ins(Op::Unbox(Rep::F64), &[v]);
        to_int(b, f)
    };
    let sround = |b: &mut Builder<'_, 'gc>, v: Val| {
        let f = b.ins_exit(Op::ToF64(TypeSet::STR), &[v], ExitTag::Slow);
        let f = b.f.result(f);
        to_int(b, f)
    };
    let int = |b: &mut Builder<'_, 'gc>, v: Val| match rep {
        Rep::I32 => b.as_i32(v),
        _ => b.as_i64(v),
    };
    let mut arms: Vec<Arm<'_, 'gc>> = Vec::new();
    let mut covered = TypeSet::empty();
    if k.intersects(TypeSet::INT) {
        let set = if rep == Rep::I32 { TypeSet::SMALL } else { TypeSet::INT };
        arms.push((set, rep, &int));
        covered |= set;
    }
    if k.contains(TypeSet::FLOAT) {
        arms.push((TypeSet::FLOAT, rep, &fround));
        covered |= TypeSet::FLOAT;
    }
    if k.contains(TypeSet::STR) {
        arms.push((TypeSet::STR, rep, &sround));
        covered |= TypeSet::STR;
    }
    dispatch(b, a, x, &arms, rest_tag(covered));
    None
}

/// `fmod(x, y)`: C's `%` when both are integers (i64 once either may be
/// big; a zero divisor exits for the builtin to raise), else libm's `fmod`
/// of the numbers. An argument that may not be an integer has the float
/// path to itself, guarded so; two that may be either take both arms,
/// joined boxed.
fn fmod<'gc>(b: &mut Builder<'_, 'gc>, a: u8, x: Val, y: Val, bits: fb::Bits) -> Option<Val> {
    let kx = arg_kinds(b, x, bits & fb::KINDS);
    let ky = arg_kinds(b, y, bits >> fb::ARGS_SHIFT & fb::KINDS);
    let wide = (kx | ky).contains(TypeSet::BIGINT);
    let int = |b: &mut Builder<'_, 'gc>, v: Val| {
        if wide { b.as_i64(v) } else { b.as_i32(v) }
    };
    let irem = |b: &mut Builder<'_, 'gc>, x: Val, y: Val| {
        let (x, y) = (int(b, x), int(b, y));
        let op = if wide { Op::LRem } else { Op::IRem };
        let i = b.ins_exit(op, &[x, y], ExitTag::Slow);
        b.f.result(i)
    };
    let frem = |b: &mut Builder<'_, 'gc>, x: Val, y: Val| {
        let (x, y) = (float_arg(b, x), float_arg(b, y));
        b.ins(Op::Helper(HelperId::Libm(Libm::Fmod)), &[x, y])
    };
    let (ix, iy) = (kx.intersects(TypeSet::INT), ky.intersects(TypeSet::INT));
    if !ix || !iy {
        // Never two integers: the one that is not stays so.
        let not_int = TypeSet::ANY - TypeSet::INT;
        let (x, y) = if !ix {
            (b.guard(x, not_int), y)
        } else {
            (x, b.guard(y, not_int))
        };
        return Some(frem(b, x, y));
    }
    if TypeSet::INT.contains(kx) && TypeSet::INT.contains(ky) {
        return Some(irem(b, x, y));
    }
    let (xv, yv) = (b.boxed(x), b.boxed(y));
    let join = b.new_block();
    let maybe: Vec<Val> = [(xv, kx), (yv, ky)]
        .into_iter()
        .filter(|&(_, k)| !TypeSet::INT.contains(k))
        .map(|(v, _)| v)
        .collect();
    let floats = all_in(b, TypeSet::INT, &maybe);
    let r = irem(b, xv, yv);
    let r = b.boxed(r);
    arm_end(b, join, a, r);
    enter_join(b, floats);
    let r = frem(b, xv, yv);
    let r = b.boxed(r);
    arm_end(b, join, a, r);
    enter_join(b, join);
    None
}

/// `|x|` of an i32; `abs(i32::MIN)` leaves i32 and exits to learn it.
fn abs_i32(b: &mut Builder<'_, '_>, x: Val) -> Val {
    let i = b.ins_exit(Op::INeg, &[x], ExitTag::Overflow);
    let neg = b.f.result(i);
    let zero = b.ki32(0);
    let below = b.ins(Op::ICmp(Cc::Lt), &[x, zero]);
    b.ins(Op::Select, &[below, neg, x])
}

/// `|x|` of an i64, wrapping as Lua's: `abs(math.mininteger)` is itself.
fn abs_i64(b: &mut Builder<'_, '_>, x: Val) -> Val {
    let zero = b.ins(Op::KI64(0), &[]);
    let neg = b.ins(Op::LNeg, &[x]);
    let below = b.ins(Op::LCmp(Cc::Lt), &[x, zero]);
    b.ins(Op::Select, &[below, neg, x])
}
