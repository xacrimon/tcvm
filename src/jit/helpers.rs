//! Rust routines regions call with `bl` (the C ABI). None grows the stack,
//! switches threads, runs Lua or raises (R5).

use crate::dmm::Gc;
use crate::dmm::lock::RefLock;
use crate::env::MetamethodBits;
use crate::env::function::UpvalueCell;
use crate::env::string::LuaString;
use crate::env::table::{Step, Table, TableState};
use crate::env::thread::ThreadState;
use crate::env::value::{Value, ValueKind};
use crate::instruction::ArithKind;
use crate::instruction::TFOR_VARS;
use crate::jit::ir::ops::{HELPER_FAIL, HELPER_FAIL_VAL, HelperId};
use crate::lua::{Context, State};
use crate::vm::frame::{self, land_results};
use crate::vm::num::{self, ArithOp, SlowNum};
use crate::vm::ops::control::{close_upvalues, new_closure, vararg_get, vararg_table};
use crate::vm::ops::meta::{
    IndexChain, NewIndexChain, binop_metamethod, walk_index_chain, walk_newindex_chain,
};

/// The address a region calls for `h`.
pub(crate) fn address(h: HelperId) -> usize {
    match h {
        HelperId::FMod => jit_fmod as *const () as usize,
        HelperId::FPow => jit_pow as *const () as usize,
        HelperId::Sin => jit_sin as *const () as usize,
        HelperId::Cos => jit_cos as *const () as usize,
        HelperId::Lt => jit_lt as *const () as usize,
        HelperId::Le => jit_le as *const () as usize,
        HelperId::Eq => jit_eq as *const () as usize,
        HelperId::Arith(_) => jit_arith as *const () as usize,
        HelperId::Unm => jit_unm as *const () as usize,
        HelperId::BNot => jit_bnot as *const () as usize,
        HelperId::Len => jit_len as *const () as usize,
        HelperId::Concat => jit_concat as *const () as usize,
        HelperId::UpvalBarrier(_) => jit_upval_barrier as *const () as usize,
        HelperId::Closure { .. } => jit_closure as *const () as usize,
        HelperId::Close(_) => jit_close as *const () as usize,
        HelperId::NewTable(_) => jit_newtable as *const () as usize,
        HelperId::SetList { .. } => jit_setlist as *const () as usize,
        HelperId::VarargGet => jit_varargget as *const () as usize,
        HelperId::VarargTable(_) => jit_vararg_table as *const () as usize,
        HelperId::TforNext { .. } => jit_tfor_next as *const () as usize,
        HelperId::TforIpairs => jit_tfor_ipairs as *const () as usize,
        HelperId::TopArgs(_) => jit_top_args as *const () as usize,
        HelperId::VarargAll(_) => jit_vararg_all as *const () as usize,
        HelperId::Index => jit_index as *const () as usize,
        HelperId::Mm(_) => jit_mm as *const () as usize,
        HelperId::IndexFn => jit_index_fn as *const () as usize,
        HelperId::NewIndexFn => jit_newindex_fn as *const () as usize,
        HelperId::NewIndex { .. } => jit_newindex as *const () as usize,
        HelperId::Type => jit_type as *const () as usize,
        HelperId::GetMetatable => jit_getmetatable as *const () as usize,
        HelperId::SetMetatable => jit_setmetatable as *const () as usize,
        HelperId::RawGet => jit_rawget as *const () as usize,
        HelperId::RawSet => jit_rawset as *const () as usize,
        HelperId::RawEqual => jit_rawequal as *const () as usize,
        HelperId::RawLen => jit_rawlen as *const () as usize,
        HelperId::ToString => jit_tostring as *const () as usize,
        HelperId::ToNumber => jit_tonumber as *const () as usize,
        HelperId::StrLen => jit_strlen as *const () as usize,
        HelperId::StrSub => jit_strsub as *const () as usize,
        HelperId::StrByte => jit_strbyte as *const () as usize,
    }
}

/// A float result as a `Value` holds it: NaNs canonical.
#[inline(always)]
fn canonical(x: f64) -> f64 {
    if x.is_nan() {
        f64::from_bits(0x7FF8_0000_0000_0000)
    } else {
        x
    }
}

pub(crate) extern "C" fn jit_fmod(a: f64, b: f64) -> f64 {
    canonical(<num::Mod as ArithOp>::float_raw(a, b))
}

pub(crate) extern "C" fn jit_pow(a: f64, b: f64) -> f64 {
    canonical(<num::Pow as ArithOp>::float_raw(a, b))
}

pub(crate) extern "C" fn jit_sin(x: f64) -> f64 {
    canonical(x.sin())
}

pub(crate) extern "C" fn jit_cos(x: f64) -> f64 {
    canonical(x.cos())
}

/// The interpreter's primitive `<` (`le`: `<=`), as `cmp_slow` has it.
fn compare(a: Value<'_>, b: Value<'_>, le: bool) -> i32 {
    let r = if let (Some(x), Some(y)) = (a.get_integer(), b.get_integer()) {
        if le { x <= y } else { x < y }
    } else if let (Some(x), Some(y)) = (a.get_integer(), b.get_float()) {
        if le {
            num::le_int_float(x, y)
        } else {
            num::lt_int_float(x, y)
        }
    } else if let (Some(x), Some(y)) = (a.get_float(), b.get_integer()) {
        if le {
            num::le_float_int(x, y)
        } else {
            num::lt_float_int(x, y)
        }
    } else if let (Some(x), Some(y)) = (a.get_float(), b.get_float()) {
        if le { x <= y } else { x < y }
    } else if let (Some(x), Some(y)) = (a.get_string(), b.get_string()) {
        if le { x <= y } else { x < y }
    } else {
        return HELPER_FAIL;
    };
    r as i32
}

pub(crate) extern "C" fn jit_lt(_rt: *const State<'static>, a: u64, b: u64) -> i32 {
    unsafe { compare(Value::from_raw(a), Value::from_raw(b), false) }
}

pub(crate) extern "C" fn jit_le(_rt: *const State<'static>, a: u64, b: u64) -> i32 {
    unsafe { compare(Value::from_raw(a), Value::from_raw(b), true) }
}

/// The interpreter's `==` without running `__eq`, which fails instead.
pub(crate) extern "C" fn jit_eq(rt: *const State<'static>, a: u64, b: u64) -> i32 {
    let (a, b) = unsafe { (Value::from_raw(a), Value::from_raw(b)) };
    if num::raw_eq(a, b) {
        return 1;
    }
    // Lua 5.5: `__eq` only between two tables or two userdata.
    let both = |k| a.kind() == k && b.kind() == k;
    if both(ValueKind::Table) || both(ValueKind::Userdata) {
        let ctx = unsafe { Context::from_state(rt) };
        if !binop_metamethod(ctx, a, b, crate::env::MetamethodBits::EQ).is_nil() {
            return HELPER_FAIL;
        }
    }
    0
}

/// `arith_slow`'s numeric cases; everything it would raise or hand to a
/// metamethod fails.
pub(crate) extern "C" fn jit_arith(rt: *const State<'static>, kind: u32, a: u64, b: u64) -> u64 {
    use ArithKind::*;
    let ctx = unsafe { Context::from_state(rt) };
    let mc = ctx.mutation();
    let (a, b) = unsafe { (Value::from_raw(a), Value::from_raw(b)) };
    const KINDS: [ArithKind; 13] = [
        None, Add, Sub, Mul, Mod, Pow, Div, IDiv, BAnd, BOr, BXor, Shl, Shr,
    ];
    let r = match KINDS[kind as usize] {
        Add => num::op_arith_slow::<num::Add>(mc, a, b),
        Sub => num::op_arith_slow::<num::Sub>(mc, a, b),
        Mul => num::op_arith_slow::<num::Mul>(mc, a, b),
        Mod => num::op_arith_slow::<num::Mod>(mc, a, b),
        Pow => num::op_arith_slow::<num::Pow>(mc, a, b),
        Div => num::op_arith_slow::<num::Div>(mc, a, b),
        IDiv => num::op_arith_slow::<num::IDiv>(mc, a, b),
        BAnd => num::op_bit_slow::<num::BAnd>(mc, a, b),
        BOr => num::op_bit_slow::<num::BOr>(mc, a, b),
        BXor => num::op_bit_slow::<num::BXor>(mc, a, b),
        Shl => num::op_bit_slow::<num::Shl>(mc, a, b),
        Shr => num::op_bit_slow::<num::Shr>(mc, a, b),
        None => SlowNum::NotNumbers,
    };
    match r {
        SlowNum::Value(v) => v.to_raw(),
        _ => HELPER_FAIL_VAL,
    }
}

/// `-v` as `op_unm` and `unm_slow` compute it without a metamethod.
pub(crate) extern "C" fn jit_unm(rt: *const State<'static>, v: u64) -> u64 {
    let ctx = unsafe { Context::from_state(rt) };
    let v = unsafe { Value::from_raw(v) };
    if v.is_float() {
        Value::float(-v.read_float()).to_raw()
    } else if let Some(i) = v.get_integer() {
        Value::integer(ctx.mutation(), i.wrapping_neg()).to_raw()
    } else {
        HELPER_FAIL_VAL
    }
}

/// `~v` as `op_bnot` and `bnot_slow` compute it without a metamethod.
pub(crate) extern "C" fn jit_bnot(rt: *const State<'static>, v: u64) -> u64 {
    let ctx = unsafe { Context::from_state(rt) };
    match unsafe { Value::from_raw(v) }.get_integer() {
        Some(i) => Value::integer(ctx.mutation(), !i).to_raw(),
        None => HELPER_FAIL_VAL,
    }
}

/// `#v` of a string, or of a table without `__len`.
pub(crate) extern "C" fn jit_len(rt: *const State<'static>, v: u64) -> u64 {
    let ctx = unsafe { Context::from_state(rt) };
    let v = unsafe { Value::from_raw(v) };
    let n = if let Some(s) = v.get_string() {
        s.len()
    } else if let Some(t) = v.get_table()
        && ctx.mm_of(v, MetamethodBits::LEN).is_nil()
    {
        t.raw_len()
    } else {
        return HELPER_FAIL_VAL;
    };
    Value::integer(ctx.mutation(), n as i64).to_raw()
}

/// `a .. b` of strings and numbers, as `op_concat` builds it.
pub(crate) extern "C" fn jit_concat(rt: *const State<'static>, a: u64, b: u64) -> u64 {
    let ctx = unsafe { Context::from_state(rt) };
    let (a, b) = unsafe { (Value::from_raw(a), Value::from_raw(b)) };
    let s = ctx.with_buf(|buf| {
        buf.reserve(num::concat_len(a) + num::concat_len(b));
        (num::coerce_to_str(buf, a) && num::coerce_to_str(buf, b)).then(|| LuaString::new(ctx, buf))
    });
    match s {
        Some(s) => Value::string(s).to_raw(),
        None => HELPER_FAIL_VAL,
    }
}

/// The barrier of a store into upvalue `i`'s cell, as `op_setupval` runs it.
pub(crate) extern "C" fn jit_upval_barrier(
    rt: *const State<'static>,
    thread: *mut ThreadState<'static>,
    base: *mut Value<'static>,
    i: u64,
) {
    let ctx = unsafe { Context::from_state(rt) };
    let closure = unsafe { frame::closure(base) };
    let cell = unsafe { (*closure.upvalue_ptr().add(i as usize)).cell };
    let running = unsafe { (*thread).handle() };
    if let Some(target) = UpvalueCell::barrier_target(cell, running)
        && !Gc::is_gray(target)
    {
        ctx.mutation().backward_barrier(target, None);
    }
}

/// `CLOSURE` into `R[dst]` (`imm`: `proto | dst << 32`), reading the frame's
/// slots.
pub(crate) extern "C" fn jit_closure(
    rt: *const State<'static>,
    thread: *mut ThreadState<'static>,
    base: *mut Value<'static>,
    imm: u64,
) -> u64 {
    let ctx = unsafe { Context::from_state(rt) };
    let ts = unsafe { &mut *thread };
    let closure = unsafe { frame::closure(base) };
    let f = new_closure(
        ctx,
        ts,
        base,
        closure,
        imm as u32 as usize,
        (imm >> 32) as u8,
    );
    Value::function(f).to_raw()
}

/// Close the frame's open upvalues from `R[a]` up.
pub(crate) extern "C" fn jit_close(
    rt: *const State<'static>,
    thread: *mut ThreadState<'static>,
    base: *mut Value<'static>,
    a: u64,
) {
    let ctx = unsafe { Context::from_state(rt) };
    let ts = unsafe { &mut *thread };
    let level = ts.slot_index(base) + a as usize;
    close_upvalues(ctx.mutation(), ts, level);
}

/// A `TFORCALL_NEXT` step (`imm`: `a | count << 8`) as `op_tforcall_next`
/// and `tfor_next` take it, through the frame's slots.
pub(crate) extern "C" fn jit_tfor_next(
    rt: *const State<'static>,
    base: *mut Value<'static>,
    imm: u64,
) -> u64 {
    let ctx = unsafe { Context::from_state(rt) };
    let (a, count) = (imm as u8 as usize, (imm >> 8) as u8 as usize);
    let reg = |r: usize| unsafe { base.add(r) };
    let (Some(t), Some(pos)) = (
        unsafe { *reg(a + 1) }.get_table(),
        unsafe { *reg(a + 3) }.get_small().filter(|&p| p >= 0),
    ) else {
        return HELPER_FAIL_VAL;
    };
    let pos = pos as u32;
    let step = match t.inner().borrow().next_at_inline(pos) {
        Step::Entry(next, k, v) => Some((Some(next), k, v)),
        Step::End => None,
        Step::Slow => t.inner().borrow().next_at(ctx.mutation(), pos),
    };
    let vars = a + TFOR_VARS as usize;
    unsafe {
        match step {
            Some((next, k, v)) => {
                // Past a position that doesn't fit, `next` resumes from the key.
                *reg(a + 3) = next.map_or(Value::nil(), |p| Value::small(p as i32));
                *reg(vars) = k;
                if count > 1 {
                    *reg(vars + 1) = v;
                }
                for i in 2..count {
                    *reg(vars + i) = Value::nil();
                }
                1
            }
            None => {
                for i in 0..count {
                    *reg(vars + i) = Value::nil();
                }
                0
            }
        }
    }
}

/// An `ipairs` step past the array part (`tfor_ipairs`): `t[k]`, nil at the
/// end, or a failure where `__index` may answer.
pub(crate) extern "C" fn jit_tfor_ipairs(_rt: *const State<'static>, t: u64, k: u64) -> u64 {
    let (t, k) = unsafe { (Value::from_raw(t), Value::from_raw(k)) };
    let (Some(t), Some(k)) = (t.get_table(), k.get_integer()) else {
        return HELPER_FAIL_VAL;
    };
    let state = t.inner().borrow();
    let v = state.get_int(k);
    if v.is_nil() && state.shape().has_mm(MetamethodBits::INDEX) {
        return HELPER_FAIL_VAL;
    }
    v.to_raw()
}

/// `NEWTABLE` from template `t` of the running prototype.
pub(crate) extern "C" fn jit_newtable(
    rt: *const State<'static>,
    base: *mut Value<'static>,
    t: u64,
) -> u64 {
    let ctx = unsafe { Context::from_state(rt) };
    let closure = unsafe { frame::closure(base) };
    let t = &closure.proto.templates[t as usize];
    Value::table(Table::from_template(ctx.mutation(), t)).to_raw()
}

/// `SETLIST` (`imm`: `a | n << 8 | off << 16`, `n` 0 for up to `top`) from
/// the frame's slots, as `op_setlist` does.
pub(crate) extern "C" fn jit_setlist(
    rt: *const State<'static>,
    thread: *mut ThreadState<'static>,
    base: *mut Value<'static>,
    imm: u64,
) {
    let ctx = unsafe { Context::from_state(rt) };
    let (a, n, off) = (
        imm as u8 as usize,
        (imm >> 8) as u8 as usize,
        (imm >> 16) as usize,
    );
    let n = if n == 0 {
        let ts = unsafe { &*thread };
        ts.top - ts.slot_index(base) - a - 1
    } else {
        n
    };
    // SAFETY: the compiler gives SETLIST a table and its items' registers.
    let t = unsafe { (*base.add(a)).get_table().unwrap_unchecked() };
    let items = unsafe { std::slice::from_raw_parts(base.add(a + 1), n) };
    if !Gc::is_gray(t.inner()) {
        ctx.mutation().backward_barrier(Gc::erase(t.inner()), None);
    }
    unsafe { t.borrow_mut_barriered() }.set_list(ctx.mutation(), off, items);
}

/// The arguments of the CALL at `R[a]` up to `top`.
pub(crate) extern "C" fn jit_top_args(
    thread: *mut ThreadState<'static>,
    base: *mut Value<'static>,
    a: u64,
) -> i32 {
    let ts = unsafe { &*thread };
    (ts.top - ts.slot_index(base) - a as usize - frame::HDR) as i32
}

/// `VARARG a 0`: every extra into `R[a..]` and `top` after them, as
/// `op_vararg` does; a failure where the stack must grow first.
pub(crate) extern "C" fn jit_vararg_all(
    thread: *mut ThreadState<'static>,
    base: *mut Value<'static>,
    a: u64,
) -> u64 {
    let ts = unsafe { &mut *thread };
    let nv = unsafe { frame::nv(base) };
    let target = unsafe { base.add(a as usize) };
    let end = unsafe { target.add(nv) };
    if end.cast_const() > ts.stack_end {
        return HELPER_FAIL_VAL;
    }
    ts.top = ts.slot_index(end);
    // The extras end at the header and the target starts at or above the
    // base, so the ranges don't overlap.
    unsafe { land_results(target, base.sub(frame::HDR + nv), nv, nv) };
    0
}

/// `VARARGGET` of `key`.
pub(crate) extern "C" fn jit_varargget(
    rt: *const State<'static>,
    base: *mut Value<'static>,
    key: u64,
) -> u64 {
    let ctx = unsafe { Context::from_state(rt) };
    vararg_get(ctx, base, unsafe { Value::from_raw(key) }).to_raw()
}

/// `VARARGPREP`'s vararg table into `R[np]`.
pub(crate) extern "C" fn jit_vararg_table(
    rt: *const State<'static>,
    base: *mut Value<'static>,
    np: u64,
) -> u64 {
    let ctx = unsafe { Context::from_state(rt) };
    Value::table(vararg_table(ctx, base, np as usize)).to_raw()
}

/// `recv[key]` as `get_slow` resolves it without calling a function.
pub(crate) extern "C" fn jit_index(rt: *const State<'static>, recv: u64, key: u64) -> u64 {
    let ctx = unsafe { Context::from_state(rt) };
    let (recv, key) = unsafe { (Value::from_raw(recv), Value::from_raw(key)) };
    if let Some(t) = recv.get_table() {
        let v = t.raw_get(key);
        if !v.is_nil() {
            return v.to_raw();
        }
    }
    match walk_index_chain(ctx, recv, key) {
        IndexChain::Resolved(v) => v.to_raw(),
        _ => HELPER_FAIL_VAL,
    }
}

/// The metamethod of `bit` from `a`, else `b` (`binop_metamethod`), or nil.
pub(crate) extern "C" fn jit_mm(rt: *const State<'static>, bit: u64, a: u64, b: u64) -> u64 {
    let ctx = unsafe { Context::from_state(rt) };
    let (a, b) = unsafe { (Value::from_raw(a), Value::from_raw(b)) };
    binop_metamethod(ctx, a, b, MetamethodBits::from_bits_retain(bit as u32)).to_raw()
}

/// The `__index` function `recv[key]` calls with `recv` itself, or nil: a
/// call the walk reaches through other tables first is the interpreter's.
pub(crate) extern "C" fn jit_index_fn(rt: *const State<'static>, recv: u64, key: u64) -> u64 {
    let ctx = unsafe { Context::from_state(rt) };
    let (recv, key) = unsafe { (Value::from_raw(recv), Value::from_raw(key)) };
    match walk_index_chain(ctx, recv, key) {
        IndexChain::Invoke { func, receiver } if receiver.to_raw() == recv.to_raw() => {
            Value::function(func).to_raw()
        }
        _ => Value::nil().to_raw(),
    }
}

/// The same for `__newindex`.
pub(crate) extern "C" fn jit_newindex_fn(rt: *const State<'static>, recv: u64, key: u64) -> u64 {
    let ctx = unsafe { Context::from_state(rt) };
    let (recv, key) = unsafe { (Value::from_raw(recv), Value::from_raw(key)) };
    match walk_newindex_chain(ctx, recv, key) {
        NewIndexChain::Invoke { func, receiver } if receiver.to_raw() == recv.to_raw() => {
            Value::function(func).to_raw()
        }
        _ => Value::nil().to_raw(),
    }
}

/// `recv[key] = v` as `set_slow` does it without calling a function or
/// raising; `keyed` stores as `t[k] = v` does.
pub(crate) extern "C" fn jit_newindex(
    rt: *const State<'static>,
    keyed: u64,
    recv: u64,
    key: u64,
    v: u64,
) -> u64 {
    let ctx = unsafe { Context::from_state(rt) };
    let (recv, key, v) = unsafe {
        (
            Value::from_raw(recv),
            Value::from_raw(key),
            Value::from_raw(v),
        )
    };
    let NewIndexChain::RawSet(t) = walk_newindex_chain(ctx, recv, key) else {
        return HELPER_FAIL_VAL;
    };
    if key.is_nil() || key.get_float().is_some_and(f64::is_nan) {
        return HELPER_FAIL_VAL;
    }
    if keyed != 0 {
        t.raw_set_keyed(ctx, key, v);
    } else {
        t.raw_set(ctx, key, v);
    }
    0
}

/// The backward barrier of the table whose lock is `t`, not gray.
pub(crate) extern "C" fn jit_barrier(
    rt: *const State<'static>,
    t: *const RefLock<TableState<'static>>,
) {
    let ctx = unsafe { Context::from_state(rt) };
    let g = unsafe { Gc::from_ptr(t) };
    ctx.mutation().backward_barrier(Gc::erase(g), None);
}

/// Box an integer outside i32.
pub(crate) extern "C" fn jit_box_i64(rt: *const State<'static>, v: i64) -> u64 {
    let ctx = unsafe { Context::from_state(rt) };
    Value::integer(ctx.mutation(), v).to_raw()
}

/// Land a call's `nret` results at `values` into `R[a..]`: `c - 1` of them,
/// or all with `top` set for MULTRET (`c == 0`).
pub(crate) extern "C" fn jit_land(
    _rt: *const State<'static>,
    thread: *mut ThreadState<'static>,
    base: *mut Value<'static>,
    a: usize,
    values: *const Value<'static>,
    nret: usize,
    c: usize,
) {
    unsafe {
        let dst = base.add(a);
        let wanted = if c == 0 { nret } else { c - 1 };
        land_results(dst, values, nret, wanted);
        if c == 0 {
            let ts = &mut *thread;
            ts.top = ts.slot_index(dst) + wanted;
        }
    }
}

// --- builtins (`jit-design.md` 12) -------------------------------------------

/// `type(v)`: the interned name.
pub(crate) extern "C" fn jit_type(rt: *const State<'static>, v: u64) -> u64 {
    let ctx = unsafe { Context::from_state(rt) };
    let v = unsafe { Value::from_raw(v) };
    Value::string(LuaString::new(ctx, v.type_name().as_bytes())).to_raw()
}

/// `getmetatable(v)`, its `__metatable` field shadowing it.
pub(crate) extern "C" fn jit_getmetatable(rt: *const State<'static>, v: u64) -> u64 {
    let ctx = unsafe { Context::from_state(rt) };
    let v = unsafe { Value::from_raw(v) };
    match ctx.metatable_of(v) {
        Some(mt) => {
            let prot = mt.raw_get(Value::string(ctx.symbols().metatable));
            if prot.is_nil() { Value::table(mt) } else { prot }.to_raw()
        }
        None => Value::nil().to_raw(),
    }
}

/// `setmetatable(t, mt)` on a table without one, as `ff_setmetatable`; the
/// builtin does the rest (a protected metatable, a nil one, errors).
pub(crate) extern "C" fn jit_setmetatable(rt: *const State<'static>, t: u64, mt: u64) -> u64 {
    let ctx = unsafe { Context::from_state(rt) };
    let (tv, mt) = unsafe { (Value::from_raw(t), Value::from_raw(mt)) };
    let (Some(t), Some(mt)) = (tv.get_table(), mt.get_table()) else {
        return HELPER_FAIL_VAL;
    };
    if t.set_metatable_fast(ctx.mutation(), mt, ctx.symbols()).is_none() {
        if t.metatable().is_some() {
            return HELPER_FAIL_VAL;
        }
        t.set_metatable(ctx, Some(mt));
    }
    tv.to_raw()
}

/// `rawget(t, k)`.
pub(crate) extern "C" fn jit_rawget(t: u64, k: u64) -> u64 {
    let (t, k) = unsafe { (Value::from_raw(t), Value::from_raw(k)) };
    match t.get_table() {
        Some(t) => t.raw_get(k).to_raw(),
        None => HELPER_FAIL_VAL,
    }
}

/// `rawset(t, k, v)`: `t`, or the builtin's error.
pub(crate) extern "C" fn jit_rawset(rt: *const State<'static>, t: u64, k: u64, v: u64) -> u64 {
    let ctx = unsafe { Context::from_state(rt) };
    let (tv, k, v) = unsafe { (Value::from_raw(t), Value::from_raw(k), Value::from_raw(v)) };
    let Some(t) = tv.get_table() else {
        return HELPER_FAIL_VAL;
    };
    if k.is_nil() || k.get_float().is_some_and(f64::is_nan) {
        return HELPER_FAIL_VAL;
    }
    t.raw_set_keyed(ctx, k, v);
    tv.to_raw()
}

/// `rawequal(a, b)`: 1 or 0.
pub(crate) extern "C" fn jit_rawequal(a: u64, b: u64) -> u64 {
    let (a, b) = unsafe { (Value::from_raw(a), Value::from_raw(b)) };
    num::raw_eq(a, b) as u64
}

/// `rawlen(v)` of a table or string.
pub(crate) extern "C" fn jit_rawlen(rt: *const State<'static>, v: u64) -> u64 {
    let ctx = unsafe { Context::from_state(rt) };
    let v = unsafe { Value::from_raw(v) };
    let n = if let Some(s) = v.get_string() {
        s.len() as i64
    } else if let Some(t) = v.get_table() {
        t.raw_len() as i64
    } else {
        return HELPER_FAIL_VAL;
    };
    Value::integer(ctx.mutation(), n).to_raw()
}

/// `tostring(v)` without a `__tostring` to call.
pub(crate) extern "C" fn jit_tostring(rt: *const State<'static>, v: u64) -> u64 {
    let ctx = unsafe { Context::from_state(rt) };
    let v = unsafe { Value::from_raw(v) };
    if !ctx.mm_of(v, MetamethodBits::TOSTRING).is_nil() {
        return HELPER_FAIL_VAL;
    }
    Value::string(crate::builtin::util::basic_tostring(ctx, v)).to_raw()
}

/// `tonumber(v)` with one argument.
pub(crate) extern "C" fn jit_tonumber(rt: *const State<'static>, v: u64) -> u64 {
    let ctx = unsafe { Context::from_state(rt) };
    let v = unsafe { Value::from_raw(v) };
    if v.get_integer().is_some() || v.get_float().is_some() {
        return v.to_raw();
    }
    let Some(s) = v.get_string() else {
        return Value::nil().to_raw();
    };
    crate::builtin::util::str_to_number(s.as_bytes())
        .map_or(Value::nil(), |n| n.into_value(ctx.mutation()))
        .to_raw()
}

/// `string.sub(s, i [, j])` with small-integer positions (`j` nil: -1).
pub(crate) extern "C" fn jit_strsub(rt: *const State<'static>, s: u64, i: u64, j: u64) -> u64 {
    let ctx = unsafe { Context::from_state(rt) };
    let (s, i, j) = unsafe { (Value::from_raw(s), Value::from_raw(i), Value::from_raw(j)) };
    let (Some(s), Some(i)) = (s.get_string(), i.get_small()) else {
        return HELPER_FAIL_VAL;
    };
    let j = match j.get_small() {
        Some(j) => i64::from(j),
        None if j.is_nil() => -1,
        None => return HELPER_FAIL_VAL,
    };
    let bytes = s.as_bytes();
    let len = bytes.len();
    let start = crate::builtin::posrelat(i64::from(i), len).max(1);
    let end = crate::builtin::posrelat(j, len).min(len as i64);
    let r = if start <= end {
        &bytes[(start - 1) as usize..end as usize]
    } else {
        &[][..]
    };
    Value::string(LuaString::new(ctx, r)).to_raw()
}

/// `string.byte(s [, i])`'s one result: the byte, or nil out of range.
pub(crate) extern "C" fn jit_strbyte(s: u64, i: u64) -> u64 {
    let (s, i) = unsafe { (Value::from_raw(s), Value::from_raw(i)) };
    let Some(s) = s.get_string() else {
        return HELPER_FAIL_VAL;
    };
    let i = match i.get_small() {
        Some(i) => i64::from(i),
        None if i.is_nil() => 1,
        None => return HELPER_FAIL_VAL,
    };
    let bytes = s.as_bytes();
    let k = crate::builtin::posrelat(i, bytes.len());
    match (k >= 1).then(|| bytes.get(k as usize - 1)).flatten() {
        Some(&b) => Value::small(b as i32).to_raw(),
        None => Value::nil().to_raw(),
    }
}

/// `string.len(s)` of a string.
pub(crate) extern "C" fn jit_strlen(rt: *const State<'static>, v: u64) -> u64 {
    let ctx = unsafe { Context::from_state(rt) };
    let v = unsafe { Value::from_raw(v) };
    match v.get_string() {
        Some(s) => Value::integer(ctx.mutation(), s.len() as i64).to_raw(),
        None => HELPER_FAIL_VAL,
    }
}
