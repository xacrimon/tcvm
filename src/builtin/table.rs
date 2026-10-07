use crate::Context;
use crate::builtin::util;
use crate::env::{
    Error, Function, LuaString, MetamethodBits, NativeClosure, NativeFn, Stack, Table, Value,
};
use crate::vm::async_native::{AsyncError, Cx};
use crate::vm::native::{ContFn, NativeOut, OnOk, Protect, cont};
use crate::vm::num;
use crate::vm::{
    IndexChain, NewIndexChain, binop_metamethod, walk_index_chain, walk_newindex_chain,
};

// What an argument must support (`ltablib.c`'s `TAB_R`/`TAB_W`/`TAB_L`).
const TAB_R: MetamethodBits = MetamethodBits::INDEX;
const TAB_W: MetamethodBits = MetamethodBits::NEWINDEX;
const TAB_L: MetamethodBits = MetamethodBits::LEN;

/// `checktab`: accept a table, or a value whose metatable has every
/// metamethod in `what` (a string needs no `__len`). Yields the table when it
/// has none of them, so the caller can take its raw path; `None` means every
/// access must go through [`util::geti`] and friends.
#[inline]
fn check_tab<'gc>(
    ctx: Context<'gc>,
    v: Value<'gc>,
    fname: &str,
    arg: usize,
    what: MetamethodBits,
) -> Result<Option<Table<'gc>>, Error<'gc>> {
    if let Some(t) = v.get_table() {
        return Ok((!t.shape().has_any_mm(what)).then_some(t));
    }
    check_tab_meta(ctx, v, fname, arg, what)
}

#[cold]
#[inline(never)]
fn check_tab_meta<'gc>(
    ctx: Context<'gc>,
    v: Value<'gc>,
    fname: &str,
    arg: usize,
    what: MetamethodBits,
) -> Result<Option<Table<'gc>>, Error<'gc>> {
    let s = ctx.symbols();
    let has = |mt: Table<'gc>, name| !mt.raw_get(Value::string(name)).is_nil();
    let ok = ctx.metatable_of(v).is_some_and(|mt| {
        (!what.contains(TAB_R) || has(mt, s.mm_index))
            && (!what.contains(TAB_W) || has(mt, s.mm_newindex))
            && (!what.contains(TAB_L) || v.get_string().is_some() || has(mt, s.mm_len))
    });
    if ok {
        Ok(None)
    } else {
        Err(util::type_error(ctx, fname, arg, "table", Some(v)))
    }
}

pub fn load<'gc>(ctx: Context<'gc>) {
    let fns: &[(&str, NativeFn)] = &[("create", lua_create), ("pack", lua_pack)];
    let actions: &[(&str, ContFn)] = &[
        ("concat", lua_concat),
        ("insert", lua_insert),
        ("move", lua_move),
        ("remove", lua_remove),
        ("sort", lua_sort),
        ("unpack", lua_unpack),
    ];

    let lib = Table::new(ctx);
    let set = |name: &str, f: Function<'gc>| {
        let key = Value::string(LuaString::new(ctx, name.as_bytes()));
        lib.raw_set(ctx, key, Value::function(f));
    };
    for &(name, handler) in fns {
        set(name, Function::new_native(ctx.mutation(), handler, &[]));
    }
    for &(name, handler) in actions {
        set(name, Function::new_cont(ctx.mutation(), handler, &[]));
    }

    let lib_name = Value::string(LuaString::new(ctx, b"table"));
    ctx.globals().raw_set(ctx, lib_name, Value::table(lib));
}

/// `concat(t [, sep [, i [, j]]])` — concatenate `t[i]..t[j]` (numbers
/// stringified) joined by `sep`. `sep` defaults to `""`, `i` to 1, `j` to `#t`.
fn lua_concat<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> NativeOut {
    let Some(t) = check_tab(ctx, stack.get(0), "concat", 1, TAB_R | TAB_L)? else {
        return concat_meta(ctx, &mut stack);
    };
    let (sep, mut i, last) = concat_args(ctx, &stack, t.raw_len() as i64)?;
    let s = ctx.with_buf(|out| {
        // `i < last` rather than `i <= last`, so `i` never steps past `last`.
        while i < last {
            add_field(ctx, out, t.raw_get(Value::integer(ctx.mutation(), i)), i)?;
            out.extend_from_slice(&sep);
            i += 1;
        }
        if i == last {
            add_field(ctx, out, t.raw_get(Value::integer(ctx.mutation(), i)), i)?;
        }
        Ok(LuaString::new(ctx, out))
    })?;
    stack.ret1(Value::string(s));
    NativeOut::RETURN
}

/// `concat` on a value whose accesses may run metamethods.
#[cold]
#[inline(never)]
fn concat_meta<'gc>(ctx: Context<'gc>, stack: &mut Stack<'gc, '_>) -> NativeOut {
    stack.spawn_action(ctx, |cx| async move {
        let last = util::len(&cx, 0).await?;
        let (sep, mut i, last) = cx.try_enter(|ctx, stack| concat_args(ctx, stack, last))?;
        let mut out = Vec::new();
        while i < last {
            concat_field(&cx, &mut out, i).await?;
            out.extend_from_slice(&sep);
            i += 1;
        }
        if i == last {
            concat_field(&cx, &mut out, i).await?;
        }
        cx.enter(|ctx, mut stack| stack.ret1(Value::string(LuaString::new(ctx, &out))));
        Ok(())
    })
}

/// `concat`'s separator and range; `last` is `#t`, the default end.
fn concat_args<'gc>(
    ctx: Context<'gc>,
    stack: &Stack<'gc, '_>,
    last: i64,
) -> Result<(Vec<u8>, i64, i64), Error<'gc>> {
    let sep = util::opt_string(ctx, stack.get(1), "concat", 2)?
        .map_or(Vec::new(), |s| s.as_bytes().to_vec());
    let i_arg = stack.get(2);
    let i = if i_arg.is_nil() {
        1
    } else {
        util::check_integer(ctx, i_arg, "concat", 3)?
    };
    let j_arg = stack.get(3);
    let j = if j_arg.is_nil() {
        last
    } else {
        util::check_integer(ctx, j_arg, "concat", 4)?
    };
    Ok((sep, i, j))
}

async fn concat_field(cx: &Cx, out: &mut Vec<u8>, i: i64) -> Result<(), AsyncError> {
    util::geti(cx, 0, i).await?;
    cx.try_enter(|ctx, stack| add_field(ctx, out, stack.pop(), i))
}

/// `addfield`: append `t[i]`'s value `v`, which must be a string or number.
fn add_field<'gc>(
    ctx: Context<'gc>,
    out: &mut Vec<u8>,
    v: Value<'gc>,
    i: i64,
) -> Result<(), Error<'gc>> {
    if let Some(s) = v.get_string() {
        out.extend_from_slice(s.as_bytes());
    } else if let Some(n) = v.get_integer() {
        util::push_int(out, n);
    } else if let Some(f) = v.get_float() {
        util::push_float(out, f);
    } else {
        return Err(Error::from_str(
            ctx,
            &format!(
                "invalid value ({}) at index {i} in table for 'concat'",
                v.type_name()
            ),
        ));
    }
    Ok(())
}

/// `create(n [, m])` — return a fresh table. The `n`/`m` capacity hints are
/// accepted and validated but not yet used for preallocation; the table grows
/// on demand. TODO(#27): honor the hints once `Table` exposes reservation.
fn lua_create<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<(), Error<'gc>> {
    let n = util::check_integer(ctx, stack.get(0), "create", 1)?;
    let m_arg = stack.get(1);
    let m = if m_arg.is_nil() {
        0
    } else {
        util::check_integer(ctx, m_arg, "create", 2)?
    };
    let in_range = |k: i64| (0..=i64::from(i32::MAX)).contains(&k);
    if !in_range(n) {
        return Err(util::arg_error(ctx, "create", 1, "out of range"));
    }
    if !in_range(m) {
        return Err(util::arg_error(ctx, "create", 2, "out of range"));
    }
    // PUC's hash part holds at most 2^30 nodes (`MAXHBITS`).
    if m > 1 << 30 {
        return Err(util::runtime_error(ctx, "table overflow"));
    }
    let t = Table::with_capacity(ctx, n as usize, m as usize);
    stack.ret1(Value::table(t));
    Ok(())
}

/// `insert(t, [pos,] value)` — append `value`, or insert it at `pos`, shifting
/// later elements up by one.
fn lua_insert<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> NativeOut {
    let Some(t) = check_tab(ctx, stack.get(0), "insert", 1, TAB_R | TAB_W | TAB_L)? else {
        return insert_meta(ctx, &mut stack);
    };
    let e = (t.raw_len() as i64).wrapping_add(1);
    let pos = insert_pos(ctx, &stack, e)?;
    let mut i = e;
    while i > pos {
        let v = t.raw_get(Value::integer(ctx.mutation(), i - 1));
        t.raw_set(ctx, Value::integer(ctx.mutation(), i), v);
        i -= 1;
    }
    let v = stack.pop();
    t.raw_set(ctx, Value::integer(ctx.mutation(), pos), v);
    stack.clear();
    NativeOut::RETURN
}

/// `insert` on a value whose accesses may run metamethods.
#[cold]
#[inline(never)]
fn insert_meta<'gc>(ctx: Context<'gc>, stack: &mut Stack<'gc, '_>) -> NativeOut {
    stack.spawn_action(ctx, |cx| async move {
        let e = util::len(&cx, 0).await?.wrapping_add(1);
        let pos = cx.try_enter(|ctx, stack| insert_pos(ctx, stack, e))?;
        let mut i = e;
        while i > pos {
            util::geti(&cx, 0, i - 1).await?;
            util::seti(&cx, 0, i).await?;
            i -= 1;
        }
        // The value argument is on top.
        util::seti(&cx, 0, pos).await?;
        cx.enter(|_ctx, mut stack| stack.clear());
        Ok(())
    })
}

/// `insert`'s target position; `e` is `#t + 1`, the default.
fn insert_pos<'gc>(ctx: Context<'gc>, stack: &Stack<'gc, '_>, e: i64) -> Result<i64, Error<'gc>> {
    match stack.len() {
        2 => Ok(e),
        3 => {
            let pos = util::check_integer(ctx, stack.get(1), "insert", 2)?;
            // `pos` in `[1, e]`, compared unsigned as the reference does.
            if (pos as u64).wrapping_sub(1) >= e as u64 {
                return Err(util::arg_error(ctx, "insert", 2, "position out of bounds"));
            }
            Ok(pos)
        }
        _ => Err(Error::from_str(
            ctx,
            "wrong number of arguments to 'insert'",
        )),
    }
}

/// `move(a1, f, e, t [, a2])` — copy `a1[f..e]` to `a2[t..]` (`a2` defaults to
/// `a1`), returning `a2`. Overlapping ranges within one table are handled in
/// the safe direction.
fn lua_move<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> NativeOut {
    let f = util::check_integer(ctx, stack.get(1), "move", 2)?;
    let e = util::check_integer(ctx, stack.get(2), "move", 3)?;
    let t = util::check_integer(ctx, stack.get(3), "move", 4)?;
    let tt = if stack.get(4).is_nil() { 0 } else { 4 };
    let src = check_tab(ctx, stack.get(0), "move", 1, TAB_R)?;
    let dst = check_tab(ctx, stack.get(tt), "move", tt + 1, TAB_W)?;
    let (a1, a2) = (stack.get(0), stack.get(tt));
    if e < f {
        stack.ret1(a2);
        return NativeOut::RETURN;
    }
    // PUC-Lua's two bounds: the element count `e - f + 1` must fit a Lua
    // integer (else `e - f` itself overflows), and the destination range
    // `t .. t + n - 1` must not wrap past maxinteger.
    if !(f > 0 || e < i64::MAX + f) {
        return NativeOut::error(util::arg_error(ctx, "move", 3, "too many elements to move"));
    }
    let n = e - f + 1;
    if t > i64::MAX - n + 1 {
        return NativeOut::error(util::arg_error(ctx, "move", 4, "destination wrap around"));
    }

    // Copy forward unless the destination overlaps the tail of the source
    // and `a1 == a2`, which may take `__eq` (`None`: ask it). The guards
    // above keep `f + i` / `t + i` in range.
    let mut eq_mm = Value::nil();
    let forward = if t > e || t <= f {
        Some(true)
    } else if tt == 0 || util::raw_eq(a1, a2) {
        Some(false)
    } else {
        let same_kind = (a1.get_table().is_some() && a2.get_table().is_some())
            || (a1.get_userdata().is_some() && a2.get_userdata().is_some());
        if same_kind {
            eq_mm = binop_metamethod(ctx, a1, a2, MetamethodBits::EQ);
        }
        eq_mm.is_nil().then_some(true)
    };

    if let (Some(src), Some(dst), Some(forward)) = (src, dst, forward) {
        for k in 0..n {
            let i = if forward { k } else { n - 1 - k };
            let v = src.raw_get(Value::integer(ctx.mutation(), f + i));
            dst.raw_set(ctx, Value::integer(ctx.mutation(), t + i), v);
        }
        stack.ret1(a2);
        return NativeOut::RETURN;
    }
    move_meta(ctx, &mut stack, f, n, t, tt, forward, eq_mm)
}

/// `move` of `n` elements when an access may run metamethods or the copy
/// direction needs `eq_mm` (`forward` is `None`).
#[cold]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn move_meta<'gc>(
    ctx: Context<'gc>,
    stack: &mut Stack<'gc, '_>,
    f: i64,
    n: i64,
    t: i64,
    tt: usize,
    forward: Option<bool>,
    eq_mm: Value<'gc>,
) -> NativeOut {
    // `__eq` goes above the arguments, for the future to call.
    if forward.is_none() {
        stack.push(eq_mm);
    }
    stack.spawn_action(ctx, move |cx| async move {
        let forward = match forward {
            Some(forward) => forward,
            None => {
                let at = cx.enter(|_ctx, mut stack| {
                    let at = stack.len() - 1;
                    stack.extend([stack.get(0), stack.get(tt)]);
                    at
                });
                cx.call(at).await;
                cx.enter(|_ctx, mut stack| {
                    let equal = !stack.get(at).is_falsy();
                    stack.truncate(at);
                    !equal
                })
            }
        };
        for k in 0..n {
            let i = if forward { k } else { n - 1 - k };
            util::geti(&cx, 0, f + i).await?;
            util::seti(&cx, tt, t + i).await?;
        }
        cx.enter(|_ctx, mut stack| stack.ret1(stack.get(tt)));
        Ok(())
    })
}

/// `pack(...)` — collect all arguments into a new table with field `n` set to
/// the argument count.
fn lua_pack<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<(), Error<'gc>> {
    let n = stack.len();
    let t = Table::new(ctx);
    for i in 0..n {
        t.raw_set(
            ctx,
            Value::integer(ctx.mutation(), i as i64 + 1),
            stack.get(i),
        );
    }
    t.raw_set(
        ctx,
        Value::string(ctx.symbols().n),
        Value::integer(ctx.mutation(), n as i64),
    );
    stack.ret1(Value::table(t));
    Ok(())
}

/// `remove(t [, pos])` — remove and return `t[pos]` (default `#t`), shifting
/// later elements down by one.
fn lua_remove<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> NativeOut {
    let Some(t) = check_tab(ctx, stack.get(0), "remove", 1, TAB_R | TAB_W | TAB_L)? else {
        return remove_meta(ctx, &mut stack);
    };
    let size = t.raw_len() as i64;
    let mut pos = remove_pos(ctx, &stack, size)?;
    let result = t.raw_get(Value::integer(ctx.mutation(), pos));
    while pos < size {
        let v = t.raw_get(Value::integer(ctx.mutation(), pos + 1));
        t.raw_set(ctx, Value::integer(ctx.mutation(), pos), v);
        pos += 1;
    }
    t.raw_set(ctx, Value::integer(ctx.mutation(), pos), Value::nil());
    stack.ret1(result);
    NativeOut::RETURN
}

/// `remove` on a value whose accesses may run metamethods.
#[cold]
#[inline(never)]
fn remove_meta<'gc>(ctx: Context<'gc>, stack: &mut Stack<'gc, '_>) -> NativeOut {
    stack.spawn_action(ctx, |cx| async move {
        let size = util::len(&cx, 0).await?;
        let mut pos = cx.try_enter(|ctx, stack| remove_pos(ctx, stack, size))?;
        // The result stays on the stack below the shuffling.
        util::geti(&cx, 0, pos).await?;
        while pos < size {
            util::geti(&cx, 0, pos + 1).await?;
            util::seti(&cx, 0, pos).await?;
            pos += 1;
        }
        cx.enter(|_ctx, mut stack| stack.push(Value::nil()));
        util::seti(&cx, 0, pos).await?;
        cx.enter(|_ctx, mut stack| {
            let result = stack.pop();
            stack.ret1(result)
        });
        Ok(())
    })
}

/// `remove`'s position; `size` is `#t`, the default.
fn remove_pos<'gc>(
    ctx: Context<'gc>,
    stack: &Stack<'gc, '_>,
    size: i64,
) -> Result<i64, Error<'gc>> {
    let pos_arg = stack.get(1);
    let pos = if pos_arg.is_nil() {
        size
    } else {
        util::check_integer(ctx, pos_arg, "remove", 2)?
    };
    // Any position but the default must lie in `[1, size + 1]`.
    if pos != size && (pos as u64).wrapping_sub(1) > size as u64 {
        return Err(util::arg_error(ctx, "remove", 2, "position out of bounds"));
    }
    Ok(pos)
}

/// `sort(t [, comp])` — sort `t[1..#t]` in place. With no comparator the default
/// `<` order is used (numbers, strings, or an `__lt` metamethod); otherwise
/// `comp(a, b)` must return true when `a` should precede `b`. An inconsistent
/// comparator raises "invalid order function for sorting".
fn lua_sort<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> NativeOut {
    let n = match check_tab(ctx, stack.get(0), "sort", 1, TAB_R | TAB_W | TAB_L)? {
        Some(t) => t.raw_len() as i64,
        None => {
            let v = stack.get(0);
            if let Some(s) = v.get_string() {
                s.len() as i64
            } else {
                let mm = ctx.mm_of(v, MetamethodBits::LEN);
                if mm.is_nil() {
                    // `checktab` let it through for its `__len`.
                    v.get_table().map_or(0, |t| t.raw_len() as i64)
                } else {
                    // `#t` is `__len`'s first result.
                    stack.truncate(2);
                    while stack.len() < 2 {
                        stack.push(Value::nil());
                    }
                    stack.extend([mm, v, v]);
                    return NativeOut::call_then(2, cont::SORT_LEN, Protect::No, OnOk::Cont);
                }
            }
        }
    };
    sort_start(ctx, stack, n)
}

/// `__len` returned `sort`'s `#t`.
pub(crate) fn sort_len_cont<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
    _status: Result<(), Error<'gc>>,
) -> NativeOut {
    let n = util::to_integer(stack.get(2))
        .ok_or_else(|| Error::from_str(ctx, "object length is not an integer"))?;
    stack.truncate(2);
    sort_start(ctx, stack, n)
}

/// Sort `t[1..n]`, `n` being `#t`.
fn sort_start<'gc>(ctx: Context<'gc>, mut stack: Stack<'gc, '_>, n: i64) -> NativeOut {
    if !sort_args(ctx, &mut stack, n)? {
        stack.clear();
        return NativeOut::RETURN;
    }
    if stack.get(1).is_nil()
        && let Some(t) = stack.get(0).get_table()
        && !t.shape().has_any_mm(TAB_R | TAB_W)
        && sort_primitive(ctx, t, n as usize)
    {
        stack.clear();
        return NativeOut::RETURN;
    }
    stack.extend([Value::nil(); LO - S0]);
    stack.extend([Value::small(0); STATE - LO]);
    stack.as_mut_slice()[LO] = Value::small(1);
    stack.as_mut_slice()[UP] = Value::small(n as i32);
    sort_drive(ctx, stack, SortResume::Start)
}

/// Validate `sort`'s arguments for a length-`n` array and leave the window as
/// `[t, comp]`. False when there is nothing to sort, in which case, matching
/// Lua, the comparator is not even type-checked.
fn sort_args<'gc>(
    ctx: Context<'gc>,
    stack: &mut Stack<'gc, '_>,
    n: i64,
) -> Result<bool, Error<'gc>> {
    if n <= 1 {
        return Ok(false);
    }
    if n >= i32::MAX as i64 {
        return Err(util::arg_error(ctx, "sort", 1, "array too big"));
    }
    let comp = stack.get(1);
    if !comp.is_nil() && comp.get_function().is_none() {
        return Err(util::type_error(ctx, "sort", 2, "function", Some(comp)));
    }
    stack.truncate(2);
    if stack.len() == 1 {
        stack.push(Value::nil());
    }
    Ok(true)
}

/// `sort` without a comparator of numbers but NaN, or of strings: a total
/// order, under which no comparison can fail and equal values may end up in
/// any order (the manual), so any sort gives a reference result. False,
/// having done nothing, for any other contents.
fn sort_primitive<'gc>(ctx: Context<'gc>, t: Table<'gc>, n: usize) -> bool {
    let mc = ctx.mutation();
    let mut vals: Vec<Value<'gc>> = (1..=n as i64)
        .map(|k| t.raw_get(Value::integer(mc, k)))
        .collect();
    let number =
        |v: &Value<'gc>| v.get_integer().is_some() || v.get_float().is_some_and(|f| !f.is_nan());
    if vals.iter().all(|v| v.get_integer().is_some()) {
        let mut is: Vec<i64> = vals.iter().map(|v| v.get_integer().unwrap()).collect();
        is.sort_unstable();
        vals = is.into_iter().map(|i| Value::integer(mc, i)).collect();
    } else if vals.iter().all(number) {
        vals.sort_unstable_by(|x, y| {
            if prim_lt(*x, *y) == Some(true) {
                std::cmp::Ordering::Less
            } else if prim_lt(*y, *x) == Some(true) {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        });
    } else if vals.iter().all(|v| v.get_string().is_some()) {
        vals.sort_unstable_by(|x, y| {
            let (a, b) = (x.get_string().unwrap(), y.get_string().unwrap());
            a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal)
        });
    } else {
        return false;
    }
    for (k, v) in vals.into_iter().enumerate() {
        t.raw_set(ctx, Value::integer(mc, k as i64 + 1), v);
    }
    true
}

// `sort_drive`'s working values and state live in fixed window slots above
// `[t, comp]`, so they survive the calls it makes as a continuation; then the
// pending ranges, two slots each, and the call being made. Small integers
// all, kept in `Value::small` form.
const S0: usize = 2;
const S1: usize = 3;
const S2: usize = 4;
const LO: usize = 5;
const UP: usize = 6;
const I: usize = 7;
const J: usize = 8;
const P: usize = 9;
const PHASE: usize = 10;
/// Where the call being made sits.
const AT: usize = 11;
/// The reads and writes a step queued (`QUEUE_LEN` of three slots each:
/// kind, key, slot), done in order before the next step: a comparator may
/// give the table `__index` or `__newindex`, through which they then go.
const NQ: usize = 12;
const QUEUE: usize = 13;
const QUEUE_LEN: usize = 4;
const STATE: usize = QUEUE + 3 * QUEUE_LEN;

const OP_GET: i32 = 0;
const OP_SET: i32 = 1;

// `sort_drive`'s steps: the comparison sites of `auxsort` and the steps
// between them.
const PH_RANGE: i32 = 0;
const PH_A: i32 = 1;
const PH_B: i32 = 2;
const PH_C: i32 = 3;
const PH_PIVOT: i32 = 4;
const PH_D_NEXT: i32 = 5;
const PH_D: i32 = 6;
const PH_E_NEXT: i32 = 7;
const PH_E: i32 = 8;
const PH_SWAP: i32 = 9;

/// What the call `sort_drive` made returned.
enum SortResume<'gc> {
    Start,
    /// The comparator's (or `__lt`'s) answer.
    Less(bool),
    /// The queued read's value, from `__index`.
    Got(Value<'gc>),
    /// The queued write went through `__newindex`.
    Set,
}

/// PUC-Lua's `auxsort` over `[1, n]`, down to the order of every read, write
/// and comparison, as a state machine in the window: comparisons and the
/// reads and writes that need no call run in the loop; the rest return a
/// `CallThen` whose continuation comes back here. The one divergence: no
/// randomized pivot for badly unbalanced large partitions
/// (`l_randomizePivot` is clock-seeded anyway). The pending ranges stand in
/// for recursion: the smaller side is sorted first, the larger one queued.
fn sort_drive<'gc>(
    ctx: Context<'gc>,
    mut stack: Stack<'gc, '_>,
    resume: SortResume<'gc>,
) -> NativeOut {
    let mc = ctx.mutation();
    let tv = stack.get(0);
    let comp = stack.get(1);
    let int = |stack: &Stack<'gc, '_>, i: usize| stack.get(i).get_small().unwrap_or(0);
    let (mut lo, mut up) = (int(&stack, LO) as usize, int(&stack, UP) as usize);
    let (mut i, mut j, mut p) = (
        int(&stack, I) as usize,
        int(&stack, J) as usize,
        int(&stack, P) as usize,
    );
    let mut phase = int(&stack, PHASE);
    let mut nq = int(&stack, NQ) as usize;
    let mut r = None;
    match resume {
        SortResume::Start => {}
        SortResume::Less(b) => r = Some(b),
        SortResume::Got(v) => {
            let s = int(&stack, QUEUE + 2) as usize;
            stack.as_mut_slice()[s] = v;
            pop_op(&mut stack, &mut nq);
        }
        SortResume::Set => pop_op(&mut stack, &mut nq),
    }
    // Save the state and make the call `[$f, $args..]`.
    macro_rules! call {
        ($f:expr, [$($arg:expr),*]) => {{
            for (s, v) in [(LO, lo), (UP, up), (I, i), (J, j), (P, p), (NQ, nq)] {
                stack.as_mut_slice()[s] = Value::small(v as i32);
            }
            stack.as_mut_slice()[PHASE] = Value::small(phase);
            let at = stack.len();
            stack.as_mut_slice()[AT] = Value::small(at as i32);
            stack.extend([$f, $($arg),*]);
            return NativeOut::call_then(at, cont::SORT, Protect::No, OnOk::Cont);
        }};
    }
    // Queue a read of `t[$k]` into slot `$s`, or a write the other way.
    macro_rules! op {
        ($kind:expr, $k:expr, $s:expr) => {{
            let q = QUEUE + 3 * nq;
            let slots = stack.as_mut_slice();
            slots[q] = Value::small($kind);
            slots[q + 1] = Value::small($k as i32);
            slots[q + 2] = Value::small($s as i32);
            nq += 1;
        }};
    }
    // Whether the value in slot `$x` sorts before the one in `$y`.
    macro_rules! less {
        ($x:expr, $y:expr) => {{
            match r.take() {
                Some(b) => b,
                None => {
                    let (x, y) = (stack.get($x), stack.get($y));
                    if !comp.is_nil() {
                        call!(comp, [x, y]);
                    }
                    match prim_lt(x, y) {
                        Some(b) => b,
                        None => {
                            let m = binop_metamethod(ctx, x, y, MetamethodBits::LT);
                            if m.is_nil() {
                                let msg = util::compare_error_msg(x, y);
                                return NativeOut::error(util::runtime_error(ctx, &msg));
                            }
                            call!(m, [x, y]);
                        }
                    }
                }
            }
        }};
    }
    loop {
        // The queued reads and writes come first, in order. A comparator may
        // have given the table metamethods, so check each time.
        while nq > 0 {
            let (kind, k, s) = (
                int(&stack, QUEUE),
                int(&stack, QUEUE + 1),
                int(&stack, QUEUE + 2) as usize,
            );
            let key = Value::integer(mc, k as i64);
            let raw = tv.get_table().filter(|t| {
                let what = if kind == OP_GET { TAB_R } else { TAB_W };
                !t.shape().has_mm(what)
            });
            if kind == OP_GET {
                // `walk_index_chain` starts from a raw miss.
                let v = tv.get_table().map_or(Value::nil(), |t| t.raw_get(key));
                let v = match raw {
                    Some(_) => v,
                    None if !v.is_nil() => v,
                    None => match walk_index_chain(ctx, tv, key) {
                        IndexChain::Resolved(v) => v,
                        IndexChain::Invoke { func, receiver } => {
                            call!(Value::function(func), [receiver, key]);
                        }
                        IndexChain::NotIndexable(v) => {
                            let msg = format!("attempt to index a {} value", v.type_name());
                            return NativeOut::error(util::runtime_error(ctx, &msg));
                        }
                        IndexChain::Exhausted => {
                            let msg = "'__index' chain too long; possible loop";
                            return NativeOut::error(util::runtime_error(ctx, msg));
                        }
                    },
                };
                stack.as_mut_slice()[s] = v;
            } else {
                let v = stack.get(s);
                match raw {
                    Some(t) => t.raw_set(ctx, key, v),
                    None => match walk_newindex_chain(ctx, tv, key) {
                        NewIndexChain::RawSet(target) => target.raw_set(ctx, key, v),
                        NewIndexChain::Invoke { func, receiver } => {
                            call!(Value::function(func), [receiver, key, v]);
                        }
                        NewIndexChain::NotIndexable(v) => {
                            let msg = format!("attempt to index a {} value", v.type_name());
                            return NativeOut::error(util::runtime_error(ctx, &msg));
                        }
                        NewIndexChain::Exhausted => {
                            let msg = "'__newindex' chain too long; possible loop";
                            return NativeOut::error(util::runtime_error(ctx, msg));
                        }
                    },
                }
            }
            pop_op(&mut stack, &mut nq);
        }
        match phase {
            PH_RANGE => {
                if lo < up {
                    op!(OP_GET, lo, S0);
                    op!(OP_GET, up, S1);
                    phase = PH_A;
                    continue;
                }
                // The next pending range, or done.
                if stack.len() == STATE {
                    stack.clear();
                    return NativeOut::RETURN;
                }
                let n = stack.len();
                (lo, up) = (int(&stack, n - 2) as usize, int(&stack, n - 1) as usize);
                stack.truncate(n - 2);
            }
            PH_A => {
                // sort elements `lo`, `p`, and `up`
                if less!(S1, S0) {
                    op!(OP_SET, lo, S1);
                    op!(OP_SET, up, S0);
                }
                if up - lo == 1 {
                    (lo, up) = (1, 0);
                    phase = PH_RANGE;
                    continue;
                }
                p = (lo + up) / 2;
                op!(OP_GET, p, S0);
                op!(OP_GET, lo, S1);
                phase = PH_B;
            }
            PH_B => {
                if less!(S0, S1) {
                    op!(OP_SET, p, S1);
                    op!(OP_SET, lo, S0);
                    phase = PH_PIVOT;
                } else {
                    op!(OP_GET, up, S1);
                    phase = PH_C;
                }
            }
            PH_C => {
                if less!(S1, S0) {
                    op!(OP_SET, p, S1);
                    op!(OP_SET, up, S0);
                }
                phase = PH_PIVOT;
            }
            PH_PIVOT => {
                if up - lo == 2 {
                    (lo, up) = (1, 0);
                    phase = PH_RANGE;
                    continue;
                }
                // Pivot P stays in `S0` for the partition; `a[p]` and
                // `a[up - 1]` swap places.
                op!(OP_GET, p, S0);
                op!(OP_GET, up - 1, S1);
                op!(OP_SET, p, S1);
                op!(OP_SET, up - 1, S0);
                (i, j) = (lo, up - 1);
                phase = PH_D_NEXT;
            }
            PH_D_NEXT => {
                // repeat ++i while a[i] < P
                i += 1;
                op!(OP_GET, i, S1);
                phase = PH_D;
            }
            PH_D => {
                if less!(S1, S0) {
                    if i == up - 1 {
                        return NativeOut::error(Error::from_str(
                            ctx,
                            "invalid order function for sorting",
                        ));
                    }
                    phase = PH_D_NEXT;
                } else {
                    phase = PH_E_NEXT;
                }
            }
            PH_E_NEXT => {
                // repeat --j while P < a[j]
                j -= 1;
                op!(OP_GET, j, S2);
                phase = PH_E;
            }
            PH_E => {
                if less!(S0, S2) {
                    if j < i {
                        return NativeOut::error(Error::from_str(
                            ctx,
                            "invalid order function for sorting",
                        ));
                    }
                    phase = PH_E_NEXT;
                } else {
                    phase = PH_SWAP;
                }
            }
            PH_SWAP => {
                if j < i {
                    // no elements out of place: swap P into a[i], and sort
                    // the smaller side first, the larger one pending
                    op!(OP_SET, up - 1, S1);
                    op!(OP_SET, i, S0);
                    let p = i;
                    let (next, pend) = if p - lo < up - p {
                        ((lo, p - 1), (p + 1, up))
                    } else {
                        ((p + 1, up), (lo, p - 1))
                    };
                    stack.extend([Value::small(pend.0 as i32), Value::small(pend.1 as i32)]);
                    (lo, up) = next;
                    phase = PH_RANGE;
                } else {
                    op!(OP_SET, i, S2);
                    op!(OP_SET, j, S1);
                    phase = PH_D_NEXT;
                }
            }
            _ => unreachable!("sort phase {phase}"),
        }
    }
}

/// Drop the first queued read or write.
fn pop_op<'gc>(stack: &mut Stack<'gc, '_>, nq: &mut usize) {
    let slots = stack.as_mut_slice();
    slots.copy_within(QUEUE + 3..QUEUE + 3 * *nq, QUEUE);
    *nq -= 1;
}

pub(crate) fn sort_cont<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
    _status: Result<(), Error<'gc>>,
) -> NativeOut {
    let at = stack.get(AT).get_small().unwrap_or(0) as usize;
    let first = stack.get(at);
    stack.truncate(at);
    // A queued read or write was waiting on the call, else a comparison.
    let resume = match stack.get(NQ).get_small() {
        Some(0) => SortResume::Less(!first.is_falsy()),
        _ if stack.get(QUEUE).get_small() == Some(OP_GET) => SortResume::Got(first),
        _ => SortResume::Set,
    };
    sort_drive(ctx, stack, resume)
}

/// The default order's `<` on primitives (no string→number coercion).
fn prim_lt(x: Value<'_>, y: Value<'_>) -> Option<bool> {
    if let (Some(x), Some(y)) = (x.get_integer(), y.get_integer()) {
        Some(x < y)
    } else if let (Some(x), Some(y)) = (x.get_float(), y.get_float()) {
        Some(x < y)
    } else if let (Some(x), Some(y)) = (x.get_integer(), y.get_float()) {
        Some(num::lt_int_float(x, y))
    } else if let (Some(x), Some(y)) = (x.get_float(), y.get_integer()) {
        Some(num::lt_float_int(x, y))
    } else if let (Some(x), Some(y)) = (x.get_string(), y.get_string()) {
        Some(x < y)
    } else {
        None
    }
}

/// `unpack(t [, i [, j]])` — return `t[i]..t[j]` (`i` defaults to 1, `j` to
/// `#t`).
fn lua_unpack<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> NativeOut {
    let Some(t) = check_tab(ctx, stack.get(0), "unpack", 1, TAB_R | TAB_L)? else {
        return unpack_meta(ctx, &mut stack);
    };
    let Some((i, n)) = unpack_range(ctx, &stack, t.raw_len() as i64)? else {
        stack.clear();
        return NativeOut::RETURN;
    };
    let out = stack
        .replace_slots(n)
        .expect("unpack_range checked the room");
    for (k, slot) in out.iter_mut().enumerate() {
        *slot = t.raw_get(Value::integer(ctx.mutation(), i + k as i64));
    }
    NativeOut::RETURN
}

/// `unpack` on a value whose accesses may run metamethods.
#[cold]
#[inline(never)]
fn unpack_meta<'gc>(ctx: Context<'gc>, stack: &mut Stack<'gc, '_>) -> NativeOut {
    stack.spawn_action(ctx, |cx| async move {
        let len = util::len(&cx, 0).await?;
        let range = cx.try_enter(|ctx, stack| unpack_range(ctx, stack, len))?;
        let Some((i, n)) = range else {
            cx.enter(|_ctx, mut stack| stack.clear());
            return Ok(());
        };
        // Results go above `t`, which is dropped at the end.
        cx.enter(|_ctx, mut stack| stack.truncate(1));
        for k in 0..n {
            util::geti(&cx, 0, i + k as i64).await?;
        }
        cx.enter(|_ctx, mut stack| stack.remove(0));
        Ok(())
    })
}

/// `unpack`'s first index and count, `None` when empty; `len` is `#t`, the
/// default end. The count fits on the stack, so `i + k` for `k < n` can't overflow.
fn unpack_range<'gc>(
    ctx: Context<'gc>,
    stack: &Stack<'gc, '_>,
    len: i64,
) -> Result<Option<(i64, usize)>, Error<'gc>> {
    let i_arg = stack.get(1);
    let i = if i_arg.is_nil() {
        1
    } else {
        util::check_integer(ctx, i_arg, "unpack", 2)?
    };
    let j_arg = stack.get(2);
    let j = if j_arg.is_nil() {
        len
    } else {
        util::check_integer(ctx, j_arg, "unpack", 3)?
    };
    if i > j {
        return Ok(None);
    }
    // `n - 1` as unsigned, so a full-i64-span range can't overflow the
    // subtraction (PUC-Lua's `tunpack`).
    let n1 = (j as u64).wrapping_sub(i as u64);
    if n1 >= i32::MAX as u64 || !stack.check_stack(n1 as usize + 1) {
        return Err(Error::from_str(ctx, "too many results to unpack"));
    }
    Ok(Some((i, n1 as usize + 1)))
}
