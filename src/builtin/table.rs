use crate::Context;
use crate::builtin::util;
use crate::env::{
    Error, Function, LuaString, MetamethodBits, NativeClosure, NativeFn, Stack, Table, Value,
};
use crate::lua::{StashedError, StashedFunction, StashedTable, StashedValue};
use crate::vm::async_sequence::{AsyncSequence, SequenceReturn, async_sequence};
use crate::vm::interp::binop_metamethod;
use crate::vm::num;
use crate::vm::sequence::CallbackAction;

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
    let fns: &[(&str, NativeFn)] = &[
        ("concat", lua_concat),
        ("create", lua_create),
        ("insert", lua_insert),
        ("move", lua_move),
        ("pack", lua_pack),
        ("remove", lua_remove),
        ("sort", lua_sort),
        ("unpack", lua_unpack),
    ];

    let lib = Table::new(ctx);
    for &(name, handler) in fns {
        let handler = Function::new_native(ctx.mutation(), handler, Box::new([]));
        let key = Value::string(LuaString::new(ctx, name.as_bytes()));
        lib.raw_set(ctx, key, Value::function(handler));
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
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let Some(t) = check_tab(ctx, stack.get(0), "concat", 1, TAB_R | TAB_L)? else {
        return Ok(concat_meta(ctx));
    };
    let (sep, mut i, last) = concat_args(ctx, &stack, t.raw_len() as i64)?;
    let mut out = Vec::new();
    // `i < last` rather than `i <= last`, so `i` never steps past `last`.
    while i < last {
        add_field(
            ctx,
            &mut out,
            t.raw_get(Value::integer(ctx.mutation(), i)),
            i,
        )?;
        out.extend_from_slice(&sep);
        i += 1;
    }
    if i == last {
        add_field(
            ctx,
            &mut out,
            t.raw_get(Value::integer(ctx.mutation(), i)),
            i,
        )?;
    }
    stack.ret1(Value::string(LuaString::new(ctx, &out)));
    Ok(CallbackAction::Return)
}

/// `concat` on a value whose accesses may run metamethods.
#[cold]
#[inline(never)]
fn concat_meta<'gc>(ctx: Context<'gc>) -> CallbackAction<'gc> {
    let seq = async_sequence(ctx.mutation(), |_locals, mut seq| async move {
        let last = util::len(&mut seq, 0).await?;
        let (sep, mut i, last) =
            seq.try_enter(|ctx, _locals, _exec, stack| concat_args(ctx, &stack, last))?;
        let mut out = Vec::new();
        while i < last {
            concat_field(&mut seq, &mut out, i).await?;
            out.extend_from_slice(&sep);
            i += 1;
        }
        if i == last {
            concat_field(&mut seq, &mut out, i).await?;
        }
        seq.enter(|ctx, _locals, _exec, mut stack| {
            stack.ret1(Value::string(LuaString::new(ctx, &out)))
        });
        Ok(SequenceReturn::Return)
    });
    CallbackAction::sequence(seq)
}

/// `concat`'s separator and range; `last` is `#t`, the default end.
fn concat_args<'gc>(
    ctx: Context<'gc>,
    stack: &Stack<'gc, '_>,
    last: i64,
) -> Result<(Vec<u8>, i64, i64), Error<'gc>> {
    let sep_arg = stack.get(1);
    let sep = if sep_arg.is_nil() {
        Vec::new()
    } else if let Some(s) = sep_arg.get_string() {
        s.as_bytes().to_vec()
    } else if sep_arg.get_integer().is_some() || sep_arg.get_float().is_some() {
        util::basic_tostring(ctx, sep_arg).as_bytes().to_vec()
    } else {
        return Err(util::type_error(ctx, "concat", 2, "string", Some(sep_arg)));
    };
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

async fn concat_field(
    seq: &mut AsyncSequence,
    out: &mut Vec<u8>,
    i: i64,
) -> Result<(), StashedError> {
    util::geti(seq, 0, i).await?;
    seq.try_enter(|ctx, _locals, _exec, mut stack| add_field(ctx, out, stack.pop(), i))
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
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let n = util::check_integer(ctx, stack.get(0), "create", 1)?;
    let m_arg = stack.get(1);
    let m = if m_arg.is_nil() {
        0
    } else {
        util::check_integer(ctx, m_arg, "create", 2)?
    };
    if n < 0 {
        return Err(Error::from_str(
            ctx,
            "bad argument #1 to 'create' (out of range)",
        ));
    }
    if m < 0 {
        return Err(Error::from_str(
            ctx,
            "bad argument #2 to 'create' (out of range)",
        ));
    }
    stack.ret1(Value::table(Table::new(ctx)));
    Ok(CallbackAction::Return)
}

/// `insert(t, [pos,] value)` — append `value`, or insert it at `pos`, shifting
/// later elements up by one.
fn lua_insert<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let Some(t) = check_tab(ctx, stack.get(0), "insert", 1, TAB_R | TAB_W | TAB_L)? else {
        return Ok(insert_meta(ctx));
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
    Ok(CallbackAction::Return)
}

/// `insert` on a value whose accesses may run metamethods.
#[cold]
#[inline(never)]
fn insert_meta<'gc>(ctx: Context<'gc>) -> CallbackAction<'gc> {
    let seq = async_sequence(ctx.mutation(), |_locals, mut seq| async move {
        let e = util::len(&mut seq, 0).await?.wrapping_add(1);
        let pos = seq.try_enter(|ctx, _locals, _exec, stack| insert_pos(ctx, &stack, e))?;
        let mut i = e;
        while i > pos {
            util::geti(&mut seq, 0, i - 1).await?;
            util::seti(&mut seq, 0, i).await?;
            i -= 1;
        }
        // The value argument is on top.
        util::seti(&mut seq, 0, pos).await?;
        seq.enter(|_ctx, _locals, _exec, mut stack| stack.clear());
        Ok(SequenceReturn::Return)
    });
    CallbackAction::sequence(seq)
}

/// `insert`'s target position; `e` is `#t + 1`, the default.
fn insert_pos<'gc>(ctx: Context<'gc>, stack: &Stack<'gc, '_>, e: i64) -> Result<i64, Error<'gc>> {
    match stack.len() {
        2 => Ok(e),
        3 => {
            let pos = util::check_integer(ctx, stack.get(1), "insert", 2)?;
            // `pos` in `[1, e]`, compared unsigned as the reference does.
            if (pos as u64).wrapping_sub(1) >= e as u64 {
                return Err(Error::from_str(
                    ctx,
                    "bad argument #2 to 'insert' (position out of bounds)",
                ));
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
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let f = util::check_integer(ctx, stack.get(1), "move", 2)?;
    let e = util::check_integer(ctx, stack.get(2), "move", 3)?;
    let t = util::check_integer(ctx, stack.get(3), "move", 4)?;
    let tt = if stack.get(4).is_nil() { 0 } else { 4 };
    let src = check_tab(ctx, stack.get(0), "move", 1, TAB_R)?;
    let dst = check_tab(ctx, stack.get(tt), "move", tt + 1, TAB_W)?;
    let (a1, a2) = (stack.get(0), stack.get(tt));
    if e < f {
        stack.ret1(a2);
        return Ok(CallbackAction::Return);
    }
    // PUC-Lua's two bounds: the element count `e - f + 1` must fit a Lua
    // integer (else `e - f` itself overflows), and the destination range
    // `t .. t + n - 1` must not wrap past maxinteger.
    if !(f > 0 || e < i64::MAX + f) {
        return Err(Error::from_str(
            ctx,
            "bad argument #3 to 'move' (too many elements to move)",
        ));
    }
    let n = e - f + 1;
    if t > i64::MAX - n + 1 {
        return Err(Error::from_str(
            ctx,
            "bad argument #4 to 'move' (destination wrap around)",
        ));
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
            eq_mm = binop_metamethod(ctx, a1, a2, ctx.symbols().mm_eq);
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
        return Ok(CallbackAction::Return);
    }
    Ok(move_meta(ctx, f, n, t, tt, forward, eq_mm))
}

/// `move` of `n` elements when an access may run metamethods or the copy
/// direction needs `eq_mm` (`forward` is `None`).
#[cold]
#[inline(never)]
fn move_meta<'gc>(
    ctx: Context<'gc>,
    f: i64,
    n: i64,
    t: i64,
    tt: usize,
    forward: Option<bool>,
    eq_mm: Value<'gc>,
) -> CallbackAction<'gc> {
    let mc = ctx.mutation();
    let seq = async_sequence(mc, move |locals, mut seq| {
        let eq_mm = (!eq_mm.is_nil()).then(|| locals.stash(mc, eq_mm));
        async move {
            let forward = match forward {
                Some(forward) => forward,
                None => {
                    let bottom = seq.enter(|_ctx, _locals, _exec, mut stack| {
                        let bottom = stack.len();
                        stack.extend([stack.get(0), stack.get(tt)]);
                        bottom
                    });
                    seq.call(eq_mm.as_ref().unwrap(), bottom).await?;
                    seq.enter(|_ctx, _locals, _exec, mut stack| {
                        let equal = !stack.get(bottom).is_falsy();
                        stack.truncate(bottom);
                        !equal
                    })
                }
            };
            for k in 0..n {
                let i = if forward { k } else { n - 1 - k };
                util::geti(&mut seq, 0, f + i).await?;
                util::seti(&mut seq, tt, t + i).await?;
            }
            seq.enter(|_ctx, _locals, _exec, mut stack| stack.ret1(stack.get(tt)));
            Ok(SequenceReturn::Return)
        }
    });
    CallbackAction::sequence(seq)
}

/// `pack(...)` — collect all arguments into a new table with field `n` set to
/// the argument count.
fn lua_pack<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
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
        Value::string(LuaString::new(ctx, b"n")),
        Value::integer(ctx.mutation(), n as i64),
    );
    stack.ret1(Value::table(t));
    Ok(CallbackAction::Return)
}

/// `remove(t [, pos])` — remove and return `t[pos]` (default `#t`), shifting
/// later elements down by one.
fn lua_remove<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let Some(t) = check_tab(ctx, stack.get(0), "remove", 1, TAB_R | TAB_W | TAB_L)? else {
        return Ok(remove_meta(ctx));
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
    Ok(CallbackAction::Return)
}

/// `remove` on a value whose accesses may run metamethods.
#[cold]
#[inline(never)]
fn remove_meta<'gc>(ctx: Context<'gc>) -> CallbackAction<'gc> {
    let seq = async_sequence(ctx.mutation(), |_locals, mut seq| async move {
        let size = util::len(&mut seq, 0).await?;
        let mut pos = seq.try_enter(|ctx, _locals, _exec, stack| remove_pos(ctx, &stack, size))?;
        // The result stays on the stack below the shuffling.
        util::geti(&mut seq, 0, pos).await?;
        while pos < size {
            util::geti(&mut seq, 0, pos + 1).await?;
            util::seti(&mut seq, 0, pos).await?;
            pos += 1;
        }
        seq.enter(|_ctx, _locals, _exec, mut stack| stack.push(Value::nil()));
        util::seti(&mut seq, 0, pos).await?;
        seq.enter(|_ctx, _locals, _exec, mut stack| {
            let result = stack.pop();
            stack.ret1(result)
        });
        Ok(SequenceReturn::Return)
    });
    CallbackAction::sequence(seq)
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
        return Err(Error::from_str(
            ctx,
            "bad argument #2 to 'remove' (position out of bounds)",
        ));
    }
    Ok(pos)
}

/// `sort(t [, comp])` — sort `t[1..#t]` in place. With no comparator the default
/// `<` order is used (numbers, strings, or an `__lt` metamethod); otherwise
/// `comp(a, b)` must return true when `a` should precede `b`. An inconsistent
/// comparator raises "invalid order function for sorting". Because a comparator
/// (or an `__lt` metamethod) re-enters the interpreter, the work runs as an
/// async `Sequence`; a comparator-free primitive sort still completes in a
/// single poll without ever suspending.
fn lua_sort<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let t = stack
        .get(0)
        .get_table()
        .ok_or_else(|| util::type_error(ctx, "sort", 1, "table", Some(stack.get(0))))?;
    let n = t.raw_len();
    if n <= 1 {
        // Nothing to do — and, matching Lua, the comparator argument is not even
        // type-checked for a trivial array.
        stack.replace(&[]);
        return Ok(CallbackAction::Return);
    }
    if n >= i32::MAX as usize {
        return Err(Error::from_str(
            ctx,
            "bad argument #1 to 'sort' (array too big)",
        ));
    }
    let comp_arg = stack.get(1);
    let comp = if comp_arg.is_nil() {
        None
    } else if let Some(f) = comp_arg.get_function() {
        Some(f)
    } else {
        return Err(util::type_error(ctx, "sort", 2, "function", Some(comp_arg)));
    };

    let mc = ctx.mutation();
    let seq = async_sequence(mc, move |locals, seq| {
        let t = locals.stash(mc, t);
        let comp = comp.map(|f| locals.stash(mc, f));
        async move {
            let mut seq = seq;
            sort_run(&mut seq, t, comp, n).await?;
            seq.enter(|_ctx, _locals, _exec, mut stack| stack.replace(&[]));
            Ok(SequenceReturn::Return)
        }
    });
    Ok(CallbackAction::sequence(seq))
}

/// Iterative quicksort (median-of-3 pivot) over the 1-based range `[1, n]`,
/// faithfully mirroring PUC-Lua's `auxsort`/`partition` — including its
/// detection of an inconsistent comparator. Recurses on the smaller side and
/// loops on the larger, so the explicit work stack stays shallow.
async fn sort_run(
    seq: &mut AsyncSequence,
    t: StashedTable,
    comp: Option<StashedFunction>,
    n: usize,
) -> Result<(), StashedError> {
    let mut work = vec![(1usize, n)];
    while let Some((mut lo, mut up)) = work.pop() {
        while lo < up {
            // order the endpoints: a[lo] <= a[up]
            if sort_less(seq, &t, &comp, up, lo).await? {
                sort_swap(seq, &t, lo, up);
            }
            if up - lo == 1 {
                break;
            }
            let p = (lo + up) / 2;
            // median of 3: leave a[lo] <= a[p] <= a[up]
            if sort_less(seq, &t, &comp, p, lo).await? {
                sort_swap(seq, &t, p, lo);
            } else if sort_less(seq, &t, &comp, up, p).await? {
                sort_swap(seq, &t, p, up);
            }
            if up - lo == 2 {
                break;
            }
            // stash the pivot at a[up-1], then partition (lo, up)
            sort_swap(seq, &t, p, up - 1);
            let piv = up - 1;
            let mut i = lo;
            let mut j = up - 1;
            let part = loop {
                // advance i past elements strictly less than the pivot
                loop {
                    i += 1;
                    if !sort_less(seq, &t, &comp, i, piv).await? {
                        break;
                    }
                    if i == up - 1 {
                        return Err(invalid_order_err(seq));
                    }
                }
                // retreat j past elements strictly greater than the pivot
                loop {
                    j -= 1;
                    if !sort_less(seq, &t, &comp, piv, j).await? {
                        break;
                    }
                    if j < i {
                        return Err(invalid_order_err(seq));
                    }
                }
                if j < i {
                    sort_swap(seq, &t, piv, i); // move pivot into place
                    break i;
                }
                sort_swap(seq, &t, i, j);
            };
            // recurse on the smaller interval, loop on the larger
            if part - lo < up - part {
                work.push((lo, part - 1));
                lo = part + 1;
            } else {
                work.push((part + 1, up));
                up = part - 1;
            }
        }
    }
    Ok(())
}

/// `a[xi] < a[yi]` under the active order. Primitive number/string pairs resolve
/// without re-entering the VM; a comparator or `__lt` metamethod is called via
/// the sequence (the only suspending case).
async fn sort_less(
    seq: &mut AsyncSequence,
    t: &StashedTable,
    comp: &Option<StashedFunction>,
    xi: usize,
    yi: usize,
) -> Result<bool, StashedError> {
    enum Plan {
        Ready(bool),
        CallComp,
        CallMeta(StashedValue),
    }
    let plan = seq.try_enter(|ctx, locals, _exec, mut stack| {
        let tbl = locals.fetch(ctx.mutation(), t);
        let a = tbl.raw_get(Value::integer(ctx.mutation(), xi as i64));
        let b = tbl.raw_get(Value::integer(ctx.mutation(), yi as i64));
        if comp.is_some() {
            stack.replace(&[a, b]);
            return Ok(Plan::CallComp);
        }
        // Default order follows the `<` operator (no string→number coercion).
        let prim = if let (Some(x), Some(y)) = (a.get_integer(), b.get_integer()) {
            Some(x < y)
        } else if let (Some(x), Some(y)) = (a.get_float(), b.get_float()) {
            Some(x < y)
        } else if let (Some(x), Some(y)) = (a.get_integer(), b.get_float()) {
            Some(num::lt_int_float(x, y))
        } else if let (Some(x), Some(y)) = (a.get_float(), b.get_integer()) {
            Some(num::lt_float_int(x, y))
        } else if let (Some(x), Some(y)) = (a.get_string(), b.get_string()) {
            Some(x < y)
        } else {
            None
        };
        if let Some(r) = prim {
            return Ok(Plan::Ready(r));
        }
        let m = binop_metamethod(ctx, a, b, ctx.symbols().mm_lt);
        if m.is_nil() {
            return Err(Error::from_str(ctx, &util::compare_error_msg(a, b)));
        }
        stack.replace(&[a, b]);
        Ok(Plan::CallMeta(locals.stash(ctx.mutation(), m)))
    })?;
    match plan {
        Plan::Ready(r) => Ok(r),
        Plan::CallComp => {
            seq.call(comp.as_ref().unwrap(), 0).await?;
            Ok(sort_truthy(seq))
        }
        Plan::CallMeta(f) => {
            seq.call(&f, 0).await?;
            Ok(sort_truthy(seq))
        }
    }
}

/// Swap `a[x]` and `a[y]` in place (raw, no metamethods).
fn sort_swap(seq: &mut AsyncSequence, t: &StashedTable, x: usize, y: usize) {
    seq.enter(|ctx, locals, _exec, _stack| {
        let tbl = locals.fetch(ctx.mutation(), t);
        let kx = Value::integer(ctx.mutation(), x as i64);
        let ky = Value::integer(ctx.mutation(), y as i64);
        let vx = tbl.raw_get(kx);
        let vy = tbl.raw_get(ky);
        tbl.raw_set(ctx, kx, vy);
        tbl.raw_set(ctx, ky, vx);
    });
}

/// Truthiness (anything but `nil`/`false`) of the comparator's first result,
/// left at the sequence window's bottom by the preceding `call`.
fn sort_truthy(seq: &mut AsyncSequence) -> bool {
    seq.enter(|_ctx, _locals, _exec, stack| {
        let v = stack.get(0);
        !(v.is_nil() || v.get_boolean() == Some(false))
    })
}

/// Build the "invalid order function for sorting" error as a stashed error.
fn invalid_order_err(seq: &mut AsyncSequence) -> StashedError {
    seq.try_enter(|ctx, _locals, _exec, _stack| {
        Result::<(), _>::Err(Error::from_str(ctx, "invalid order function for sorting"))
    })
    .unwrap_err()
}

/// `unpack(t [, i [, j]])` — return `t[i]..t[j]` (`i` defaults to 1, `j` to
/// `#t`).
fn lua_unpack<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let Some(t) = check_tab(ctx, stack.get(0), "unpack", 1, TAB_R | TAB_L)? else {
        return Ok(unpack_meta(ctx));
    };
    let Some((mut i, e)) = unpack_range(ctx, &stack, t.raw_len() as i64)? else {
        stack.clear();
        return Ok(CallbackAction::Return);
    };
    let mut out = Vec::with_capacity(e.wrapping_sub(i) as usize + 1);
    // Iterate `i < e` then push `t[e]` separately, so `i += 1` never steps
    // past i64::MAX when `e == i64::MAX`.
    while i < e {
        out.push(t.raw_get(Value::integer(ctx.mutation(), i)));
        i += 1;
    }
    out.push(t.raw_get(Value::integer(ctx.mutation(), e)));
    stack.replace(&out);
    Ok(CallbackAction::Return)
}

/// `unpack` on a value whose accesses may run metamethods.
#[cold]
#[inline(never)]
fn unpack_meta<'gc>(ctx: Context<'gc>) -> CallbackAction<'gc> {
    let seq = async_sequence(ctx.mutation(), |_locals, mut seq| async move {
        let len = util::len(&mut seq, 0).await?;
        let range = seq.try_enter(|ctx, _locals, _exec, stack| unpack_range(ctx, &stack, len))?;
        let Some((mut i, e)) = range else {
            seq.enter(|_ctx, _locals, _exec, mut stack| stack.clear());
            return Ok(SequenceReturn::Return);
        };
        // Results go above `t`, which is dropped at the end.
        seq.enter(|_ctx, _locals, _exec, mut stack| stack.truncate(1));
        while i < e {
            util::geti(&mut seq, 0, i).await?;
            i += 1;
        }
        util::geti(&mut seq, 0, e).await?;
        seq.enter(|_ctx, _locals, _exec, mut stack| stack.remove(0));
        Ok(SequenceReturn::Return)
    });
    CallbackAction::sequence(seq)
}

/// `unpack`'s range, `None` when empty; `len` is `#t`, the default end.
fn unpack_range<'gc>(
    ctx: Context<'gc>,
    stack: &Stack<'gc, '_>,
    len: i64,
) -> Result<Option<(i64, i64)>, Error<'gc>> {
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
    // subtraction (PUC-Lua's `tunpack`). Cap the count at i32::MAX results.
    if (j as u64).wrapping_sub(i as u64) >= i32::MAX as u64 {
        return Err(Error::from_str(ctx, "too many results to unpack"));
    }
    Ok(Some((i, j)))
}
