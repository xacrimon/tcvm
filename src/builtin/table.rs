use crate::Context;
use crate::builtin::util;
use crate::env::{
    Error, Function, LuaString, MetamethodBits, NativeClosure, NativeFn, Stack, Table, Value,
};
use crate::lua::stash::Fetchable;
use crate::lua::{StashedError, StashedValue};
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
/// comparator raises "invalid order function for sorting".
fn lua_sort<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let n = match check_tab(ctx, stack.get(0), "sort", 1, TAB_R | TAB_W | TAB_L)? {
        Some(t) => {
            let n = t.raw_len() as i64;
            if !sort_args(ctx, &mut stack, n)? {
                stack.clear();
                return Ok(CallbackAction::Return);
            }
            Some(n)
        }
        None => None,
    };
    let seq = async_sequence(ctx.mutation(), move |_locals, mut seq| async move {
        let n = match n {
            Some(n) => n,
            None => {
                let n = util::len(&mut seq, 0).await?;
                if !seq.try_enter(|ctx, _locals, _exec, mut stack| sort_args(ctx, &mut stack, n))? {
                    seq.enter(|_ctx, _locals, _exec, mut stack| stack.clear());
                    return Ok(SequenceReturn::Return);
                }
                n
            }
        };
        sort_run(&mut seq, n as usize).await?;
        seq.enter(|_ctx, _locals, _exec, mut stack| stack.clear());
        Ok(SequenceReturn::Return)
    });
    Ok(CallbackAction::sequence(seq))
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
        return Err(Error::from_str(
            ctx,
            "bad argument #1 to 'sort' (array too big)",
        ));
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

// `auxsort`'s working values live in fixed window slots above `[t, comp]`:
// where the reference pushes and pops, the port keeps each value at the depth
// it would occupy. Moving `top` per comparison chained loads and stores
// through it and cost the sort about half its speed.
const S0: usize = 2;
const S1: usize = 3;
const S2: usize = 4;

/// `sort_comp` after [`fetch`]ing `ks`: whether the value in slot `a` sorts
/// before the one in slot `b`. A macro, not an `async fn`: building, resuming
/// and dropping a future per comparison cost more than the comparison, and
/// every nesting level is walked again when a comparator call returns.
macro_rules! sort_less {
    ($seq:expr, $comp:expr, $ks:expr, $a:expr, $b:expr) => {{
        let mut ks: &[(usize, usize)] = $ks;
        let step = loop {
            if let Some(step) = less_step($seq, $comp.is_some(), ks, $a, $b)? {
                break step;
            }
            fetch($seq, ks).await?;
            ks = &[];
        };
        match step {
            Less::Ready(r) => r,
            Less::Comp(bottom) => call_truthy($seq, $comp.as_ref().unwrap(), bottom).await?,
            Less::Lt(m, bottom) => call_truthy($seq, &m, bottom).await?,
        }
    }};
}

/// PUC-Lua's `auxsort` over `[1, n]`, down to the order of every read, write
/// and comparison. The one divergence: no randomized pivot for badly
/// unbalanced large partitions (`l_randomizePivot` is clock-seeded anyway).
/// The explicit `pending` stack stands in for recursion: the smaller side is
/// sorted first, the larger one queued.
async fn sort_run(seq: &mut AsyncSequence, n: usize) -> Result<(), StashedError> {
    let comp = seq.enter(|ctx, locals, _exec, mut stack| {
        stack.extend([Value::nil(); 3]);
        stack
            .get(1)
            .get_function()
            .map(|f| locals.stash(ctx.mutation(), f))
    });
    let comp = &comp;
    let mut pending = Vec::new();
    let (mut lo, mut up) = (1, n);
    loop {
        while lo < up {
            // sort elements `lo`, `p`, and `up`
            if sort_less!(seq, comp, &[(lo, S0), (up, S1)], S1, S0) {
                store(seq, &[(S1, lo), (S0, up)]).await?;
            }
            if up - lo == 1 {
                break;
            }
            let p = (lo + up) / 2;
            if sort_less!(seq, comp, &[(p, S0), (lo, S1)], S0, S1) {
                store(seq, &[(S1, p), (S0, lo)]).await?;
            } else if sort_less!(seq, comp, &[(up, S1)], S1, S0) {
                store(seq, &[(S1, p), (S0, up)]).await?;
            }
            if up - lo == 2 {
                break;
            }
            // Pivot P stays in `S0` for the partition; `a[p]` and `a[up - 1]`
            // swap places.
            fetch(seq, &[(p, S0), (up - 1, S1)]).await?;
            store(seq, &[(S1, p), (S0, up - 1)]).await?;
            // `partition`: reorder so `a[lo .. p - 1] <= a[p] == P <= a[p + 1 .. up]`.
            let (mut i, mut j) = (lo, up - 1);
            let p = loop {
                // repeat ++i while a[i] < P
                loop {
                    i += 1;
                    if !sort_less!(seq, comp, &[(i, S1)], S1, S0) {
                        break;
                    }
                    if i == up - 1 {
                        return invalid_order(seq);
                    }
                }
                // repeat --j while P < a[j]
                loop {
                    j -= 1;
                    if !sort_less!(seq, comp, &[(j, S2)], S0, S2) {
                        break;
                    }
                    if j < i {
                        return invalid_order(seq);
                    }
                }
                if j < i {
                    // no elements out of place: swap P into a[i]
                    store(seq, &[(S1, up - 1), (S0, i)]).await?;
                    break i;
                }
                store(seq, &[(S2, i), (S1, j)]).await?;
            };
            if p - lo < up - p {
                pending.push((p + 1, up));
                up = p - 1;
            } else {
                pending.push((lo, p - 1));
                lo = p + 1;
            }
        }
        match pending.pop() {
            Some((l, u)) => (lo, up) = (l, u),
            None => return Ok(()),
        }
    }
}

// While `t` lacks the metamethod involved (rechecked each time: a comparator
// may set one), a batch of reads or writes, and a read with its primitive
// comparison, takes a single `enter`.

/// Read `t[k]` into slot `s` for each `(k, s)`, if `t` needs no `__index`.
#[inline(always)]
fn fetch_raw<'gc>(ctx: Context<'gc>, stack: &mut Stack<'gc, '_>, ks: &[(usize, usize)]) -> bool {
    if ks.is_empty() {
        return true;
    }
    let Some(t) = stack.get(0).get_table() else {
        return false;
    };
    if t.shape().has_mm(TAB_R) {
        return false;
    }
    for &(k, s) in ks {
        stack.as_mut_slice()[s] = t.raw_get(Value::integer(ctx.mutation(), k as i64));
    }
    true
}

/// Read `t[k]` into slot `s` for each `(k, s)`, in order.
async fn fetch(seq: &mut AsyncSequence, ks: &[(usize, usize)]) -> Result<(), StashedError> {
    if seq.enter(|ctx, _locals, _exec, mut stack| fetch_raw(ctx, &mut stack, ks)) {
        return Ok(());
    }
    for &(k, s) in ks {
        util::geti(seq, 0, k as i64).await?;
        seq.enter(|_ctx, _locals, _exec, mut stack| {
            let v = stack.pop();
            stack.as_mut_slice()[s] = v;
        });
    }
    Ok(())
}

/// Write slot `s` into `t[k]` for each `(s, k)`, in order.
async fn store(seq: &mut AsyncSequence, sk: &[(usize, usize)]) -> Result<(), StashedError> {
    let raw = seq.enter(|ctx, _locals, _exec, stack| {
        let Some(t) = stack.get(0).get_table() else {
            return false;
        };
        if t.shape().has_mm(TAB_W) {
            return false;
        }
        for &(s, k) in sk {
            t.raw_set(ctx, Value::integer(ctx.mutation(), k as i64), stack.get(s));
        }
        true
    });
    if raw {
        return Ok(());
    }
    for &(s, k) in sk {
        seq.enter(|_ctx, _locals, _exec, mut stack| stack.push(stack.get(s)));
        util::seti(seq, 0, k as i64).await?;
    }
    Ok(())
}

/// The synchronous part of [`sort_less!`].
enum Less {
    Ready(bool),
    /// Call the comparator with the arguments staged at this slot.
    Comp(usize),
    /// Call this `__lt` metamethod with the arguments staged at this slot.
    Lt(StashedValue, usize),
}

/// `None` when reading `ks` needs `__index`.
#[inline(always)]
fn less_step(
    seq: &mut AsyncSequence,
    has_comp: bool,
    ks: &[(usize, usize)],
    a: usize,
    b: usize,
) -> Result<Option<Less>, StashedError> {
    seq.try_enter(|ctx, locals, _exec, mut stack| {
        if !fetch_raw(ctx, &mut stack, ks) {
            return Ok(None);
        }
        let (x, y) = (stack.get(a), stack.get(b));
        let bottom = stack.len();
        if has_comp {
            stack.extend([x, y]);
            return Ok(Some(Less::Comp(bottom)));
        }
        // Default order follows the `<` operator (no string→number coercion).
        let prim = if let (Some(x), Some(y)) = (x.get_integer(), y.get_integer()) {
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
        };
        if let Some(r) = prim {
            return Ok(Some(Less::Ready(r)));
        }
        let m = binop_metamethod(ctx, x, y, ctx.symbols().mm_lt);
        if m.is_nil() {
            return Err(Error::from_str(ctx, &util::compare_error_msg(x, y)));
        }
        stack.extend([x, y]);
        Ok(Some(Less::Lt(locals.stash(ctx.mutation(), m), bottom)))
    })
}

/// Call `f` with the arguments staged at `bottom`; the truthiness of its
/// first result.
async fn call_truthy<F>(seq: &mut AsyncSequence, f: &F, bottom: usize) -> Result<bool, StashedError>
where
    F: Fetchable,
    for<'gc> F::Fetched<'gc>: Into<Value<'gc>>,
{
    seq.call(f, bottom).await?;
    Ok(seq.enter(|_ctx, _locals, _exec, mut stack| {
        let r = !stack.get(bottom).is_falsy();
        stack.truncate(bottom);
        r
    }))
}

/// Raise "invalid order function for sorting".
fn invalid_order(seq: &mut AsyncSequence) -> Result<(), StashedError> {
    seq.try_enter(|ctx, _locals, _exec, _stack| {
        Err(Error::from_str(ctx, "invalid order function for sorting"))
    })
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
