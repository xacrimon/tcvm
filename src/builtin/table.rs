use crate::Context;
use crate::builtin::util;
use crate::env::{Error, Function, LuaString, NativeContext, NativeFn, Stack, Table, Value};
use crate::lua::{StashedError, StashedFunction, StashedTable};
use crate::vm::async_sequence::{AsyncSequence, SequenceReturn, async_sequence};
use crate::vm::sequence::CallbackAction;

/// Fetch argument 1 as a table or raise the standard bad-argument error.
fn check_table<'gc>(
    ctx: Context<'gc>,
    v: Value<'gc>,
    fname: &str,
) -> Result<Table<'gc>, Error<'gc>> {
    v.get_table().ok_or_else(|| {
        Error::from_str(
            ctx,
            &format!(
                "bad argument #1 to '{fname}' (table expected, got {})",
                v.type_name()
            ),
        )
    })
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
/// Indexing is raw, so `__index`/`__len` are not consulted.
fn lua_concat<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let t = check_table(nctx.ctx, stack.get(0), "concat")?;
    let sep_arg = stack.get(1);
    let sep = if sep_arg.is_nil() {
        Vec::new()
    } else if let Some(s) = sep_arg.get_string() {
        s.as_bytes().to_vec()
    } else if sep_arg.get_integer().is_some() || sep_arg.get_float().is_some() {
        util::basic_tostring(nctx.ctx, sep_arg).as_bytes().to_vec()
    } else {
        return Err(Error::from_str(
            nctx.ctx,
            &format!(
                "bad argument #2 to 'concat' (string expected, got {})",
                sep_arg.type_name()
            ),
        ));
    };
    let i_arg = stack.get(2);
    let i = if i_arg.is_nil() {
        1
    } else {
        util::check_integer(nctx.ctx, i_arg, "concat", 3)?
    };
    let j_arg = stack.get(3);
    let j = if j_arg.is_nil() {
        t.raw_len() as i64
    } else {
        util::check_integer(nctx.ctx, j_arg, "concat", 4)?
    };

    let mut out = Vec::new();
    let mut k = i;
    while k <= j {
        let v = t.raw_get(Value::integer(k));
        if let Some(s) = v.get_string() {
            out.extend_from_slice(s.as_bytes());
        } else if let Some(n) = v.get_integer() {
            util::push_int(&mut out, n);
        } else if let Some(f) = v.get_float() {
            util::push_float(&mut out, f);
        } else {
            return Err(Error::from_str(
                nctx.ctx,
                &format!(
                    "invalid value ({}) at index {k} in table for 'concat'",
                    v.type_name()
                ),
            ));
        }
        if k != j {
            out.extend_from_slice(&sep);
        }
        k += 1;
    }
    stack.replace(&[Value::string(LuaString::new(nctx.ctx, &out))]);
    Ok(CallbackAction::Return)
}

/// `create(n [, m])` — return a fresh table. The `n`/`m` capacity hints are
/// accepted and validated but not yet used for preallocation; the table grows
/// on demand. TODO(#27): honor the hints once `Table` exposes reservation.
fn lua_create<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let n = util::check_integer(nctx.ctx, stack.get(0), "create", 1)?;
    let m_arg = stack.get(1);
    let m = if m_arg.is_nil() {
        0
    } else {
        util::check_integer(nctx.ctx, m_arg, "create", 2)?
    };
    if n < 0 {
        return Err(Error::from_str(
            nctx.ctx,
            "bad argument #1 to 'create' (out of range)",
        ));
    }
    if m < 0 {
        return Err(Error::from_str(
            nctx.ctx,
            "bad argument #2 to 'create' (out of range)",
        ));
    }
    stack.replace(&[Value::table(Table::new(nctx.ctx))]);
    Ok(CallbackAction::Return)
}

/// `insert(t, [pos,] value)` — append `value`, or insert it at `pos`, shifting
/// later elements up by one.
fn lua_insert<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let t = check_table(nctx.ctx, stack.get(0), "insert")?;
    let n = t.raw_len() as i64;
    match stack.len() {
        2 => {
            t.raw_set(nctx.ctx, Value::integer(n + 1), stack.get(1));
        }
        3 => {
            let pos = util::check_integer(nctx.ctx, stack.get(1), "insert", 2)?;
            if pos < 1 || pos > n + 1 {
                return Err(Error::from_str(
                    nctx.ctx,
                    "bad argument #2 to 'insert' (position out of bounds)",
                ));
            }
            let mut k = n;
            while k >= pos {
                let v = t.raw_get(Value::integer(k));
                t.raw_set(nctx.ctx, Value::integer(k + 1), v);
                k -= 1;
            }
            t.raw_set(nctx.ctx, Value::integer(pos), stack.get(2));
        }
        _ => {
            return Err(Error::from_str(
                nctx.ctx,
                "wrong number of arguments to 'insert'",
            ));
        }
    }
    stack.replace(&[]);
    Ok(CallbackAction::Return)
}

/// `move(a1, f, e, t [, a2])` — copy `a1[f..e]` to `a2[t..]` (`a2` defaults to
/// `a1`), returning `a2`. Overlapping ranges within one table are handled in
/// the safe direction.
fn lua_move<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let a1 = check_table(nctx.ctx, stack.get(0), "move")?;
    let f = util::check_integer(nctx.ctx, stack.get(1), "move", 2)?;
    let e = util::check_integer(nctx.ctx, stack.get(2), "move", 3)?;
    let t = util::check_integer(nctx.ctx, stack.get(3), "move", 4)?;
    let a2_arg = stack.get(4);
    let a2 = if a2_arg.is_nil() {
        a1
    } else {
        check_table(nctx.ctx, a2_arg, "move")?
    };

    if e >= f {
        // PUC-Lua's two bounds: the element count `e - f + 1` must fit a Lua
        // integer (else `e - f` itself overflows), and the destination range
        // `t .. t + n - 1` must not wrap past maxinteger.
        if !(f > 0 || e < i64::MAX + f) {
            return Err(Error::from_str(
                nctx.ctx,
                "bad argument #3 to 'move' (too many elements to move)",
            ));
        }
        let n = e - f + 1;
        if t > i64::MAX - n + 1 {
            return Err(Error::from_str(
                nctx.ctx,
                "bad argument #4 to 'move' (destination wrap around)",
            ));
        }
        let same = crate::dmm::Gc::ptr_eq(a1.inner(), a2.inner());
        // Copy forward unless the destination overlaps the tail of the source
        // within the same table; the guards above keep `f + i` / `t + i` in range.
        if t > e || t <= f || !same {
            for i in 0..n {
                let v = a1.raw_get(Value::integer(f + i));
                a2.raw_set(nctx.ctx, Value::integer(t + i), v);
            }
        } else {
            let mut i = n - 1;
            loop {
                let v = a1.raw_get(Value::integer(f + i));
                a2.raw_set(nctx.ctx, Value::integer(t + i), v);
                if i == 0 {
                    break;
                }
                i -= 1;
            }
        }
    }
    stack.replace(&[Value::table(a2)]);
    Ok(CallbackAction::Return)
}

/// `pack(...)` — collect all arguments into a new table with field `n` set to
/// the argument count.
fn lua_pack<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let n = stack.len();
    let t = Table::new(nctx.ctx);
    for i in 0..n {
        t.raw_set(nctx.ctx, Value::integer(i as i64 + 1), stack.get(i));
    }
    t.raw_set(
        nctx.ctx,
        Value::string(LuaString::new(nctx.ctx, b"n")),
        Value::integer(n as i64),
    );
    stack.replace(&[Value::table(t)]);
    Ok(CallbackAction::Return)
}

/// `remove(t [, pos])` — remove and return `t[pos]` (default `#t`), shifting
/// later elements down by one.
fn lua_remove<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let t = check_table(nctx.ctx, stack.get(0), "remove")?;
    let n = t.raw_len() as i64;
    let pos_arg = stack.get(1);
    let pos = if pos_arg.is_nil() {
        n
    } else {
        util::check_integer(nctx.ctx, pos_arg, "remove", 2)?
    };
    // Any position but the (empty-table) default must name a real border slot.
    if pos != n && (pos < 1 || pos > n + 1) {
        return Err(Error::from_str(
            nctx.ctx,
            "bad argument #2 to 'remove' (position out of bounds)",
        ));
    }
    let result = t.raw_get(Value::integer(pos));
    let mut k = pos;
    while k < n {
        let v = t.raw_get(Value::integer(k + 1));
        t.raw_set(nctx.ctx, Value::integer(k), v);
        k += 1;
    }
    t.raw_set(nctx.ctx, Value::integer(pos.max(n)), Value::nil());
    stack.replace(&[result]);
    Ok(CallbackAction::Return)
}

/// `sort(t [, comp])` — sort `t[1..#t]` in place. With no comparator the default
/// `<` order is used (numbers, strings, or an `__lt` metamethod); otherwise
/// `comp(a, b)` must return true when `a` should precede `b`. An inconsistent
/// comparator raises "invalid order function for sorting". Because a comparator
/// (or an `__lt` metamethod) re-enters the interpreter, the work runs as an
/// async `Sequence`; a comparator-free primitive sort still completes in a
/// single poll without ever suspending.
fn lua_sort<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let t = check_table(nctx.ctx, stack.get(0), "sort")?;
    let n = t.raw_len();
    if n <= 1 {
        // Nothing to do — and, matching Lua, the comparator argument is not even
        // type-checked for a trivial array.
        stack.replace(&[]);
        return Ok(CallbackAction::Return);
    }
    if n >= i32::MAX as usize {
        return Err(Error::from_str(
            nctx.ctx,
            "bad argument #1 to 'sort' (array too big)",
        ));
    }
    let comp_arg = stack.get(1);
    let comp = if comp_arg.is_nil() {
        None
    } else if let Some(f) = comp_arg.get_function() {
        Some(f)
    } else {
        return Err(Error::from_str(
            nctx.ctx,
            &format!(
                "bad argument #2 to 'sort' (function expected, got {})",
                comp_arg.type_name()
            ),
        ));
    };

    let mc = nctx.ctx.mutation();
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
    Ok(CallbackAction::Sequence(seq))
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
        CallMeta(StashedFunction),
    }
    let plan = seq.try_enter(|ctx, locals, _exec, mut stack| {
        let tbl = locals.fetch(t);
        let a = tbl.raw_get(Value::integer(xi as i64));
        let b = tbl.raw_get(Value::integer(yi as i64));
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
            Some((x as f64) < y)
        } else if let (Some(x), Some(y)) = (a.get_float(), b.get_integer()) {
            Some(x < (y as f64))
        } else if let (Some(x), Some(y)) = (a.get_string(), b.get_string()) {
            Some(x < y)
        } else {
            None
        };
        if let Some(r) = prim {
            return Ok(Plan::Ready(r));
        }
        let m = lt_metamethod(ctx, a, b);
        if let Some(f) = m.get_function() {
            stack.replace(&[a, b]);
            Ok(Plan::CallMeta(locals.stash(ctx.mutation(), f)))
        } else {
            Err(Error::from_str(ctx, &util::compare_error_msg(a, b)))
        }
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
        let tbl = locals.fetch(t);
        let kx = Value::integer(x as i64);
        let ky = Value::integer(y as i64);
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

/// The `__lt` metamethod for an ordering of `a < b` (checked on `a` then `b`),
/// mirroring the VM's `binop_metamethod`.
fn lt_metamethod<'gc>(ctx: Context<'gc>, a: Value<'gc>, b: Value<'gc>) -> Value<'gc> {
    let name = ctx.symbols().mm_lt;
    if let Some(t) = a.get_table() {
        let m = t.get_metamethod(name);
        if !m.is_nil() {
            return m;
        }
    }
    if let Some(t) = b.get_table() {
        return t.get_metamethod(name);
    }
    Value::nil()
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
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let t = check_table(nctx.ctx, stack.get(0), "unpack")?;
    let i_arg = stack.get(1);
    let i = if i_arg.is_nil() {
        1
    } else {
        util::check_integer(nctx.ctx, i_arg, "unpack", 2)?
    };
    let j_arg = stack.get(2);
    let j = if j_arg.is_nil() {
        t.raw_len() as i64
    } else {
        util::check_integer(nctx.ctx, j_arg, "unpack", 3)?
    };
    if i > j {
        stack.replace(&[]); // empty range
        return Ok(CallbackAction::Return);
    }
    // `n - 1` as unsigned, so a full-i64-span range can't overflow the
    // subtraction (PUC-Lua's `tunpack`). Cap the count at i32::MAX results.
    let n_minus_1 = (j as u64).wrapping_sub(i as u64);
    if n_minus_1 >= i32::MAX as u64 {
        return Err(Error::from_str(nctx.ctx, "too many results to unpack"));
    }
    let mut out = Vec::with_capacity(n_minus_1 as usize + 1);
    // Iterate `k < j` then push `t[j]` separately, so `k += 1` never steps
    // past i64::MAX when `j == i64::MAX`.
    let mut k = i;
    while k < j {
        out.push(t.raw_get(Value::integer(k)));
        k += 1;
    }
    out.push(t.raw_get(Value::integer(j)));
    stack.replace(&out);
    Ok(CallbackAction::Return)
}
