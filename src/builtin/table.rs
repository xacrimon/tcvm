use crate::Context;
use crate::builtin::util;
use crate::env::{Error, Function, LuaString, NativeContext, NativeFn, Stack, Table, Value};
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
    if n < 0 || m < 0 {
        return Err(Error::from_str(
            nctx.ctx,
            "bad argument #1 to 'create' (size out of bounds)",
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

// `table.sort` accepts an optional Lua comparator that must be invoked between
// elements, which requires re-entering the interpreter from native code (a
// `Sequence`-driven sort). Deferred to #27 follow-up alongside `pcall`.
fn lua_sort<'gc>(
    _ctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    todo!("table.sort: comparator callback needs native->Lua call support")
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
