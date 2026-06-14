use std::io::Write;

use crate::Context;
use crate::builtin::util;
use crate::env::{Error, Function, LuaString, NativeContext, NativeFn, Stack, Value};
use crate::vm::sequence::CallbackAction;

// TODO(#27): _G, _VERSION

pub fn load<'gc>(ctx: Context<'gc>) {
    let fns: &[(&str, NativeFn)] = &[
        ("assert", lua_assert),
        ("collectgarbage", lua_collectgarbage),
        ("dofile", lua_dofile),
        ("error", lua_error),
        ("getmetatable", lua_getmetatable),
        ("ipairs", lua_ipairs),
        ("load", lua_load),
        ("loadfile", lua_loadfile),
        ("next", lua_next),
        ("pairs", lua_pairs),
        ("pcall", lua_pcall),
        ("print", lua_print),
        ("rawequal", lua_rawequal),
        ("rawget", lua_rawget),
        ("rawlen", lua_rawlen),
        ("rawset", lua_rawset),
        ("select", lua_select),
        ("setmetatable", lua_setmetatable),
        ("tonumber", lua_tonumber),
        ("tostring", lua_tostring),
        ("type", lua_type),
        ("warn", lua_warn),
        ("xpcall", lua_xpcall),
    ];

    for &(name, handler) in fns {
        let handler = Function::new_native(ctx.mutation(), handler, Box::new([]));
        let key = Value::string(LuaString::new(ctx, name.as_bytes()));

        ctx.globals().raw_set(ctx, key, Value::function(handler));
    }
}

/// `assert(v [, message, ...])` — if `v` is truthy, return all arguments
/// unchanged; otherwise raise `message` (default `"assertion failed!"`). The
/// message is raised verbatim, matching `assert`'s delegation to `error`
/// (no position prefix is added when the caller is a native frame).
fn lua_assert<'gc>(
    nctx: NativeContext<'gc, '_>,
    stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    if stack.is_empty() {
        return Err(Error::from_str(
            nctx.ctx,
            "bad argument #1 to 'assert' (value expected)",
        ));
    }
    if !stack.get(0).is_falsy() {
        // Leaving the window untouched returns all arguments.
        return Ok(CallbackAction::Return);
    }
    if stack.len() >= 2 {
        Err(Error::new(stack.get(1)))
    } else {
        Err(Error::from_str(nctx.ctx, "assertion failed!"))
    }
}

/// `collectgarbage([opt [, arg]])` — light stand-in until the GC exposes the
/// control/introspection API this needs. Recognized options return
/// plausible values; collection requests are accepted as no-ops.
fn lua_collectgarbage<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let opt = stack.get(0);
    let opt = opt.get_string().map_or(&b"collect"[..], |s| s.as_bytes());
    match opt {
        // Lua returns a single value: total memory in use, in Kbytes. (The
        // absolute figure differs from PUC-Lua — different allocator — but the
        // shape/units match; full GC accounting tracked in #62.)
        b"count" => {
            let kb = nctx.ctx.mutation().metrics().total_allocation() as f64 / 1024.0;
            stack.replace(&[Value::float(kb)]);
        }
        b"isrunning" => stack.replace(&[Value::boolean(true)]),
        b"step" => stack.replace(&[Value::boolean(false)]),
        _ => stack.replace(&[Value::integer(0)]),
    }
    Ok(CallbackAction::Return)
}

fn lua_dofile<'gc>(
    _ctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    todo!()
}

/// `error(message [, level])` — raise `message` as a Lua error. The `level`
/// argument selects which call frame's position is prepended to a string
/// message; we don't yet have native access to caller line info, so the value
/// is raised verbatim (equivalent to `level == 0`). TODO(#27): prepend
/// `"source:line:"` for `level >= 1` once reachable from a native frame.
fn lua_error<'gc>(
    _nctx: NativeContext<'gc, '_>,
    stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    Err(Error::new(stack.get(0)))
}

fn lua_getmetatable<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let v = stack.get(0);
    // A `__metatable` field shadows the real metatable (protection); tables
    // and userdata both carry per-object metatables.
    let metatable = if let Some(t) = v.get_table() {
        t.metatable()
    } else {
        v.get_userdata().and_then(|u| u.metatable())
    };
    let result = match metatable {
        Some(mt) => {
            let prot = mt.raw_get(Value::string(LuaString::new(nctx.ctx, b"__metatable")));
            if prot.is_nil() {
                Value::table(mt)
            } else {
                prot
            }
        }
        None => Value::nil(),
    };
    stack.replace(&[result]);
    Ok(CallbackAction::Return)
}

/// `ipairs(t)` — returns `(iterator, t, 0)`. The iterator yields `1,t[1]`,
/// `2,t[2]`, … stopping at the first absent index.
///
/// Indexing is raw (no `__index`); Lua 5.3+ routes `ipairs` through
/// metamethod-aware `geti`, but that requires invoking `__index`, which can
/// re-enter Lua. TODO(#27): honor `__index` once native→Lua calls are wired.
fn lua_ipairs<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    if stack.is_empty() {
        return Err(Error::from_str(
            nctx.ctx,
            "bad argument #1 to 'ipairs' (value expected)",
        ));
    }
    let t = stack.get(0);
    let iter = Function::new_native(nctx.ctx.mutation(), ipairs_aux, Box::new([]));
    stack.replace(&[Value::function(iter), t, Value::integer(0)]);
    Ok(CallbackAction::Return)
}

/// Stateless iterator body for `ipairs`: `(t, i) -> (i+1, t[i+1])`, or a lone
/// `nil` once the array part ends.
fn ipairs_aux<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let t = stack.get(0).get_table().ok_or_else(|| {
        Error::from_str(
            nctx.ctx,
            "bad argument #1 to 'ipairs iterator' (table expected)",
        )
    })?;
    let i = stack.get(1).get_integer().unwrap_or(0) + 1;
    let v = t.raw_get(Value::integer(i));
    if v.is_nil() {
        stack.replace(&[Value::nil()]);
    } else {
        stack.replace(&[Value::integer(i), v]);
    }
    Ok(CallbackAction::Return)
}

fn lua_load<'gc>(
    _ctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    todo!()
}

fn lua_loadfile<'gc>(
    _ctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    todo!()
}

fn lua_next<'gc>(
    _ctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    todo!()
}

fn lua_pairs<'gc>(
    _ctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    todo!()
}

fn lua_pcall<'gc>(
    _ctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    todo!()
}

/// `print(...)` — write each argument's `tostring` form to stdout, separated
/// by tabs and followed by a newline. Uses the default representation;
/// TODO(#27): honor `__tostring` once metamethod calls from natives exist.
fn lua_print<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    let n = stack.len();
    for i in 0..n {
        if i > 0 {
            let _ = out.write_all(b"\t");
        }
        let s = util::basic_tostring(nctx.ctx, stack.get(i));
        let _ = out.write_all(s.as_bytes());
    }
    let _ = out.write_all(b"\n");
    stack.replace(&[]);
    Ok(CallbackAction::Return)
}

/// `rawequal(a, b)` — primitive equality, bypassing `__eq`.
fn lua_rawequal<'gc>(
    _nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let eq = util::raw_eq(stack.get(0), stack.get(1));
    stack.replace(&[Value::boolean(eq)]);
    Ok(CallbackAction::Return)
}

fn lua_rawget<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let t_arg = stack.get(0);
    let key = stack.get(1);
    let Some(t) = t_arg.get_table() else {
        return Err(Error::from_str(
            nctx.ctx,
            "bad argument #1 to 'rawget' (table expected)",
        ));
    };
    let v = t.raw_get(key);
    stack.replace(&[v]);
    Ok(CallbackAction::Return)
}

/// `rawlen(v)` — length of a table (border) or string (byte count), bypassing
/// `__len`.
fn lua_rawlen<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let v = stack.get(0);
    let len = if let Some(s) = v.get_string() {
        s.len() as i64
    } else if let Some(t) = v.get_table() {
        t.raw_len() as i64
    } else {
        let got = if stack.is_empty() {
            "no value"
        } else {
            v.type_name()
        };
        return Err(Error::from_str(
            nctx.ctx,
            &format!("bad argument #1 to 'rawlen' (table or string expected, got {got})"),
        ));
    };
    stack.replace(&[Value::integer(len)]);
    Ok(CallbackAction::Return)
}

fn lua_rawset<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let t_arg = stack.get(0);
    let key = stack.get(1);
    let value = stack.get(2);
    let Some(t) = t_arg.get_table() else {
        return Err(Error::from_str(
            nctx.ctx,
            "bad argument #1 to 'rawset' (table expected)",
        ));
    };
    t.raw_set(nctx.ctx, key, value);
    stack.replace(&[Value::table(t)]);
    Ok(CallbackAction::Return)
}

/// `select('#', ...)` returns the count of extra arguments; `select(n, ...)`
/// returns the arguments from position `n` onward (negative `n` counts from
/// the end).
fn lua_select<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let m = stack.len().saturating_sub(1); // count of arguments after the selector
    let sel = stack.get(0);
    if let Some(s) = sel.get_string()
        && s.as_bytes() == b"#"
    {
        stack.replace(&[Value::integer(m as i64)]);
        return Ok(CallbackAction::Return);
    }
    let i = util::check_integer(nctx.ctx, sel, "select", 1)?;
    let pos = if i < 0 { m as i64 + i + 1 } else { i };
    if pos < 1 {
        return Err(Error::from_str(
            nctx.ctx,
            "bad argument #1 to 'select' (index out of range)",
        ));
    }
    let mut out = Vec::new();
    let mut k = pos as usize;
    while k <= m {
        out.push(stack.get(k));
        k += 1;
    }
    stack.replace(&out);
    Ok(CallbackAction::Return)
}

/// `setmetatable(t, mt)` — attach `mt` (a table or nil) as `t`'s
/// metatable, returning `t`. Drives a shape transition along the
/// `set_metatable` edge so future accesses observe the new identity.
fn lua_setmetatable<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let t_arg = stack.get(0);
    let mt_arg = stack.get(1);
    let Some(t) = t_arg.get_table() else {
        return Err(Error::from_str(
            nctx.ctx,
            "bad argument #1 to 'setmetatable' (table expected)",
        ));
    };
    let mt = if mt_arg.is_nil() {
        None
    } else if let Some(mt) = mt_arg.get_table() {
        Some(mt)
    } else {
        return Err(Error::from_str(
            nctx.ctx,
            "bad argument #2 to 'setmetatable' (nil or table expected)",
        ));
    };
    // If the existing metatable carries a `__metatable` field, the
    // metatable is locked: refuse the change. Matches Lua 5.5 reference
    // behavior (`luaL_error("cannot change a protected metatable")`).
    if let Some(existing) = t.metatable() {
        let lock_key = LuaString::new(nctx.ctx, b"__metatable");
        let lock_val = existing.raw_get(Value::string(lock_key));
        if !lock_val.is_nil() {
            return Err(Error::from_str(
                nctx.ctx,
                "cannot change a protected metatable",
            ));
        }
    }
    t.set_metatable(nctx.ctx, mt);
    stack.replace(&[Value::table(t)]);
    Ok(CallbackAction::Return)
}

fn lua_tonumber<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let v = stack.get(0);
    let base_arg = stack.get(1);
    let result = if !base_arg.is_nil() {
        // 2-arg form `tonumber(s, base)`. Lua's argument order: the base must be
        // an integer (#2), then `s` must be a string (#1), and only then is the
        // base range validated (#2) — so e.g. `tonumber(nil, 99)` complains
        // about #1, not the out-of-range base.
        let base = util::check_integer(nctx.ctx, base_arg, "tonumber", 2)?;
        let s = v.get_string().ok_or_else(|| {
            Error::from_str(
                nctx.ctx,
                &format!(
                    "bad argument #1 to 'tonumber' (string expected, got {})",
                    v.type_name()
                ),
            )
        })?;
        if !(2..=36).contains(&base) {
            return Err(Error::from_str(
                nctx.ctx,
                "bad argument #2 to 'tonumber' (base out of range)",
            ));
        }
        util::str_to_int_base(s.as_bytes(), base as u32).map_or(Value::nil(), Value::integer)
    } else if v.get_integer().is_some() || v.get_float().is_some() {
        v
    } else if let Some(s) = v.get_string() {
        util::str_to_number(s.as_bytes()).unwrap_or(Value::nil())
    } else {
        Value::nil()
    };
    stack.replace(&[result]);
    Ok(CallbackAction::Return)
}

/// `tostring(v)` — default string representation. TODO(#27): honor
/// `__tostring`/`__name`, which require calling back into Lua.
fn lua_tostring<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    if stack.is_empty() {
        return Err(Error::from_str(
            nctx.ctx,
            "bad argument #1 to 'tostring' (value expected)",
        ));
    }
    let s = util::basic_tostring(nctx.ctx, stack.get(0));
    stack.replace(&[Value::string(s)]);
    Ok(CallbackAction::Return)
}

/// `type(v)` — the type name of `v` as a string.
fn lua_type<'gc>(
    nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    if stack.is_empty() {
        return Err(Error::from_str(
            nctx.ctx,
            "bad argument #1 to 'type' (value expected)",
        ));
    }
    let name = stack.get(0).type_name();
    stack.replace(&[Value::string(LuaString::new(nctx.ctx, name.as_bytes()))]);
    Ok(CallbackAction::Return)
}

/// `warn(msg, ...)` — Lua's warning system defaults to off and we do not yet
/// track the on/off toggle, so this validates the arguments (as Lua does, via
/// `luaL_checkstring`, which accepts strings *and* numbers) and otherwise does
/// nothing. TODO(#27): emit to stderr and honor `@on`/`@off` control messages
/// once warning state lives in `State`.
fn lua_warn<'gc>(
    nctx: NativeContext<'gc, '_>,
    stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    if stack.is_empty() {
        return Err(Error::from_str(
            nctx.ctx,
            "bad argument #1 to 'warn' (string expected, got no value)",
        ));
    }
    for i in 0..stack.len() {
        let v = stack.get(i);
        // `luaL_checkstring` coerces numbers to their string form, so integers
        // and floats are accepted; only truly non-coercible types error.
        if v.get_string().is_none() && v.get_integer().is_none() && v.get_float().is_none() {
            return Err(Error::from_str(
                nctx.ctx,
                &format!(
                    "bad argument #{} to 'warn' (string expected, got {})",
                    i + 1,
                    v.type_name()
                ),
            ));
        }
    }
    Ok(CallbackAction::Return)
}

fn lua_xpcall<'gc>(
    _ctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    todo!()
}
