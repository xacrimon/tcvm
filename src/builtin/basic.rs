use crate::Context;
use crate::env::{Error, Function, LuaString, NativeContext, NativeFn, Stack, Value};
use crate::vm::sequence::CallbackAction;

// See #27: _G, _VERSION

// let add =
//                    Function::new_native(ctx.mutation(), native_add as NativeFn, Box::new([]));
//                let key = Value::String(LuaString::new(ctx, b"add"));
//                ctx.globals()
//                    .raw_set(ctx, key, Value::Function(add));

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

fn lua_assert<'gc>(
    _ctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    todo!()
}

fn lua_collectgarbage<'gc>(
    _ctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    todo!()
}

fn lua_dofile<'gc>(
    _ctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    todo!()
}

fn lua_error<'gc>(
    _ctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    todo!()
}

fn lua_getmetatable<'gc>(
    _nctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let v = stack.get(0);
    let result = if let Some(t) = v.get_table() {
        match t.metatable() {
            Some(mt) => Value::table(mt),
            None => Value::nil(),
        }
    } else {
        Value::nil()
    };
    stack.replace(&[result]);
    Ok(CallbackAction::Return)
}

fn lua_ipairs<'gc>(
    _ctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    todo!()
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

fn lua_print<'gc>(
    _ctx: NativeContext<'gc, '_>,
    stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let arg = stack[0];
    let num = arg.get_integer().unwrap();
    println!("{}", num);
    Ok(CallbackAction::Return)
}

fn lua_rawequal<'gc>(
    _ctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    todo!()
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

fn lua_rawlen<'gc>(
    _ctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    todo!()
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

fn lua_select<'gc>(
    _ctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    todo!()
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
    _ctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    // TODO: 2-arg form `tonumber(s, base)` for integer parsing in arbitrary base.
    let v = stack.get(0);
    let result = if v.is_nil() {
        Value::nil()
    } else if v.get_integer().is_some() || v.get_float().is_some() {
        v
    } else if let Some(s) = v.get_string() {
        let bytes = s.as_bytes();
        let trimmed = trim_ascii(bytes);
        parse_lua_number(trimmed).unwrap_or(Value::nil())
    } else {
        Value::nil()
    };
    stack.replace(&[result]);
    Ok(CallbackAction::Return)
}

fn trim_ascii(b: &[u8]) -> &[u8] {
    let mut start = 0;
    let mut end = b.len();
    while start < end && b[start].is_ascii_whitespace() {
        start += 1;
    }
    while end > start && b[end - 1].is_ascii_whitespace() {
        end -= 1;
    }
    &b[start..end]
}

fn parse_lua_number<'gc>(b: &[u8]) -> Option<Value<'gc>> {
    if b.is_empty() {
        return None;
    }
    let s = std::str::from_utf8(b).ok()?;
    // TODO: hex literals (0x...) and hex floats (0x1.8p3) per Lua spec.
    if let Ok(i) = s.parse::<i64>() {
        return Some(Value::integer(i));
    }
    if let Ok(f) = s.parse::<f64>() {
        return Some(Value::float(f));
    }
    None
}

fn lua_tostring<'gc>(
    _ctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    todo!()
}

fn lua_type<'gc>(
    _ctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    todo!()
}

fn lua_warn<'gc>(
    _ctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    todo!()
}

fn lua_xpcall<'gc>(
    _ctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    todo!()
}
