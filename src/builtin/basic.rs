use crate::Context;
use crate::env::{Function, LuaString, NativeContext, NativeError, NativeFn, Stack, Value};

// See #27: _G, _VERSION

// let add =
//                    Function::new_native(ctx.mutation(), native_add as NativeFn, Box::new([]));
//                let key = Value::String(LuaString::new(ctx.mutation(), b"add"));
//                ctx.globals()
//                    .raw_set(ctx.mutation(), key, Value::Function(add));

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
        let key = Value::String(LuaString::new(ctx.mutation(), name.as_bytes()));

        ctx.globals()
            .raw_set(ctx.mutation(), key, Value::Function(handler));
    }
}

fn lua_assert<'gc>(
    _ctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<(), NativeError> {
    todo!()
}

fn lua_collectgarbage<'gc>(
    _ctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<(), NativeError> {
    todo!()
}

fn lua_dofile<'gc>(
    _ctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<(), NativeError> {
    todo!()
}

fn lua_error<'gc>(_ctx: NativeContext<'gc, '_>, _stack: Stack<'gc, '_>) -> Result<(), NativeError> {
    todo!()
}

fn lua_getmetatable<'gc>(
    _ctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<(), NativeError> {
    todo!()
}

fn lua_ipairs<'gc>(
    _ctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<(), NativeError> {
    todo!()
}

fn lua_load<'gc>(_ctx: NativeContext<'gc, '_>, _stack: Stack<'gc, '_>) -> Result<(), NativeError> {
    todo!()
}

fn lua_loadfile<'gc>(
    _ctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<(), NativeError> {
    todo!()
}

fn lua_next<'gc>(_ctx: NativeContext<'gc, '_>, _stack: Stack<'gc, '_>) -> Result<(), NativeError> {
    todo!()
}

fn lua_pairs<'gc>(_ctx: NativeContext<'gc, '_>, _stack: Stack<'gc, '_>) -> Result<(), NativeError> {
    todo!()
}

fn lua_pcall<'gc>(_ctx: NativeContext<'gc, '_>, _stack: Stack<'gc, '_>) -> Result<(), NativeError> {
    todo!()
}

fn lua_print<'gc>(_ctx: NativeContext<'gc, '_>, stack: Stack<'gc, '_>) -> Result<(), NativeError> {
    let arg = stack[0];
    let num = arg.get_integer().unwrap();
    println!("{}", num);
    Ok(())
}

fn lua_rawequal<'gc>(
    _ctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<(), NativeError> {
    todo!()
}

fn lua_rawget<'gc>(
    _ctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<(), NativeError> {
    todo!()
}

fn lua_rawlen<'gc>(
    _ctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<(), NativeError> {
    todo!()
}

fn lua_rawset<'gc>(
    _ctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<(), NativeError> {
    todo!()
}

fn lua_select<'gc>(
    _ctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<(), NativeError> {
    todo!()
}

fn lua_setmetatable<'gc>(
    _ctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<(), NativeError> {
    todo!()
}

fn lua_tonumber<'gc>(
    _ctx: NativeContext<'gc, '_>,
    mut stack: Stack<'gc, '_>,
) -> Result<(), NativeError> {
    // TODO: 2-arg form `tonumber(s, base)` for integer parsing in arbitrary base.
    let v = stack.get(0);
    let result = match v {
        Value::Nil => Value::Nil,
        Value::Integer(_) | Value::Float(_) => v,
        Value::String(s) => {
            let bytes = s.as_bytes();
            let trimmed = trim_ascii(bytes);
            parse_lua_number(trimmed).unwrap_or(Value::Nil)
        }
        _ => Value::Nil,
    };
    stack.replace(&[result]);
    Ok(())
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
        return Some(Value::Integer(i));
    }
    if let Ok(f) = s.parse::<f64>() {
        return Some(Value::Float(f));
    }
    None
}

fn lua_tostring<'gc>(
    _ctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<(), NativeError> {
    todo!()
}

fn lua_type<'gc>(_ctx: NativeContext<'gc, '_>, _stack: Stack<'gc, '_>) -> Result<(), NativeError> {
    todo!()
}

fn lua_warn<'gc>(_ctx: NativeContext<'gc, '_>, _stack: Stack<'gc, '_>) -> Result<(), NativeError> {
    todo!()
}

fn lua_xpcall<'gc>(
    _ctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<(), NativeError> {
    todo!()
}
