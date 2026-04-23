use crate::Context;
use crate::env::{Function, LuaString, NativeContext, NativeError, NativeFn, Stack, Table, Value};

// TODO: constant — charpattern

pub fn load<'gc>(ctx: Context<'gc>) {
    let fns: &[(&str, NativeFn)] = &[
        ("char", lua_char),
        ("codes", lua_codes),
        ("codepoint", lua_codepoint),
        ("len", lua_len),
        ("offset", lua_offset),
    ];

    let lib = Table::new(ctx.mutation());
    for &(name, handler) in fns {
        let handler = Function::new_native(ctx.mutation(), handler, Box::new([]));
        let key = Value::String(LuaString::new(ctx.mutation(), name.as_bytes()));
        lib.raw_set(ctx.mutation(), key, Value::Function(handler));
    }

    let lib_name = Value::String(LuaString::new(ctx.mutation(), b"utf8"));
    ctx.globals()
        .raw_set(ctx.mutation(), lib_name, Value::Table(lib));
}

fn lua_char<'gc>(_ctx: NativeContext<'gc, '_>, _stack: Stack<'gc, '_>) -> Result<(), NativeError> {
    todo!()
}

fn lua_codes<'gc>(_ctx: NativeContext<'gc, '_>, _stack: Stack<'gc, '_>) -> Result<(), NativeError> {
    todo!()
}

fn lua_codepoint<'gc>(
    _ctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<(), NativeError> {
    todo!()
}

fn lua_len<'gc>(_ctx: NativeContext<'gc, '_>, _stack: Stack<'gc, '_>) -> Result<(), NativeError> {
    todo!()
}

fn lua_offset<'gc>(
    _ctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<(), NativeError> {
    todo!()
}
