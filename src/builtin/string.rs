use crate::Context;
use crate::env::{Function, LuaString, NativeContext, NativeError, NativeFn, Stack, Table, Value};

pub fn load<'gc>(ctx: Context<'gc>) {
    let fns: &[(&str, NativeFn)] = &[
        ("byte", lua_byte),
        ("char", lua_char),
        ("dump", lua_dump),
        ("find", lua_find),
        ("format", lua_format),
        ("gmatch", lua_gmatch),
        ("gsub", lua_gsub),
        ("len", lua_len),
        ("lower", lua_lower),
        ("match", lua_match),
        ("pack", lua_pack),
        ("packsize", lua_packsize),
        ("rep", lua_rep),
        ("reverse", lua_reverse),
        ("sub", lua_sub),
        ("unpack", lua_unpack),
        ("upper", lua_upper),
    ];

    let lib = Table::new(ctx.mutation());
    for &(name, handler) in fns {
        let handler = Function::new_native(ctx.mutation(), handler, Box::new([]));
        let key = Value::String(LuaString::new(ctx.mutation(), name.as_bytes()));
        lib.raw_set(ctx.mutation(), key, Value::Function(handler));
    }

    let lib_name = Value::String(LuaString::new(ctx.mutation(), b"string"));
    ctx.globals()
        .raw_set(ctx.mutation(), lib_name, Value::Table(lib));
}

fn lua_byte<'gc>(_ctx: NativeContext<'gc, '_>, _stack: Stack<'gc, '_>) -> Result<(), NativeError> {
    todo!()
}

fn lua_char<'gc>(_ctx: NativeContext<'gc, '_>, _stack: Stack<'gc, '_>) -> Result<(), NativeError> {
    todo!()
}

fn lua_dump<'gc>(_ctx: NativeContext<'gc, '_>, _stack: Stack<'gc, '_>) -> Result<(), NativeError> {
    todo!()
}

fn lua_find<'gc>(_ctx: NativeContext<'gc, '_>, _stack: Stack<'gc, '_>) -> Result<(), NativeError> {
    todo!()
}

fn lua_format<'gc>(
    _ctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<(), NativeError> {
    todo!()
}

fn lua_gmatch<'gc>(
    _ctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<(), NativeError> {
    todo!()
}

fn lua_gsub<'gc>(_ctx: NativeContext<'gc, '_>, _stack: Stack<'gc, '_>) -> Result<(), NativeError> {
    todo!()
}

fn lua_len<'gc>(_ctx: NativeContext<'gc, '_>, _stack: Stack<'gc, '_>) -> Result<(), NativeError> {
    todo!()
}

fn lua_lower<'gc>(_ctx: NativeContext<'gc, '_>, _stack: Stack<'gc, '_>) -> Result<(), NativeError> {
    todo!()
}

fn lua_match<'gc>(_ctx: NativeContext<'gc, '_>, _stack: Stack<'gc, '_>) -> Result<(), NativeError> {
    todo!()
}

fn lua_pack<'gc>(_ctx: NativeContext<'gc, '_>, _stack: Stack<'gc, '_>) -> Result<(), NativeError> {
    todo!()
}

fn lua_packsize<'gc>(
    _ctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<(), NativeError> {
    todo!()
}

fn lua_rep<'gc>(_ctx: NativeContext<'gc, '_>, _stack: Stack<'gc, '_>) -> Result<(), NativeError> {
    todo!()
}

fn lua_reverse<'gc>(
    _ctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<(), NativeError> {
    todo!()
}

fn lua_sub<'gc>(_ctx: NativeContext<'gc, '_>, _stack: Stack<'gc, '_>) -> Result<(), NativeError> {
    todo!()
}

fn lua_unpack<'gc>(
    _ctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<(), NativeError> {
    todo!()
}

fn lua_upper<'gc>(_ctx: NativeContext<'gc, '_>, _stack: Stack<'gc, '_>) -> Result<(), NativeError> {
    todo!()
}
