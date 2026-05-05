use crate::Context;
use crate::env::{Function, LuaString, NativeContext, NativeError, NativeFn, Stack, Table, Value};

// See #27: constants/tables — config, cpath, loaded, path, preload, searchers

pub fn load<'gc>(ctx: Context<'gc>) {
    let fns: &[(&str, NativeFn)] = &[("loadlib", lua_loadlib), ("searchpath", lua_searchpath)];

    let lib = Table::new(ctx);
    for &(name, handler) in fns {
        let handler = Function::new_native(ctx.mutation(), handler, Box::new([]));
        let key = Value::string(LuaString::new(ctx, name.as_bytes()));
        lib.raw_set(ctx.mutation(), key, Value::function(handler));
    }

    let lib_name = Value::string(LuaString::new(ctx, b"package"));
    ctx.globals()
        .raw_set(ctx.mutation(), lib_name, Value::table(lib));

    let require = Function::new_native(ctx.mutation(), lua_require, Box::new([]));
    let require_key = Value::string(LuaString::new(ctx, b"require"));
    ctx.globals()
        .raw_set(ctx.mutation(), require_key, Value::function(require));
}

fn lua_loadlib<'gc>(
    _ctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<(), NativeError> {
    todo!()
}

fn lua_searchpath<'gc>(
    _ctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<(), NativeError> {
    todo!()
}

fn lua_require<'gc>(
    _ctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<(), NativeError> {
    todo!()
}
