use crate::Context;
use crate::builtin::util;
use crate::env::{Error, Function, LuaString, NativeClosure, NativeFn, Stack, Table, Value};
use crate::vm::sequence::CallbackAction;

pub fn load<'gc>(ctx: Context<'gc>) {
    let fns: &[(&str, NativeFn)] = &[
        ("getmetatable", lua_getmetatable),
        ("setmetatable", lua_setmetatable),
    ];

    let lib = Table::new(ctx);
    for &(name, handler) in fns {
        let handler = Function::new_native(ctx.mutation(), handler, &[]);
        let key = Value::string(LuaString::new(ctx, name.as_bytes()));
        lib.raw_set(ctx, key, Value::function(handler));
    }
    util::set_not_implemented(
        ctx,
        lib,
        "debug",
        &[
            "debug",
            "gethook",
            "getinfo",
            "getlocal",
            "getregistry",
            "getupvalue",
            "getuservalue",
            "sethook",
            "setlocal",
            "setupvalue",
            "setuservalue",
            "traceback",
            "upvalueid",
            "upvaluejoin",
        ],
    );

    let lib_name = Value::string(LuaString::new(ctx, b"debug"));
    ctx.globals().raw_set(ctx, lib_name, Value::table(lib));
}

/// `debug.getmetatable(v)` — `v`'s metatable, ignoring `__metatable`.
fn lua_getmetatable<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    if stack.is_empty() {
        return Err(util::arg_error(ctx, "getmetatable", 1, "value expected"));
    }
    let mt = ctx.metatable_of(stack.get(0));
    stack.replace(&[mt.map_or(Value::nil(), Value::table)]);
    Ok(CallbackAction::Return)
}

/// `debug.setmetatable(v, mt)` — set `v`'s metatable (shared by its whole
/// type unless `v` is a table or userdata), ignoring `__metatable`; returns `v`.
fn lua_setmetatable<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let v = stack.get(0);
    let mt = match stack.arg(1) {
        Some(v) if v.is_nil() => None,
        Some(v) if let Some(mt) = v.get_table() => Some(mt),
        got => {
            return Err(util::type_error(
                ctx,
                "setmetatable",
                2,
                "nil or table",
                got,
            ));
        }
    };
    ctx.set_metatable_of(v, mt);
    stack.replace(&[v]);
    Ok(CallbackAction::Return)
}
