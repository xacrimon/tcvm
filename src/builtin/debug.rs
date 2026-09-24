use crate::Context;
use crate::env::{Error, Function, LuaString, NativeClosure, NativeFn, Stack, Table, Value};
use crate::vm::sequence::CallbackAction;

pub fn load<'gc>(ctx: Context<'gc>) {
    let fns: &[(&str, NativeFn)] = &[
        ("debug", lua_debug),
        ("gethook", lua_gethook),
        ("getinfo", lua_getinfo),
        ("getlocal", lua_getlocal),
        ("getmetatable", lua_getmetatable),
        ("getregistry", lua_getregistry),
        ("getupvalue", lua_getupvalue),
        ("getuservalue", lua_getuservalue),
        ("sethook", lua_sethook),
        ("setlocal", lua_setlocal),
        ("setmetatable", lua_setmetatable),
        ("setupvalue", lua_setupvalue),
        ("setuservalue", lua_setuservalue),
        ("traceback", lua_traceback),
        ("upvalueid", lua_upvalueid),
        ("upvaluejoin", lua_upvaluejoin),
    ];

    let lib = Table::new(ctx);
    for &(name, handler) in fns {
        let handler = Function::new_native(ctx.mutation(), handler, Box::new([]));
        let key = Value::string(LuaString::new(ctx, name.as_bytes()));
        lib.raw_set(ctx, key, Value::function(handler));
    }

    let lib_name = Value::string(LuaString::new(ctx, b"debug"));
    ctx.globals().raw_set(ctx, lib_name, Value::table(lib));
}

fn lua_debug<'gc>(
    _ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    _stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    todo!()
}

fn lua_gethook<'gc>(
    _ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    _stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    todo!()
}

fn lua_getinfo<'gc>(
    _ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    _stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    todo!()
}

fn lua_getlocal<'gc>(
    _ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    _stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    todo!()
}

/// `debug.getmetatable(v)` — `v`'s metatable, ignoring `__metatable`.
fn lua_getmetatable<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    if stack.is_empty() {
        return Err(Error::from_str(
            ctx,
            "bad argument #1 to 'getmetatable' (value expected)",
        ));
    }
    let mt = ctx.metatable_of(stack.get(0));
    stack.replace(&[mt.map_or(Value::nil(), Value::table)]);
    Ok(CallbackAction::Return)
}

fn lua_getregistry<'gc>(
    _ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    _stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    todo!()
}

fn lua_getupvalue<'gc>(
    _ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    _stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    todo!()
}

fn lua_getuservalue<'gc>(
    _ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    _stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    todo!()
}

fn lua_sethook<'gc>(
    _ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    _stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    todo!()
}

fn lua_setlocal<'gc>(
    _ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    _stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    todo!()
}

/// `debug.setmetatable(v, mt)` — set `v`'s metatable (shared by its whole
/// type unless `v` is a table or userdata), ignoring `__metatable`; returns `v`.
fn lua_setmetatable<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    let v = stack.get(0);
    let mt_arg = stack.get(1);
    let mt = match mt_arg.get_table() {
        Some(mt) => Some(mt),
        None if mt_arg.is_nil() && stack.len() >= 2 => None,
        None => {
            let got = if stack.len() < 2 {
                "no value"
            } else {
                mt_arg.type_name()
            };
            return Err(Error::from_str(
                ctx,
                &format!("bad argument #2 to 'setmetatable' (nil or table expected, got {got})"),
            ));
        }
    };
    ctx.set_metatable_of(v, mt);
    stack.replace(&[v]);
    Ok(CallbackAction::Return)
}

fn lua_setupvalue<'gc>(
    _ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    _stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    todo!()
}

fn lua_setuservalue<'gc>(
    _ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    _stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    todo!()
}

fn lua_traceback<'gc>(
    _ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    _stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    todo!()
}

fn lua_upvalueid<'gc>(
    _ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    _stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    todo!()
}

fn lua_upvaluejoin<'gc>(
    _ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    _stack: Stack<'gc, '_>,
) -> Result<CallbackAction<'gc>, Error<'gc>> {
    todo!()
}
