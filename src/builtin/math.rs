use crate::Context;
use crate::env::{Function, LuaString, NativeContext, NativeError, NativeFn, Stack, Table, Value};

// TODO: constants — huge, maxinteger, mininteger, pi

pub fn load<'gc>(ctx: Context<'gc>) {
    let fns: &[(&str, NativeFn)] = &[
        ("abs", lua_abs),
        ("acos", lua_acos),
        ("asin", lua_asin),
        ("atan", lua_atan),
        ("ceil", lua_ceil),
        ("cos", lua_cos),
        ("deg", lua_deg),
        ("exp", lua_exp),
        ("floor", lua_floor),
        ("fmod", lua_fmod),
        ("log", lua_log),
        ("max", lua_max),
        ("min", lua_min),
        ("modf", lua_modf),
        ("rad", lua_rad),
        ("random", lua_random),
        ("randomseed", lua_randomseed),
        ("sin", lua_sin),
        ("sqrt", lua_sqrt),
        ("tan", lua_tan),
        ("tointeger", lua_tointeger),
        ("type", lua_type),
        ("ult", lua_ult),
    ];

    let lib = Table::new(ctx.mutation());
    for &(name, handler) in fns {
        let handler = Function::new_native(ctx.mutation(), handler, Box::new([]));
        let key = Value::String(LuaString::new(ctx.mutation(), name.as_bytes()));
        lib.raw_set(ctx.mutation(), key, Value::Function(handler));
    }

    let lib_name = Value::String(LuaString::new(ctx.mutation(), b"math"));
    ctx.globals()
        .raw_set(ctx.mutation(), lib_name, Value::Table(lib));
}

fn lua_abs<'gc>(_ctx: NativeContext<'gc, '_>, _stack: Stack<'gc, '_>) -> Result<(), NativeError> {
    todo!()
}

fn lua_acos<'gc>(_ctx: NativeContext<'gc, '_>, _stack: Stack<'gc, '_>) -> Result<(), NativeError> {
    todo!()
}

fn lua_asin<'gc>(_ctx: NativeContext<'gc, '_>, _stack: Stack<'gc, '_>) -> Result<(), NativeError> {
    todo!()
}

fn lua_atan<'gc>(_ctx: NativeContext<'gc, '_>, _stack: Stack<'gc, '_>) -> Result<(), NativeError> {
    todo!()
}

fn lua_ceil<'gc>(_ctx: NativeContext<'gc, '_>, _stack: Stack<'gc, '_>) -> Result<(), NativeError> {
    todo!()
}

fn lua_cos<'gc>(_ctx: NativeContext<'gc, '_>, _stack: Stack<'gc, '_>) -> Result<(), NativeError> {
    todo!()
}

fn lua_deg<'gc>(_ctx: NativeContext<'gc, '_>, _stack: Stack<'gc, '_>) -> Result<(), NativeError> {
    todo!()
}

fn lua_exp<'gc>(_ctx: NativeContext<'gc, '_>, _stack: Stack<'gc, '_>) -> Result<(), NativeError> {
    todo!()
}

fn lua_floor<'gc>(_ctx: NativeContext<'gc, '_>, _stack: Stack<'gc, '_>) -> Result<(), NativeError> {
    todo!()
}

fn lua_fmod<'gc>(_ctx: NativeContext<'gc, '_>, _stack: Stack<'gc, '_>) -> Result<(), NativeError> {
    todo!()
}

fn lua_log<'gc>(_ctx: NativeContext<'gc, '_>, _stack: Stack<'gc, '_>) -> Result<(), NativeError> {
    todo!()
}

fn lua_max<'gc>(_ctx: NativeContext<'gc, '_>, _stack: Stack<'gc, '_>) -> Result<(), NativeError> {
    todo!()
}

fn lua_min<'gc>(_ctx: NativeContext<'gc, '_>, _stack: Stack<'gc, '_>) -> Result<(), NativeError> {
    todo!()
}

fn lua_modf<'gc>(_ctx: NativeContext<'gc, '_>, _stack: Stack<'gc, '_>) -> Result<(), NativeError> {
    todo!()
}

fn lua_rad<'gc>(_ctx: NativeContext<'gc, '_>, _stack: Stack<'gc, '_>) -> Result<(), NativeError> {
    todo!()
}

fn lua_random<'gc>(
    _ctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<(), NativeError> {
    todo!()
}

fn lua_randomseed<'gc>(
    _ctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<(), NativeError> {
    todo!()
}

fn lua_sin<'gc>(_ctx: NativeContext<'gc, '_>, _stack: Stack<'gc, '_>) -> Result<(), NativeError> {
    todo!()
}

fn lua_sqrt<'gc>(_ctx: NativeContext<'gc, '_>, _stack: Stack<'gc, '_>) -> Result<(), NativeError> {
    todo!()
}

fn lua_tan<'gc>(_ctx: NativeContext<'gc, '_>, _stack: Stack<'gc, '_>) -> Result<(), NativeError> {
    todo!()
}

fn lua_tointeger<'gc>(
    _ctx: NativeContext<'gc, '_>,
    _stack: Stack<'gc, '_>,
) -> Result<(), NativeError> {
    todo!()
}

fn lua_type<'gc>(_ctx: NativeContext<'gc, '_>, _stack: Stack<'gc, '_>) -> Result<(), NativeError> {
    todo!()
}

fn lua_ult<'gc>(_ctx: NativeContext<'gc, '_>, _stack: Stack<'gc, '_>) -> Result<(), NativeError> {
    todo!()
}
