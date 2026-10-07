//! A native's unprotected call frame (the kind `sort` and `gsub` have) is
//! transparent to errors and to an enclosing `xpcall` handler.

use crate::env::{Error, Function, LuaString, NativeClosure, Stack, Value};
use crate::vm::native::{NativeOut, OnOk, Protect, cont};
use crate::{Context, Executor, LoadError, Lua};

fn run_with<T: for<'gc> crate::FromMultiValue<'gc>>(
    src: &str,
    setup: impl for<'gc> FnOnce(crate::Context<'gc>),
) -> T {
    let mut lua = Lua::new();
    lua.load_all();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            setup(ctx);
            let chunk = ctx.load(src, Some("=c"))?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    lua.execute(&ex).expect("run")
}

fn lua_frame_count<'gc>(
    ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    mut stack: Stack<'gc, '_>,
) -> Result<(), Error<'gc>> {
    let n = stack.lua_frame_count();
    stack.replace(&[Value::integer(ctx.mutation(), n as i64)]);
    Ok(())
}

/// `through(f, ...)`: call `f` from an unprotected native frame, the kind a
/// native with a callback (`sort`, `gsub`) has, which must not shadow an
/// enclosing `xpcall` handler.
fn lua_through<'gc>(
    _ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    _stack: Stack<'gc, '_>,
) -> NativeOut {
    NativeOut::call_then(0, cont::TEST_THROUGH, Protect::No, OnOk::Cont)
}

pub(crate) fn through_cont<'gc>(
    _ctx: Context<'gc>,
    _closure: &NativeClosure<'gc>,
    _stack: Stack<'gc, '_>,
    _status: Result<(), Error<'gc>>,
) -> NativeOut {
    NativeOut::RETURN
}

fn install_through<'gc>(ctx: crate::Context<'gc>) {
    let set = |name: &str, f: Function<'gc>| {
        let key = Value::string(LuaString::new(ctx, name.as_bytes()));
        ctx.globals().raw_set(ctx, key, Value::function(f));
    };
    set(
        "lua_frames",
        Function::new_native(ctx.mutation(), lua_frame_count, &[]),
    );
    set(
        "through",
        Function::new_cont(ctx.mutation(), lua_through, &[]),
    );
}

#[test]
fn pass_through_sequence_is_transparent_to_errors() {
    // Success path returns the callee's results; an error passes through
    // it to the enclosing pcall.
    let v: i64 = run_with(
        "local ok, e = pcall(function() through(function() error('x') end) end)\n\
         local a, b = through(function() return 1, 2 end)\n\
         return (not ok and e == 'c:1: x' and a == 1 and b == 2) and 1 or 0",
        install_through,
    );
    assert_eq!(v, 1);
}

#[test]
fn handler_looks_through_a_pass_through_frame() {
    // The nearest native frame is `through`'s, but it isn't a catch point, so
    // the xpcall handler still runs with `outer` and `deep` intact.
    let frames: i64 = run_with(
        "local function deep() error('x') end\n\
         local function outer() through(deep) end\n\
         local ok, n = xpcall(outer, lua_frames)\n\
         return n",
        install_through,
    );
    assert_eq!(frames, 3);
}
