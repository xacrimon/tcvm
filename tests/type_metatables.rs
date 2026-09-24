//! Per-type metatables (`G(L)->mt` in PUC Lua): every value of a type other
//! than table and userdata shares one metatable slot.

use tcvm::dmm::Gc;
use tcvm::env::{LuaString, Table, Value};
use tcvm::{Executor, LoadError, Lua, RuntimeError};

/// Expected strings below come from `lua` 5.5.1 running the same chunk.
fn run(src: &str) -> Result<String, String> {
    let mut lua = Lua::new();
    lua.load_all();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load(src, Some("=c"))?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    let as_string = |v: Value<'_>| {
        let s = v.get_string().expect("string value");
        String::from_utf8_lossy(s.as_bytes()).into_owned()
    };
    match lua.finish(&ex) {
        Ok(()) => Ok(lua.enter(|ctx| {
            let v = ctx.fetch(&ex).take_result::<Value>(ctx).expect("result");
            as_string(v)
        })),
        Err(RuntimeError::Lua(e)) => Err(lua.enter(|ctx| as_string(ctx.fetch(&e).value()))),
        Err(e) => panic!("unexpected failure for {src:?}: {e:?}"),
    }
}

fn ok(src: &str) -> String {
    run(src).unwrap_or_else(|e| panic!("{src:?} raised {e:?}"))
}

fn err(src: &str) -> String {
    run(src).expect_err(src)
}

#[test]
fn integers_and_floats_share_the_number_slot() {
    let mut lua = Lua::new();
    lua.enter(|ctx| {
        let mt = Table::new(ctx);
        ctx.set_metatable_of(Value::integer(ctx.mutation(), 1), Some(mt));
        let got = ctx
            .metatable_of(Value::float(2.5))
            .expect("number metatable");
        assert!(Gc::ptr_eq(got.inner(), mt.inner()));
        assert!(ctx.metatable_of(Value::nil()).is_none());
        assert!(ctx.metatable_of(Value::boolean(true)).is_none());
        let s = Value::string(LuaString::new(ctx, b"x"));
        assert!(ctx.metatable_of(s).is_none());
    });
}

#[test]
fn type_metatable_survives_collection() {
    let mut lua = Lua::new();
    lua.enter(|ctx| {
        let mt = Table::new(ctx);
        let k = Value::string(LuaString::new(ctx, b"k"));
        mt.raw_set(ctx, k, Value::integer(ctx.mutation(), 7));
        ctx.set_metatable_of(Value::string(LuaString::new(ctx, b"")), Some(mt));
    });
    lua.collect_all();
    lua.enter(|ctx| {
        let s = Value::string(LuaString::new(ctx, b"other"));
        let mt = ctx.metatable_of(s).expect("string metatable");
        let k = Value::string(LuaString::new(ctx, b"k"));
        assert_eq!(mt.raw_get(k).get_integer(), Some(7));
        ctx.set_metatable_of(s, None);
        assert!(ctx.metatable_of(s).is_none());
    });
}

#[test]
fn getmetatable_sees_type_metatables() {
    assert_eq!(
        ok("local mt = {__metatable = 'locked'}
            return tostring(debug.setmetatable(10, mt)) .. ' ' .. getmetatable(20)
                .. ' ' .. tostring(debug.getmetatable(1.5) == mt)
                .. ' ' .. tostring(debug.getmetatable(print))"),
        "10 locked true nil"
    );
    assert_eq!(
        ok("local mt = {}
            debug.setmetatable(print, mt)
            return tostring(getmetatable(function() end) == mt)
                .. ' ' .. tostring(getmetatable(coroutine.create(print)))"),
        "true nil"
    );
    assert_eq!(
        ok(
            "debug.setmetatable(nil, {x = 1}); debug.setmetatable(true, {y = 2})
            local a, b = getmetatable(nil).x, getmetatable(false).y
            debug.setmetatable(nil, nil)
            return a .. ' ' .. b .. ' ' .. tostring(getmetatable(nil))"
        ),
        "1 2 nil"
    );
}

#[test]
fn debug_setmetatable_on_tables_and_userdata() {
    assert_eq!(
        ok("local t = setmetatable({}, {__metatable = false})
            local r = tostring(getmetatable(t)) .. ' ' .. tostring(debug.getmetatable(t) ~= nil)
            return r .. ' ' .. tostring(debug.setmetatable(t, nil) == t) .. ' ' .. tostring(getmetatable(t))"),
        "false true true nil"
    );
    assert_eq!(
        ok("local f = io.stdout; local mt = getmetatable(f)
            local r = tostring(debug.setmetatable(f, nil) == f) .. ' ' .. tostring(getmetatable(f))
            debug.setmetatable(f, mt)
            return r .. ' ' .. tostring(getmetatable(f) == mt)"),
        "true nil true"
    );
}

#[test]
fn metatable_argument_errors() {
    assert_eq!(
        err("debug.setmetatable(1)"),
        "c:1: bad argument #2 to 'setmetatable' (nil or table expected, got no value)"
    );
    assert_eq!(
        err("debug.setmetatable(1, 2)"),
        "c:1: bad argument #2 to 'setmetatable' (nil or table expected, got number)"
    );
    assert_eq!(
        err("getmetatable()"),
        "c:1: bad argument #1 to 'getmetatable' (value expected)"
    );
    assert_eq!(
        err("debug.getmetatable()"),
        "c:1: bad argument #1 to 'getmetatable' (value expected)"
    );
}
