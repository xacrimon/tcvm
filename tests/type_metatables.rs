//! Per-type metatables (`G(L)->mt` in PUC Lua): every value of a type other
//! than table and userdata shares one metatable slot.

use tcvm::Lua;
use tcvm::dmm::Gc;
use tcvm::env::{LuaString, Table, Value};

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
