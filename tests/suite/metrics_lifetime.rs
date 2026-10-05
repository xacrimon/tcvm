//! Allocators and dynamic roots point at the arena's metrics without owning
//! them (#184). These exercise every holder through collection and arena drop
//! without running the interpreter, so they also run under Miri.

use tcvm::dmm::Mutation;
use tcvm::env::{LuaString, Table, Value};
use tcvm::{Context, Lua};

fn fill(ctx: Context<'_>) {
    let mc: &Mutation<'_> = ctx.mutation();
    let t = Table::new(ctx);
    for i in 0..64 {
        let k = Value::string(LuaString::new(ctx, format!("k{i}").as_bytes()));
        t.raw_set(ctx, k, Value::integer(mc, i));
        t.raw_set(ctx, Value::integer(mc, i + 1000), Value::boolean(true));
    }
    let key = Value::string(LuaString::new(ctx, b"kept"));
    ctx.globals().raw_set(ctx, key, Value::table(t));
}

#[test]
fn holders_survive_collection_and_arena_drop() {
    let mut lua = Lua::new();
    lua.enter(fill);
    let stashed = lua.enter(|ctx| ctx.stash(Table::new(ctx)));
    lua.collect_all();
    lua.enter(fill);
    drop(lua);
    // A dynamic root dropped after its arena must not touch the freed metrics.
    drop(stashed);
}
