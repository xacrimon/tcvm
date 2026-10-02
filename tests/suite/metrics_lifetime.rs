//! Allocators and dynamic roots point at the arena's metrics without owning
//! them (#184). These exercise every holder through collection and arena drop
//! without running the interpreter, so they also run under Miri.

use std::pin::Pin;

use tcvm::dmm::{Gc, Mutation, Trace};
use tcvm::env::{Error, LuaString, Table, Value};
use tcvm::vm::sequence::{BoxSequence, Execution, Sequence, SequencePoll};
use tcvm::{Context, Lua};

/// A sequence that never runs: only its allocation and drop matter here.
struct Idle;

impl<'gc> Sequence<'gc> for Idle {
    fn trace_pointers(&self, _cc: &mut dyn Trace<'gc>) {}

    fn poll(
        self: Pin<&mut Self>,
        _ctx: Context<'gc>,
        _exec: Execution<'gc>,
        _stack: tcvm::env::Stack<'gc, '_>,
    ) -> Result<SequencePoll<'gc>, Error<'gc>> {
        unreachable!()
    }
}

fn fill(ctx: Context<'_>) {
    let mc: &Mutation<'_> = ctx.mutation();
    let t = Table::new(ctx);
    for i in 0..64 {
        let k = Value::string(LuaString::new(ctx, format!("k{i}").as_bytes()));
        t.raw_set(ctx, k, Value::integer(mc, i));
        t.raw_set(ctx, Value::integer(mc, i + 1000), Value::boolean(true));
    }
    let _ = Gc::new(mc, BoxSequence::new(mc, Idle));
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
