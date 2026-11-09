//! The interpreter pays the collector for what its stores allocate, as compiled code does.

use tcvm::Lua;

use crate::common::run_steps;

#[test]
fn growing_a_table_pays_the_collector() {
    let mut lua = Lua::new();
    lua.stop_gc_stress();
    lua.set_jit(false);
    lua.load_all();
    // An 8 MiB array part, reached by doubling: without a collection the parts it outgrew add
    // as much again.
    let src = "local t = {}
        for i = 1, 1 << 20 do t[i] = i end
        return tostring(collectgarbage('count') // 1024)";
    let mib: f64 = run_steps(&mut lua, src).0.parse().expect("a number");
    assert!(mib < 12.0, "{mib} MiB in use");
}
