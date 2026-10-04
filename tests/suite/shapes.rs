//! The shape tree behind string-keyed fields. Expected results are lua
//! 5.5.1's, running the same chunks with `collectgarbage()` in place of the
//! host's collections.

use tcvm::Lua;

use crate::common::run_chunks;

fn run(src: &str) -> String {
    let mut lua = Lua::new();
    lua.load_all();
    run_chunks(&mut lua, src)
}

/// Defines `keys(t)`, the sorted `k=v` pairs `pairs` visits.
const KEYS: &str = "local function keys(t) local ks = {} \
    for k, v in pairs(t) do ks[#ks + 1] = k .. '=' .. v end \
    table.sort(ks) return table.concat(ks, ',') end ";

/// Shapes along a chain share one key array: a branch off it, a metatable
/// sibling, and a shape whose longer sibling died each add a key correctly.
#[test]
fn shared_keys() {
    let src = r#"
a = {}; a.x = 1; a.y = 2; a.z = 3
b = {}; b.x = 1; b.y = 2; b.w = 4
c = {}; c.x = 1; c.y = 2; setmetatable(c, {}); c.v = 5
e = {}; e.x = 1; e.y = 2; e.z = 3; setmetatable(e, {}); e.v = 6
a.u = 7
keep = {}; keep.p = 1; keep.q = 2
s = {}; s.p = 1; s.q = 2; s['dead' .. 1] = 3
s = nil
--gc
keep.r = 3
"#;
    let check = format!(
        "{KEYS}return table.concat({{keys(a), keys(b), keys(c), keys(e), keys(keep), \
         tostring(keep['dead' .. 1])}}, ' ')"
    );
    assert_eq!(
        run(&format!("{src}{check}")),
        "u=7,x=1,y=2,z=3 w=4,x=1,y=2 v=5,x=1,y=2 v=6,x=1,y=2,z=3 p=1,q=2,r=3 nil"
    );
}
