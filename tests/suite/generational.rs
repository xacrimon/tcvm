//! Minor collections: everything that survived a collection is old and kept, so whatever old
//! objects were given since must survive too, and only objects allocated since can be freed.

use tcvm::Lua;

use crate::common::run_steps;

fn run(src: &str) -> String {
    let mut lua = Lua::new();
    lua.load_all();
    run_steps(&mut lua, src).0
}

#[test]
fn old_table_keeps_young_values() {
    let src = r#"
t = {}
--full
for i = 1, 300 do t[i] = {i} end
t.x = {"x"}
--minor
--minor
local s = 0
for i = 1, 300 do s = s + t[i][1] end
return s .. t.x[1]
"#;
    assert_eq!(run(src), "45150x");
}

#[test]
fn old_closure_keeps_young_upvalue() {
    let src = r#"
local up = 0
function get() return up end
function set(v) up = v end
--full
set({7})
--minor
return tostring(get()[1])
"#;
    assert_eq!(run(src), "7");
}

#[test]
fn old_table_keeps_young_metatable() {
    let src = r#"
t = {}
--full
setmetatable(t, {__index = function() return 5 end})
--minor
return tostring(t.anything)
"#;
    assert_eq!(run(src), "5");
}

#[test]
fn old_coroutine_keeps_young_locals() {
    let src = r#"
co = coroutine.create(function()
  local x = coroutine.yield()
  coroutine.yield()
  coroutine.yield(x[1])
end)
coroutine.resume(co)
--full
coroutine.resume(co, {42})
--minor
return tostring(select(2, coroutine.resume(co)))
"#;
    assert_eq!(run(src), "42");
}

#[test]
fn old_weak_table_loses_young_objects() {
    let src = r#"
w = setmetatable({}, {__mode = "v"})
keep = {}
--full
w[1] = {}
w[2] = keep
--minor
return tostring(w[1] == nil) .. " " .. tostring(w[2] == keep)
"#;
    assert_eq!(run(src), "true true");
}

#[test]
fn old_table_keeps_young_huge_string() {
    let src = r#"
t = {}
--full
t.big = string.rep("x", 100000)
local garbage = string.rep("y", 100000)
--minor
return tostring(#t.big)
"#;
    assert_eq!(run(src), "100000");
}

#[test]
fn young_strings_die_and_intern_again() {
    let src = r#"
--full
local a = "young" .. 1
--minor
b = "young" .. 1
--minor
return tostring(b == "young1")
"#;
    assert_eq!(run(src), "true");
}

#[test]
fn written_after_a_full_collection_is_remembered_again() {
    let src = r#"
t = {}
--full
t[1] = {1}
--full
t[2] = {2}
--minor
return t[1][1] .. t[2][1]
"#;
    assert_eq!(run(src), "12");
}

/// A minor collection frees young garbage and keeps old garbage, which a full one frees.
#[test]
fn minor_frees_only_young_garbage() {
    let src = r#"
old = {}
for i = 1, 20000 do old[i] = {i} end
--full
old = nil
for i = 1, 20000 do local x = {i} end
--minor
--full
"#;
    let mut lua = Lua::new();
    lua.load_all();
    let (_, live) = run_steps(&mut lua, src);
    let [(_, built), (young, minor), (_, full)] = live[..] else {
        panic!("{live:?}")
    };
    let payload = built.saturating_sub(full);
    assert!(payload > 500_000, "payload too small: {live:?}");
    assert!(
        young.saturating_sub(minor) > payload / 2,
        "young garbage kept: {live:?}"
    );
    assert!(
        minor.saturating_sub(full) > payload / 2,
        "old garbage freed by a minor collection: {live:?}"
    );
}
