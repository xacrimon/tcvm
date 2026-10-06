//! Minor collections: everything that survived a collection is old and kept, so whatever old
//! objects were given since must survive too, and only objects allocated since can be freed.

use tcvm::env::LuaString;
use tcvm::{Lua, RuntimeError};

use crate::common::start_on;

/// Run each chunk of `src`, separated by `--full` or `--minor` lines, to completion, then that
/// collection. Returns the last chunk's string result and the live bytes before and after each
/// collection.
fn run_steps(lua: &mut Lua, src: &str) -> (String, Vec<(usize, usize)>) {
    let mut result = String::new();
    let mut live = Vec::new();
    let mut chunk = String::new();
    for line in src.lines().chain(["--end"]) {
        let full = match line {
            "--full" => true,
            "--minor" | "--end" => false,
            _ => {
                chunk.push_str(line);
                chunk.push('\n');
                continue;
            }
        };
        let ex = start_on(lua, &chunk);
        lua.finish(&ex).expect("run");
        result = lua
            .try_enter(|ctx| {
                let r = ctx.fetch(&ex).take_result::<Option<LuaString>>(ctx)?;
                Ok::<_, RuntimeError>(r.map_or(String::new(), |s| {
                    String::from_utf8_lossy(s.as_bytes()).into_owned()
                }))
            })
            .expect("take result");
        drop(ex);
        chunk.clear();
        if line == "--end" {
            break;
        }
        let before = lua.live_bytes();
        if full {
            lua.collect_all();
        } else {
            lua.collect_young();
        }
        live.push((before, lua.live_bytes()));
    }
    (result, live)
}

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
