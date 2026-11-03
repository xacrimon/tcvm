//! Feedback holds what it caches weakly: a collection that frees a cached
//! shape, `__index` holder or callee empties the entry, so memory reused for
//! another object never matches it. Each test churns allocation after the
//! collection to reuse what it freed. Expected strings come from `lua` 5.5.1
//! running the chunks with `collectgarbage()` for each `--full`/`--minor`.

use tcvm::Lua;

use crate::common::run_steps;

fn run(src: &str) -> String {
    let mut lua = Lua::new();
    lua.load_all();
    run_steps(&mut lua, src).0
}

/// A read site cached a shape only dead tables had.
#[test]
fn dead_own_shape() {
    let src = r#"
function get(t) return t.x end
function mk(i) local t = {} t.p = i t.x = i * 2 return t end
for i = 1, 300 do get(mk(i)) end
--full
junk = {}
for i = 1, 3000 do local t = {} t.q = i t.r = -i junk[i] = t end
local r = {}
for i = 1, 5 do r[#r + 1] = tostring(get(junk[i])) end
return table.concat(r, ",") .. " " .. get(mk(4))
"#;
    assert_eq!(run(src), "nil,nil,nil,nil,nil 8");
}

/// A store site cached a transition whose target only dead tables had.
#[test]
fn dead_transition_target() {
    let src = r#"
function mk(i) local t = {} t.a = i t.b = i + 1 return t end
for i = 1, 300 do mk(i) end
--full
junk = {}
for i = 1, 3000 do local t = {} t.c = i t.d = i junk[i] = t end
local t = mk(5)
local n = 0
for _, v in pairs(t) do n = n + v end
return t.a .. " " .. t.b .. " " .. n .. " " .. tostring(t.c)
"#;
    assert_eq!(run(src), "5 6 11 nil");
}

/// A read site cached an `__index` holder that died.
#[test]
fn dead_index_holder() {
    let src = r#"
function get(o) return o.m end
function mk(v) return setmetatable({}, {__index = {m = v}}) end
for i = 1, 300 do get(mk(i)) end
--full
junk = {}
for i = 1, 3000 do junk[i] = {m = -i} end
return get(mk(77)) .. " " .. get(mk(78))
"#;
    assert_eq!(run(src), "77 78");
}

/// A call site's callee died: one closure, then a prototype whose closures
/// all died.
#[test]
fn dead_callees() {
    let src = r#"
function call(f) return f() end
do local g = function() return 1 end for _ = 1, 300 do call(g) end end
function call2(f) return f() end
local function mk(i) return function() return i end end
for i = 1, 300 do call2(mk(i)) end
--full
junk = {}
for i = 1, 3000 do junk[i] = function() return i end end
return call(function() return 7 end) .. " " .. call2(function() return 8 end)
"#;
    assert_eq!(run(src), "7 8");
}

/// An old prototype's feedback gains young shapes and callees, which a minor
/// collection frees.
#[test]
fn young_feedback_of_an_old_prototype() {
    let src = r#"
function get(t) return t.x end
function call(f) return f() end
get({x = 1})
--full
local function mk(i) local t = {} t.y = i t.x = i return t end
for i = 1, 300 do get(mk(i)) end
do local g = function() return 1 end for _ = 1, 300 do call(g) end end
--minor
junk = {}
for i = 1, 3000 do local t = {} t.u = i t.v = i junk[i] = t end
for i = 1, 3000 do junk[3000 + i] = function() return i end end
local r = {}
for i = 1, 5 do r[#r + 1] = tostring(get(junk[i])) end
return table.concat(r, ",") .. " " .. get({x = 3}) .. " " .. call(function() return 9 end)
"#;
    assert_eq!(run(src), "nil,nil,nil,nil,nil 3 9");
}

/// Field and call sites past the feedback vector's 65,535 words share one
/// slot that never caches or records.
#[test]
fn sites_past_the_vector() {
    let src = r#"
local lines = {"local t, s = {x = 1}, 0", "local function f() return 1 end", "for _ = 1, 3 do"}
for _ = 1, 11000 do lines[#lines + 1] = "s = s + t.x + f()" end
lines[#lines + 1] = "end return s"
return load(table.concat(lines, "\n"))()
"#;
    assert_eq!(crate::common::eval::<i64>(src), 66000);
}
