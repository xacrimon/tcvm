//! Weak tables (#52). Expected results are lua 5.5.1's, running the same
//! chunks with `collectgarbage()` in place of the host's collections.

use tcvm::env::LuaString;
use tcvm::{Lua, RuntimeError};

use crate::common::{start_on, yielding_lua};

/// Defines the global `count(t)`, the number of entries `pairs` visits.
const COUNT: &str =
    "function count(t) local n = 0 for _ in pairs(t) do n = n + 1 end return n end ";

/// [`COUNT`], then each `--gc`-separated chunk of `src` run to completion with
/// a full collection after it; returns the last one's string result.
/// `yielder()` collects mid-chunk. Garbage is made in an earlier chunk than the
/// check that it is gone, since a returned call's temporaries can outlive a
/// `yielder()` (#229).
fn run_in(lua: &mut Lua, src: &str) -> String {
    let src = format!("{COUNT}{src}");
    let mut result = String::new();
    for chunk in src.split("\n--gc\n") {
        let ex = start_on(lua, chunk);
        let mut step = lua.finish(&ex);
        while let Err(RuntimeError::MainYielded) = step {
            lua.collect_all();
            step = lua.resume(&ex, ());
        }
        step.expect("run");
        result = lua
            .try_enter(|ctx| {
                let r = ctx.fetch(&ex).take_result::<Option<LuaString>>(ctx)?;
                Ok::<_, RuntimeError>(r.map_or(String::new(), |s| {
                    String::from_utf8_lossy(s.as_bytes()).into_owned()
                }))
            })
            .expect("take result");
        drop(ex);
        lua.collect_all();
    }
    result
}

fn run(src: &str) -> String {
    run_in(&mut yielding_lua(), src)
}

/// `__mode` is any string containing `k` and/or `v`; other values leave the table strong.
#[test]
fn modes() {
    let src = r#"
ts = {}
for i, m in ipairs({"k", "v", "kv", "vk", "xkx", "K", "", 1}) do
  local t = setmetatable({}, {__mode = m}) t[{}] = 1 t[1] = {} ts[i] = t
end
--gc
local out = {}
for i, t in ipairs(ts) do out[i] = count(t) end
return table.concat(out, " ")
"#;
    assert_eq!(run(src), "1 1 0 0 1 2 2 2");
}

/// Strings, numbers and booleans are values, never removed; only the objects go.
#[test]
fn values_stay() {
    let src = r#"
w = setmetatable({}, {__mode = "kv"})
w[1] = string.rep("a", 40) w[string.rep("b", 40)] = 2 w[2] = 1 << 40 w[3] = 1.5
w[true] = {} w[4.5] = string.rep("x", 3) w[{}] = "y"
--gc
return count(w) .. " " .. #w[1] .. " " .. w[string.rep("b", 40)] .. " " .. w[2] .. " " .. w[3]
  .. " " .. tostring(w[true]) .. " " .. w[4.5]
"#;
    assert_eq!(run(src), "5 40 2 1099511627776 1.5 nil xxx");
}

/// A weak key's value is reachable only through the key: a self-cycle goes, a chain held
/// from its head stays until the head goes.
#[test]
fn ephemeron() {
    let src = r#"
e = setmetatable({}, {__mode = "k"})
keep = {}
e[keep] = {"kept"}
local k = {} e[k] = {k}
-- Inserted tail first, so settling the chain takes a pass per link.
local a, b, c = {}, {}, {} e[c] = "end" e[b] = c e[a] = b
root = a
--gc
n = count(e)
root = nil
--gc
return n .. " " .. count(e) .. " " .. e[keep][1]
"#;
    assert_eq!(run(src), "4 1 kept");
}

/// Ephemeron reachability crosses tables.
#[test]
fn ephemeron_across() {
    let src = r#"
e1 = setmetatable({}, {__mode = "k"})
e2 = setmetatable({}, {__mode = "k"})
local a, b = {}, {} e2[b] = {"via e1"} e1[a] = b
root = a
--gc
n = count(e1) .. count(e2)
root = nil
--gc
return n .. " " .. count(e1) .. count(e2)
"#;
    assert_eq!(run(src), "11 00");
}

/// Weak values are cleared from every part: array, integer hash, shape slots, misc keys and
/// dict mode.
#[test]
fn weak_value_parts() {
    let src = r#"
local mt = {__mode = "v"}
keep = {}
t1, t2 = setmetatable({}, mt), setmetatable({}, mt)
t1[1] = {} t1[2] = keep
t1[1 << 40] = {} t1[(1 << 40) + 1] = keep
t1.a = {} t1.b = keep
t1[true] = {} t1[false] = keep
for i = 1, 70 do t2["k" .. i] = (i % 2 == 0) and keep or {} end
--gc
return count(t1) .. " " .. tostring(t1[2] == keep and t1.b == keep and t1[false] == keep
  and t1[(1 << 40) + 1] == keep) .. " " .. count(t2) .. " " .. tostring(t2.k70 == keep)
"#;
    assert_eq!(run(src), "4 true 35 true");
}

/// Weak values with strong keys: an object key survives while its value does.
#[test]
fn weak_value_keys() {
    let src = r#"
wv = setmetatable({}, {__mode = "v"})
keep = {}
wv[{}] = keep wv[keep] = {}
--gc
local k, v = next(wv)
return count(wv) .. " " .. tostring(v == keep) .. " " .. tostring(k ~= keep)
"#;
    assert_eq!(run(src), "1 true true");
}

/// `__mode` set or changed after `setmetatable` applies from the next cycle.
#[test]
fn late_mode() {
    let src = r#"
mt = {}
t = setmetatable({}, mt)
t[1] = {} t[{}] = 1
mt.__mode = "k"
--gc
a = count(t)
rawset(mt, "__mode", "v")
--gc
b = count(t)
mt.__mode = nil
t[2] = {}
--gc
return a .. " " .. b .. " " .. count(t)
"#;
    assert_eq!(run(src), "1 0 1");
}

/// `next` resumes from a key a collection cleared mid-traversal.
#[test]
fn pairs_across_clear() {
    let src = r#"
t = setmetatable({}, {__mode = "k"})
keep = {}
for i = 1, 5 do keep[i] = {} t[keep[i]] = true end
for i = 1, 20 do t[{}] = true end
--gc
-- Collects mid-traversal: next must resume past keys cleared under it.
local function fill() for i = 1, 20 do t[{}] = true end end
fill()
local seen, first = 0, true
for k in pairs(t) do
  if first then first = false yielder() end
  for i = 1, 5 do if k == keep[i] then seen = seen + 1 end end
end
seen_all = seen
--gc
return seen_all .. " " .. count(t)
"#;
    assert_eq!(run(src), "5 5");
}

/// A weak-valued metatable losing `__index` stops dispatching it.
#[test]
fn weak_metatable() {
    let src = r#"
mt = setmetatable({}, {__mode = "v"})
obj = setmetatable({}, mt)
mt.__index = {x = 1}
before = obj.x
--gc
return tostring(before) .. " " .. tostring(obj.x) .. " " .. tostring(rawget(mt, "__index"))
"#;
    assert_eq!(run(src), "1 nil nil");
}

/// A load cached through `__index` doesn't keep a weak-valued metatable's
/// `__index` alive, and the second collection empties the stale entry.
#[test]
fn weak_metatable_cached_index() {
    let src = r#"
mt = setmetatable({}, {__mode = "v"})
obj = setmetatable({}, mt)
mt.__index = {x = 1}
function get() return obj.x end
before = get()
--gc
gone = rawget(mt, "__index")
--gc
local after = get()
mt.__index = {x = 2}
return tostring(before) .. " " .. tostring(after) .. " " .. tostring(gone) .. " " .. tostring(get())
"#;
    assert_eq!(run(src), "1 nil nil 2");
}

/// Cache entries whose `__index` table was collected stop holding its memory.
#[test]
fn collected_index_tables_are_freed() {
    const SITES: usize = 2000;
    let live = |call: &str| {
        let mut lua = yielding_lua();
        let src = format!(
            "fs = {{}} for i = 1, {SITES} do \
             local mt = setmetatable({{}}, {{__mode = 'v'}}) mt.__index = {{x = i}} \
             local obj = setmetatable({{}}, mt) \
             local f = load('local obj = ... return function() return obj.x end')(obj) \
             {call} fs[i] = f end\n--gc\n"
        );
        run_in(&mut lua, &src);
        lua.live_bytes()
    };
    let (uncached, cached) = (live(""), live("f()"));
    assert!(
        cached < uncached + 8 * SITES,
        "uncached={uncached} cached={cached}"
    );
}

/// String values of surviving weak keys stay alive and interned.
#[test]
fn string_values() {
    let src = r#"
e = setmetatable({}, {__mode = "k"})
kv = setmetatable({}, {__mode = "kv"})
keys = {}
for i = 1, 10 do keys[i] = {} e[keys[i]] = "v" .. i kv[keys[i]] = "w" .. i end
--gc
for i = 1, 10 do
  if e[keys[i]] ~= "v" .. i or kv[keys[i]] ~= "w" .. i then error("lost " .. i) end
end
local probe = {}
probe["v" .. 3] = 1
return count(e) .. " " .. count(kv) .. " " .. tostring(probe[e[keys[3]]])
"#;
    assert_eq!(run(src), "10 10 1");
}

/// Weak entries hold up while the collector runs incrementally alongside the script.
#[test]
fn incremental_collection() {
    let src = r#"
cache = setmetatable({}, {__mode = "k"})
eph = setmetatable({}, {__mode = "k"})
vals = setmetatable({}, {__mode = "v"})
keep = {}
for i = 1, 200 do keep[i] = {} cache[keep[i]] = {keep[i], "s" .. i} vals[i] = keep[i] end
for round = 1, 3000 do
  for j = 1, 20 do local k = {} eph[k] = {k, round} vals[200 + j] = {} end
  for i = 1 + round % 7, 200, 7 do
    local v = cache[keep[i]]
    if v == nil or v[1] ~= keep[i] or v[2] ~= "s" .. i then error("lost cache entry " .. i) end
    if vals[i] ~= keep[i] then error("lost value " .. i) end
  end
end
--gc
return count(cache) .. " " .. count(vals) .. " " .. count(eph)
"#;
    assert_eq!(run(src), "200 200 0");
}

/// `__mode` flips between a table's trace and the end of marking. Strongly held entries
/// survive, and under Guard Malloc an edge left uncleared crashes.
#[test]
fn mode_change_mid_cycle() {
    let src = r#"
-- Enough strong heap that marking spans several steps, so the flips land mid-mark.
local heap = {}
for i = 1, 10000 do heap[i] = {i} end
local R = 8000
local mt = {__mode = "kv"}
local modes = {"kv", false, "v", false, "k", false}
local keep, pool, tabs = {}, {}, {}
for i = 1, 20 do keep[i] = {} end
for i = 1, R do pool[i] = {} end
-- Filled once: a write would re-gray a table and retrace it under the new mode.
for i = 1, 20 do
  local t = setmetatable({}, mt)
  for j = 1, 200 do
    local a, b = pool[(i * 200 + j) % R + 1], pool[(i * 131 + j * 7) % R + 1]
    t[a] = b t[j] = a
    if i % 2 == 0 or j <= 30 then t["p" .. j] = b end -- even tables go to dict mode
  end
  for j = 1, 20 do t[keep[j]] = keep[21 - j] t[-j] = keep[j] t["k" .. j] = keep[j] end
  tabs[i] = t
end
for round = 1, R do
  pool[round] = nil
  local junk = {} for j = 1, 8 do junk[j] = {} end
  if round % 25 == 0 then mt.__mode = modes[round // 25 % 6 + 1] or nil end
  local t = tabs[round % 20 + 1]
  for j = 1, 20 do
    if t[keep[j]] ~= keep[21 - j] or t[-j] ~= keep[j] or t["k" .. j] ~= keep[j] then
      error("lost entry " .. j)
    end
  end
end
return "ok"
"#;
    assert_eq!(run(src), "ok");
}

/// Closures and coroutines are objects: weak values and weak keys drop them unless held.
#[test]
fn function_and_thread_entries() {
    let src = r#"
wv = setmetatable({}, {__mode = "v"})
ek = setmetatable({}, {__mode = "k"})
keepf, keepc = {}, {}
for i = 1, 10 do
  local f = function() return i end
  local c = coroutine.create(function() coroutine.yield(i) end)
  wv[i] = f wv[100 + i] = c
  ek[f] = {f, i} ek[c] = {c, i}
  if i % 2 == 0 then keepf[i] = f keepc[i] = c end
end
--gc
local ok = true
for i = 2, 10, 2 do
  ok = ok and wv[i] == keepf[i] and wv[100 + i] == keepc[i]
    and ek[keepf[i]][2] == i and ek[keepc[i]][2] == i and keepf[i]() == i
    and select(2, coroutine.resume(keepc[i])) == i
end
return count(wv) .. " " .. count(ek) .. " " .. tostring(ok)
"#;
    assert_eq!(run(src), "10 10 true");
}

/// Objects held only by weak tables are freed, cycles through ephemerons included.
#[test]
fn weakly_held_objects_are_freed() {
    let live = |src: &str| {
        let mut lua = yielding_lua();
        run_in(&mut lua, src);
        lua.live_bytes()
    };
    let baseline = live("t = setmetatable({}, {__mode = 'k'})");
    let strong = live("t = {} for i = 1, 20000 do local k = {} t[k] = {k} end");
    let weak = live(
        "t = setmetatable({}, {__mode = 'k'}) \
         for i = 1, 20000 do local k = {} t[k] = {k} end",
    );
    let payload = strong.saturating_sub(baseline);
    assert!(
        payload > 1_000_000,
        "payload too small: baseline={baseline} strong={strong}"
    );
    let retained = weak.saturating_sub(baseline);
    assert!(
        retained < payload / 10,
        "ephemerons retained: baseline={baseline} weak={weak} strong={strong}"
    );
}

/// `__mode` stored on live metatables from one site: the later stores reuse
/// the first one's cached shape transition and must still weaken the table.
#[test]
fn mode_added_by_a_cached_store() {
    let src = r#"
ws = {}
local function weaken(m) m.__mode = 'k' end
for i = 1, 3 do
  local w = setmetatable({}, {})
  weaken(getmetatable(w))
  w[{}] = 1
  ws[i] = w
end
--gc
local out = {}
for i, w in ipairs(ws) do out[i] = count(w) end
return table.concat(out, ' ')
"#;
    assert_eq!(run(src), "0 0 0");
}
