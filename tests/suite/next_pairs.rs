//! `next` / `pairs` over the array, string (shape and dict) and misc-hash
//! parts. Traversal order is implementation-defined, so tests compare
//! sorted `k=v` lists. Expected strings come from `lua` 5.5.1 on the same
//! snippets.

use tcvm::env::LuaString;
use tcvm::{Executor, LoadError, Lua, RuntimeError};

fn run_i64(src: &str) -> i64 {
    let mut lua = Lua::new();
    lua.load_all();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load(src, Some("=t"))?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    lua.execute(&ex).expect("run")
}

/// The string `src` returns.
fn run_str(src: &str) -> String {
    let mut lua = Lua::new();
    lua.load_all();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load(src, Some("=t"))?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    lua.finish(&ex).expect("run");
    lua.try_enter(|ctx| {
        let s = ctx.fetch(&ex).take_result::<LuaString>(ctx)?;
        Ok::<_, RuntimeError>(String::from_utf8_lossy(s.as_bytes()).into_owned())
    })
    .expect("result")
}

/// Live GC bytes after running `src` to completion and a full collection.
fn live_bytes_after(src: &str) -> usize {
    let mut lua = Lua::new();
    lua.load_all();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load(src, Some("=t"))?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    lua.finish(&ex).expect("run");
    lua.collect_all();
    lua.live_bytes()
}

const KEYS: &str = "local function keys(t)
  local ks, n = {}, 0
  for k, v in pairs(t) do n = n + 1; ks[#ks+1] = tostring(k) .. '=' .. tostring(v) end
  table.sort(ks)
  return n .. ':' .. table.concat(ks, ' ')
end\n";

/// Run `body` with `keys` in scope; it returns `n:k=v k=v …` for a table.
fn keys(body: &str) -> String {
    run_str(&format!("{KEYS}{body}"))
}

fn check(src: &str) {
    assert_eq!(run_i64(&format!("return ({src}) and 1 or 0")), 1, "{src}");
}

/// Message of the error `call` raises under `pcall`.
fn err_msg(call: &str) -> String {
    run_str(&format!("return select(2, pcall({call}))"))
}

#[test]
fn empty() {
    assert_eq!(keys("return keys({})"), "0:");
}

#[test]
fn array_part() {
    assert_eq!(keys("return keys({10, 20, 30})"), "3:1=10 2=20 3=30");
}

#[test]
fn every_part_with_holes() {
    assert_eq!(
        keys("return keys({10, nil, 30, a = 1, b = 2, [true] = 3, [2.5] = 4, [-1] = 5})"),
        "7:-1=5 1=10 2.5=4 3=30 a=1 b=2 true=3"
    );
}

#[test]
fn dict_mode_strings() {
    // Past MAX_PROPERTIES_FAST (64), so the string part is a dict.
    let got = keys("local t = {} for i = 1, 70 do t['k' .. i] = i end return keys(t)");
    assert!(got.starts_with("70:k10=10 k11=11"), "{got}");
    assert!(got.ends_with("k8=8 k9=9"), "{got}");
}

#[test]
fn clear_every_entry_mid_traversal() {
    assert_eq!(
        keys(
            "local u = {1, 2, 3, a = 1, b = 2, c = 3, [false] = 1, [1.5] = 2}
             for k in pairs(u) do u[k] = nil end
             return tostring(next(u)) .. ' ' .. keys(u)"
        ),
        "nil 0:"
    );
}

#[test]
fn clear_dict_mode_mid_traversal() {
    assert_eq!(
        keys(
            "local t = {} for i = 1, 70 do t['k' .. i] = i end
             for k in pairs(t) do t[k] = nil end
             return tostring(next(t)) .. ' ' .. keys(t)"
        ),
        "nil 0:"
    );
}

/// Deleting must not rehash even when the hash part is exactly full, or the
/// scan would jump over entries. Sweeping sizes 1..=256 crosses every
/// hashbrown capacity boundary for both the misc and (past 64) dict parts.
#[test]
fn clear_full_hash_mid_traversal() {
    assert_eq!(
        run_str(
            "local bad = {}
             for n = 1, 256 do
               local m, d = {}, {}
               for i = 1, n do m[-i] = i d['k' .. i] = i end
               local c = 0
               for k in pairs(m) do m[k] = nil m[-999] = nil c = c + 1 end
               if c ~= n or next(m) ~= nil then bad[#bad + 1] = 'm' .. n end
               c = 0
               for k in pairs(d) do d[k] = nil d.absent = nil c = c + 1 end
               if c ~= n or next(d) ~= nil then bad[#bad + 1] = 'd' .. n end
             end
             return table.concat(bad, ' ')"
        ),
        ""
    );
}

#[test]
fn assign_existing_mid_traversal() {
    assert_eq!(
        keys(
            "local w = {a = 1, b = 2, c = 3, [true] = 1} for k, v in pairs(w) do w[k] = v * 10 end return keys(w)"
        ),
        "4:a=10 b=20 c=30 true=10"
    );
}

#[test]
fn deleted_slot_is_reused() {
    assert_eq!(
        keys("local r = {x = 1} r.x = nil r.x = 2 return keys(r)"),
        "1:x=2"
    );
}

#[test]
fn churned_keys_do_not_resurface() {
    // Churn far more distinct keys through the misc hash than stay live;
    // the dead entries must neither show up in `pairs` nor block a re-set.
    assert_eq!(
        keys("local t = {} for i = 1, 1000 do t[-i] = i t[-i] = nil end t[-5] = 5 return keys(t)"),
        "1:-5=5"
    );
}

#[test]
fn manual_next_chain() {
    check(
        "(function() local c = {5, 6, z = 1} local k = next(c) local n = 0 while k do n = n + 1 k = next(c, k) end return n end)() == 3",
    );
    check("next({}) == nil and select('#', next({})) == 1");
}

#[test]
fn invalid_key() {
    check("pcall(next, {}, 'nope') == false");
    assert_eq!(err_msg("next, {}, 'nope'"), "invalid key to 'next'");
    assert_eq!(err_msg("next, {}, true"), "invalid key to 'next'");
}

#[test]
fn bad_argument() {
    assert_eq!(
        err_msg("next, 1"),
        "bad argument #1 to 'next' (table expected, got number)"
    );
    assert_eq!(
        err_msg("next"),
        "bad argument #1 to 'next' (table expected, got no value)"
    );
    assert_eq!(
        err_msg("pairs"),
        "bad argument #1 to 'pairs' (value expected)"
    );
}

#[test]
fn pairs_returns_four_values() {
    check("select('#', pairs({})) == 4");
    check("pairs({}) == next");
    // Only presence is checked; a non-table fails later in `next`.
    check("select(2, pairs(1)) == 1");
}

#[test]
fn pairs_metamethod() {
    check(
        "(function()
            local mt = setmetatable({}, {__pairs = function(t)
                return function(_, k) if not k then return 1, 'one' end end, t, nil
            end})
            local n, lk, lv = 0
            for k, v in pairs(mt) do n = n + 1 lk, lv = k, v end
            return n == 1 and lk == 1 and lv == 'one'
        end)()",
    );
    // Results are adjusted to exactly four, like `lua_call(L, 1, 4)`.
    check("select('#', pairs(setmetatable({}, {__pairs = function() return 1 end}))) == 4");
    check(
        "select('#', pairs(setmetatable({}, {__pairs = function() return 1, 2, 3, 4, 5, 6 end}))) == 4",
    );
    check(
        "select(4, pairs(setmetatable({}, {__pairs = function() return 1, 2, 3, 4, 5, 6 end}))) == 4",
    );
}

/// A deleted entry stays in the bucket for `next`, but its key must not be
/// traced: a cleared table would otherwise pin every former key.
#[test]
fn dead_keys_are_not_retained() {
    const BUILD: &str = "T = {} for i = 1, 2000 do T[{i, i + 1}] = i end";
    let baseline = live_bytes_after("T = {}");
    let live = live_bytes_after(BUILD);
    let cleared = live_bytes_after(&format!("{BUILD} for k in pairs(T) do T[k] = nil end"));

    let payload = live.saturating_sub(baseline);
    assert!(
        payload > 50_000,
        "payload too small: baseline={baseline} live={live}"
    );
    let leaked = cleared.saturating_sub(baseline);
    assert!(
        leaked < payload / 10,
        "dead keys retained: baseline={baseline} live={live} cleared={cleared}"
    );
}
