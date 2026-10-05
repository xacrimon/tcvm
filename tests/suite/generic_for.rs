//! Execution coverage for the generic `for ... in ... do` loop: the control
//! register layout (TFORCALL/TFORLOOP) and multi-value iterator adjustment.

use crate::common::{err, eval, ok};

fn run(src: &str) -> i64 {
    eval(src)
}

#[test]
fn explicit_three_value_iterator() {
    // Explicit `f, s, control` form — exercises the TFORCALL/TFORLOOP layout.
    assert_eq!(
        run(
            "local function iter(_, c) if c < 3 then return c + 1 end end\n\
             local sum = 0\n\
             for x in iter, nil, 0 do sum = sum + x end\n\
             return sum"
        ),
        6
    );
}

#[test]
fn multi_value_iterator_call() {
    // A single call returning (iter, state, control) must spread into the
    // three control slots (the multires adjustment).
    assert_eq!(
        run(
            "local function iter(_, c) if c < 3 then return c + 1 end end\n\
             local function mk() return iter, nil, 0 end\n\
             local sum = 0\n\
             for x in mk() do sum = sum + x end\n\
             return sum"
        ),
        6
    );
}

#[test]
fn two_loop_variables() {
    // Iterator returning two values per step (key-like, value-like).
    assert_eq!(
        run("local function iter(_, c)\n\
             \x20 if c < 3 then return c + 1, (c + 1) * 10 end\n\
             end\n\
             local sum = 0\n\
             for k, v in iter, nil, 0 do sum = sum + k + v end\n\
             return sum"),
        // k: 1+2+3=6, v: 10+20+30=60
        66
    );
}

#[test]
fn empty_iteration() {
    // Iterator returns nil immediately: body never runs.
    assert_eq!(
        run("local function iter() return nil end\n\
             local n = 0\n\
             for x in iter, nil, 0 do n = n + 1 end\n\
             return n"),
        0
    );
}

#[test]
fn pairs_style_over_table() {
    // Hand-rolled stateful iterator over an array-like table, returned as a
    // single multi-value call — the realistic `for k,v in pairs(t)` shape.
    assert_eq!(
        run("local t = { 10, 20, 30, 40 }\n\
             local function inext(tbl, i)\n\
             \x20 i = i + 1\n\
             \x20 local v = tbl[i]\n\
             \x20 if v ~= nil then return i, v end\n\
             end\n\
             local function each(tbl) return inext, tbl, 0 end\n\
             local sum = 0\n\
             for i, v in each(t) do sum = sum + i + v end\n\
             return sum"),
        // i: 1+2+3+4=10, v: 10+20+30+40=100
        110
    );
}

#[test]
fn iterator_call_line() {
    // Calling the iterator is on the line the explist starts, past comments
    // and into parentheses, not the `for` line (#253). Lines from lua 5.5.1,
    // which also appends `(for iterator 'for iterator')` (#204).
    assert_eq!(
        err("\n\n for k,v in \n 3 \n do \n end"),
        "c:4: attempt to call a number value"
    );
    assert_eq!(
        err("for k in\n-- c\n--[[\n]]\n 3, 4 do end"),
        "c:5: attempt to call a number value"
    );
    assert_eq!(
        err("for k in\n(\n 3) do end"),
        "c:2: attempt to call a number value"
    );
}

/// `pairs` walks without calling `next` (see `op_tforcall`), in exactly
/// `next`'s order, through every part: array (with key 0 and holes), integer
/// and other hash keys, named slots inline and spilled, dict-mode strings.
#[test]
fn pairs_walks_every_part_in_next_order() {
    let src = r#"
local function same_order(t)
  local a, b = {}, {}
  for k, v in pairs(t) do a[#a + 1] = tostring(k) .. '=' .. tostring(v) end
  local k, v = next(t)
  while k ~= nil do b[#b + 1] = tostring(k) .. '=' .. tostring(v) k, v = next(t, k) end
  return table.concat(a, ',') == table.concat(b, ',') and #a
end
local named = load('local t = {} ' .. (function()
  local s = {} for i = 1, 40 do s[i] = 't.f' .. i .. ' = ' .. i end return table.concat(s, ' ')
end)() .. ' return t')()
local dict = {} for i = 1, 100 do dict['k' .. i] = i end
local mixed = {1, 2, nil, 4, a = 1, b = 2, [10] = 10, [-3] = 3, [1.5] = 1, [true] = 2}
mixed[{}] = 3 mixed[0] = 0
for i = 1, 80 do mixed['m' .. i] = i end
local out = {}
for _, t in ipairs({{}, {1, nil, 3, [0] = 0}, {[10] = 1, [1000] = 2, [-5] = 3},
                    {[1.5] = 1, [true] = 2, [false] = 3}, {a = 1, b = 2, c = 3}, named, dict, mixed}) do
  out[#out + 1] = tostring(same_order(t))
end
return table.concat(out, ' ')
"#;
    assert_eq!(ok(src), "0 3 3 3 3 40 100 91");
}

/// Clearing or updating existing fields mid-traversal, which Lua allows,
/// visits every key once.
#[test]
fn pairs_survives_clears_and_updates() {
    let src = r#"
local function tables()
  local named = {} for i = 1, 40 do named['f' .. i] = i end
  local dict = {} for i = 1, 100 do dict['k' .. i] = i end
  return {{1, 2, 3, 4}, {[10] = 1, [1000] = 2, [-5] = 3}, {a = 1, b = 2, c = 3}, named, dict,
          {1, 2, a = 1, [2.5] = 1, [true] = 1, [100] = 1}}
end
local out = {}
for _, t in ipairs(tables()) do
  local n = 0
  for k in pairs(t) do t[k] = nil n = n + 1 end
  out[#out + 1] = n .. '/' .. tostring(next(t))
end
for _, t in ipairs(tables()) do
  local n, sum = 0, 0
  for k, v in pairs(t) do t[k] = v * 2 n = n + 1 end
  for _, v in pairs(t) do sum = sum + v end
  out[#out + 1] = n .. ':' .. sum
end
return table.concat(out, ' ')
"#;
    assert_eq!(
        ok(src),
        "4/nil 3/nil 3/nil 40/nil 100/nil 6/nil 4:20 3:12 3:12 40:1640 100:10100 6:14"
    );
}

/// The forms around the walk: a loop started from a key, `next` under
/// another name, `__pairs`, a closing value, extra and single variables,
/// nested walks of one table, and a walk that yields.
#[test]
fn next_loop_forms() {
    let src = r#"
local out = {}
local t = {10, 20, 30}
local s = {}
for k, v in next, t, 1 do s[#s + 1] = k .. '=' .. v end
out[#out + 1] = table.concat(s, ',')
local n = next
s = {}
for k, v in n, {5, 6} do s[#s + 1] = k .. '=' .. v end
out[#out + 1] = table.concat(s, ',')
local p = setmetatable({}, {__pairs = function(t) return next, {7, 8}, nil end})
s = {}
for k, v in pairs(p) do s[#s + 1] = k .. '=' .. v end
out[#out + 1] = table.concat(s, ',')
local closed = 0
local c = setmetatable({}, {__close = function() closed = closed + 1 end})
for k in next, {1, 2, 3}, nil, c do end
for k in next, {1, 2, 3}, nil, c do break end
out[#out + 1] = closed
for k, v, x in pairs({1}) do out[#out + 1] = cat(k, v, x) end
for k in pairs({a = 1}) do out[#out + 1] = k end
s = {}
local u = {a = 1, b = 2, c = 3}
for k1 in pairs(u) do for k2 in pairs(u) do s[#s + 1] = k1 .. k2 end end
table.sort(s)
out[#out + 1] = table.concat(s, ',')
local co = coroutine.wrap(function()
  for k, v in pairs({a = 1, b = 2, c = 3, 4, 5}) do coroutine.yield(tostring(k) .. v) end
end)
s = {}
for _ = 1, 5 do s[#s + 1] = co() end
table.sort(s)
out[#out + 1] = table.concat(s, ',')
return table.concat(out, ' | ')
"#;
    assert_eq!(
        ok(src),
        "2=20,3=30 | 1=5,2=6 | 1=7,2=8 | 2 | 1 1 nil | a | aa,ab,ac,ba,bb,bc,ca,cb,cc | 14,25,a1,b2,c3"
    );
}

/// `ipairs` without a call: through the array part, the integer hash keys,
/// `__index`, and a hole made mid-loop.
#[test]
fn ipairs_walks() {
    let src = r#"
local out = {}
local function walk(t)
  local s = {}
  for i, v in ipairs(t) do s[#s + 1] = i .. '=' .. tostring(v) end
  return table.concat(s, ',')
end
out[#out + 1] = walk({1, 2, nil, 4})
out[#out + 1] = walk({[1] = 'a', [2] = 'b', [3] = 'c'})
local h = {} h[3] = 3 h[2] = 2 h[1] = 1
out[#out + 1] = walk(h)
out[#out + 1] = walk(setmetatable({1, 2}, {__index = function(_, i) if i <= 4 then return i * 10 end end}))
out[#out + 1] = walk(setmetatable({}, {__index = {'x', 'y'}}))
local t = {1, 2, 3, 4, 5}
local s = {}
for i, v in ipairs(t) do s[#s + 1] = v if i == 2 then t[4] = nil end end
out[#out + 1] = table.concat(s, ',')
s = {}
for i, v, x in ipairs({9}) do s[#s + 1] = cat(i, v, x) end
out[#out + 1] = table.concat(s, ',')
return table.concat(out, ' | ')
"#;
    assert_eq!(
        ok(src),
        "1=1,2=2 | 1=a,2=b,3=c | 1=1,2=2,3=3 | 1=1,2=2,3=30,4=40 | 1=x,2=y | 1,2,3 | 1 9 nil"
    );
}

/// Adding keys mid-traversal is undefined in Lua; the walk must still end
/// without reading out of bounds.
#[test]
fn pairs_over_a_growing_table_stays_in_bounds() {
    let src = r#"
local t = {} for i = 1, 10 do t['k' .. i] = i end
local n = 0
pcall(function()
  for k in pairs(t) do n = n + 1 if n < 200 then t['x' .. n] = n t[n] = n end end
end)
return tostring(n > 0)
"#;
    assert_eq!(ok(src), "true");
}

/// `pairs` and `ipairs` called the way a loop does go through their fast
/// entries; every other call shape, and a `__pairs` added to or removed from a
/// metatable in use, must give the builtins' results.
#[test]
fn pairs_and_ipairs_entries() {
    let src = r#"
local out = {}
local t = {1, 2}
local a, b, c, d, e = pairs(t)
out[#out + 1] = cat(a == next, b == t, c, d, e)
local f = pairs(t)
out[#out + 1] = cat(f == next, select('#', pairs(t)))
local function tail(x) return pairs(x) end
local g, h = tail(t)
out[#out + 1] = cat(g == next, h == t, select('#', tail(t)))
local mt = {}
local o = setmetatable({x = 1}, mt)
local s = {}
for k, v in pairs(o) do s[#s + 1] = k .. v end
mt.__pairs = function(self) return function(_, k) if not k then return 'p', 1 end end, self, nil end
for k, v in pairs(o) do s[#s + 1] = k .. v end
mt.__pairs = nil
for k, v in pairs(o) do s[#s + 1] = k .. v end
out[#out + 1] = table.concat(s, ',')
local i1, i2, i3, i4 = ipairs(t)
local it = ipairs(t)
out[#out + 1] = cat(i1 == it, i2 == t, i3, i4, select('#', ipairs(t)))
local n1, n2, n3 = ipairs(nil)
out[#out + 1] = cat(n2, n3)
out[#out + 1] = select(2, pcall(function() local x = pairs() end))
out[#out + 1] = select(2, pcall(function() local x = ipairs() end))
out[#out + 1] = cat(select('#', pairs(t, 1)), (pairs('abc')) == next)
return table.concat(out, ' | ')
"#;
    assert_eq!(
        ok(src),
        "true true nil nil nil | true 4 | true true 4 | x1,p1,x1 | true true 0 nil 3 | nil 0 | \
         c:25: bad argument #1 to 'pairs' (value expected) | \
         c:26: bad argument #1 to 'ipairs' (value expected) | 4 true"
    );
}
