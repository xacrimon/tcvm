//! Named slots split between a table's own cell and its spill cell, and
//! constructor arrays kept in the cell. Expected strings come from `lua`
//! 5.5.1 running the same chunk.

use crate::common::ok;

/// Fields past a constructor's inline room spill, through stores, deletes,
/// collections and `pairs`.
#[test]
fn spill_past_inline() {
    let src = r#"
local objs = {}
for i = 1, 3000 do
  local o = {a = {i}, b = i, c = 'c' .. i}
  o.d = {i * 2}
  o.e = i * 3
  o.f = {i * 4}
  objs[i] = o
  if i % 500 == 0 then collectgarbage() end
end
local s = 0
for i = 1, 3000, 13 do local o = objs[i]; s = s + o.a[1] + o.b + #o.c + o.d[1] + o.e + o.f[1] end
for i = 1, 3000, 2 do objs[i].e = nil; objs[i].a = nil end
collectgarbage()
local c = 0
for i = 1, 3000 do for _ in pairs(objs[i]) do c = c + 1 end end
return cat(s, c)
"#;
    assert_eq!(ok(src), "3802405 15000");
}

/// A constructor with more fields than any table keeps inline.
#[test]
fn template_past_largest_capacity() {
    let src = r#"
local src = {'return function(i) return {'}
for k = 1, 40 do src[#src + 1] = 'f' .. k .. ' = {i + ' .. k .. '},' end
src[#src + 1] = '} end'
local mk = load(table.concat(src))()
local bigs = {}
for i = 1, 500 do bigs[i] = mk(i); bigs[i].g = {i}; if i % 100 == 0 then collectgarbage() end end
local s = 0
for i = 1, 500, 7 do for k = 1, 40 do s = s + bigs[i]['f' .. k][1] end s = s + bigs[i].g[1] end
return cat(s)
"#;
    assert_eq!(ok(src), "795564");
}

/// One access site over tables with the same keys and different inline
/// capacities, and an `__index` holder whose method slots spill.
#[test]
fn sites_across_capacities() {
    let src = r#"
local function sum(o) return o.x + o.y + (o.z or 0) end
local a = {x = 1, y = 2}
local b = {} b.x = 3 b.y = 4
local c = {x = 5, y = 6, z = 7, w = 8, v = 9}
local s = 0
for _ = 1, 1000 do s = s + sum(a) + sum(b) + sum(c) end
local C = {}
for k = 1, 10 do C['m' .. k] = function(self) return k + self.v end end
C.__index = C
local inst = setmetatable({v = 1}, C)
local m = 0
for _ = 1, 1000 do m = m + inst:m1() + inst:m10() end
C.m10 = function() return 0 end
return cat(s, m, inst:m10())
"#;
    assert_eq!(ok(src), "28000 13000 0");
}

/// A constructor's array part starts in the table's cell and moves out when
/// it grows; a constructor table with inline fields goes dict.
#[test]
fn inline_array_and_dict() {
    let src = r#"
local m = {tag = 'div', 10, 20, 30, n = 3}
m[4] = 40 m[5] = 50
for i = 6, 100 do m[#m + 1] = i end
collectgarbage()
local k = 0
for _ in pairs(m) do k = k + 1 end
local t = {p = 1, q = 2}
for i = 1, 600 do t['k' .. i] = i end
t.p = nil
collectgarbage()
local c = 0
for _ in pairs(t) do c = c + 1 end
return cat(#m, m.tag, m.n, m[1], m[100], k, c, t.q, t.k600)
"#;
    assert_eq!(ok(src), "100 div 3 10 100 102 601 2 600");
}

/// Constructors sized by the fields later statements store into them: a
/// module, a `setmetatable({}, C)` object, a shadowed local, and a rebound
/// `setmetatable` that returns another table.
#[test]
fn sized_by_later_stores() {
    let src = r#"
local M = {}
function M.f() return 1 end
function M:g() return self.c end
M.c, M.d = 3, 4
local C = {}
C.__index = C
local function new(x)
  local self = setmetatable({}, C)
  self.x = x
  self.y = x * 2
  return self
end
local o = new(5)
local shadow = {a = 1}
local shadow = {}
shadow.b = 2
local setmetatable = function(t) return {z = 9} end
local r = setmetatable({}, C)
r.w = 1
local big = {}
for _, k in ipairs({'a','b','c','d','e','f','g','h','i','j'}) do big[k] = k end
big.k1 = 1 big.k2 = 2 big.k3 = 3
local n = 0 for _ in pairs(big) do n = n + 1 end
return cat(M.f(), M:g(), M.d, o.x, o.y, getmetatable(o) == C, shadow.a, shadow.b, r.z, r.w, n)
"#;
    assert_eq!(ok(src), "1 3 4 5 10 true nil 2 9 1 13");
}
