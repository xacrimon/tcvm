//! Constant-key accesses (`t.k`) that miss their inline cache, on shape-mode
//! and dict-mode tables. Expected strings come from `lua` 5.5.1 running the
//! same chunk.

use crate::common::ok;

/// `__newindex` fires for absent and nil-valued keys only, whichever mode the
/// receiver is in.
#[test]
fn newindex_on_misses() {
    let src = r#"
local out, log = {}, {}
local function ni(t, k, v) log[#log + 1] = k .. '=' .. tostring(v); rawset(t, k, v) end
for _, n in ipairs({0, 70}) do
  local t = {}
  for i = 1, n do t['k' .. i] = i end
  setmetatable(t, {__newindex = ni})
  for i = 1, 3 do t.a = i; t.b = nil end
  t.a = nil
  t.a = 9
  out[#out + 1] = cat(n, t.a, rawget(t, 'b'), table.concat(log, ','))
  log = {}
end
return table.concat(out, ' | ')
"#;
    assert_eq!(
        ok(src),
        "0 9 nil a=1,b=nil,b=nil,b=nil,a=9 | 70 9 nil a=1,b=nil,b=nil,b=nil,a=9"
    );
}

/// Adding and clearing `__index` with field stores on a live metatable.
#[test]
fn metamethod_stores_on_misses() {
    let src = r#"
local out = {}
local mt = {}
local o = setmetatable({}, mt)
for _ = 1, 2 do
  mt.__index = function(_, k) return 'f' .. k end
  out[#out + 1] = o.x
  mt.__index = nil
  out[#out + 1] = tostring(o.x)
end
local mtd = {}
for i = 1, 70 do mtd['k' .. i] = i end
local od = setmetatable({}, mtd)
mtd.__index = {x = 'dict'}
out[#out + 1] = od.x
mtd.__index = nil
out[#out + 1] = tostring(od.x)
return table.concat(out, ' ')
"#;
    assert_eq!(ok(src), "fx nil fx nil dict nil");
}

/// A cached absent key stops reading nil once `__index` appears in place,
/// the metatable is replaced, the key is added, or the table goes dict.
#[test]
fn absent_keys() {
    let src = r#"
local out = {}
local function z(t) return t.z end
local mt = {}
local t = setmetatable({x = 1}, mt)
out[#out + 1] = cat(z(t), z(t))
mt.__index = function() return 'mm' end
out[#out + 1] = z(t)
mt.__index = nil
out[#out + 1] = cat(z(t), z(t))
setmetatable(t, {__index = {z = 'proto'}})
out[#out + 1] = z(t)
local u = {x = 1}
out[#out + 1] = cat(z(u), z(u))
u.z = 'own'
out[#out + 1] = z(u)
local d = {x = 1}
out[#out + 1] = tostring(z(d))
for i = 1, 70 do d['k' .. i] = i end
d.z = 'dict'
out[#out + 1] = z(d)
local function g() return undefined_global end
out[#out + 1] = cat(g(), g())
undefined_global = 'defined'
out[#out + 1] = g()
return table.concat(out, ' ')
"#;
    assert_eq!(
        ok(src),
        "nil nil mm nil nil proto nil nil own nil dict nil nil defined"
    );
}

/// Stores that add a key, repeated from one site so later ones take the
/// cached transition.
#[test]
fn transitions() {
    let src = r#"
local out = {}
local function add(t, v) t.a = v; return t end
-- same site, same start shape: second and later stores hit the transition
local x, y = add({}, 1), add({}, 2)
out[#out + 1] = cat(x.a, y.a, next(x))
-- a nil store through a cached transition adds nothing
local z = add({}, nil)
out[#out + 1] = cat(z.a, next(z))
-- __newindex appearing in place on a shared metatable
local log = {}
local mt = {}
add(setmetatable({}, mt), 1)
mt.__newindex = function(_, k, v) log[#log + 1] = k .. v end
local w = add(setmetatable({}, mt), 2)
out[#out + 1] = cat(rawget(w, 'a'), table.concat(log))
-- keys added through a transition to tables that are metatables
local function idx(m) m.__index = function() return 'i' end end
local mt1, mt2 = {}, {}
local o1, o2 = setmetatable({}, mt1), setmetatable({}, mt2)
idx(mt1); idx(mt2)
out[#out + 1] = cat(o1.q, o2.q)
-- the site that crosses the 64-key cap, run on several tables
local function fill(t) for i = 1, 66 do t['k' .. i] = i end; t.last = 'end'; return t end
local f1, f2 = fill({}), fill({})
out[#out + 1] = cat(f1.k64, f1.k65, f2.k66, f2.last)
-- a constructor-like site over two different start shapes
local function mk(t) t.p = 1; t.q = 2; return t end
local m1, m2, m3 = mk({}), mk({r = 0}), mk({})
out[#out + 1] = cat(m1.p + m1.q, m2.p + m2.q + m2.r, m3.q)
local n = 0
for k in pairs(m3) do n = n + 1 end
out[#out + 1] = n
-- globals defined through SETTABUP transitions
local function def(v) newglobal_a = v; newglobal_b = v end
def(1); newglobal_a, newglobal_b = nil, nil; def(2)
out[#out + 1] = cat(newglobal_a, newglobal_b)
return table.concat(out, ' | ')
"#;
    assert_eq!(
        ok(src),
        "1 2 a 1 | nil nil | nil a2 | i i | 64 65 66 end | 3 3 2 | 2 | 2 2"
    );
}

/// Loads and method calls through an `__index` table stay correct as the
/// class, its `__index` and the receiver change under a cached entry.
#[test]
fn proto_loads() {
    let src = r#"
local out = {}
local function get(o) return o.m end
local function call(o) return o:m() end
local C = {}
C.__index = C
function C.m() return 'c1' end
local o = setmetatable({x = 1}, C)
out[#out + 1] = cat(call(o), call(o), get(o)(), get(o)())
C.m = function() return 'c2' end                       -- replaced in place
out[#out + 1] = cat(call(o), get(o)())
C.extra = 1                                             -- class shape changes
out[#out + 1] = call(o)
local B = {m = function() return 'base' end}
setmetatable(C, {__index = B})
C.m = nil                                               -- slot kept, nil: walk goes on
out[#out + 1] = cat(call(o), get(o)())
C.m = function() return 'c3' end
out[#out + 1] = cat(call(o), call(o))
C.__index = {m = function() return 'other' end}         -- __index reassigned
out[#out + 1] = cat(call(o), call(o))
C.__index = function(_, k) return function() return 'fn:' .. k end end
out[#out + 1] = cat(call(o), call(o))
local D = {m = function() return 'd' end}
C.__index = D
out[#out + 1] = call(o)
C.__index = D                                           -- same value again
out[#out + 1] = call(o)
o.m = function() return 'own' end                       -- receiver shadows
out[#out + 1] = cat(call(o), get(o)())
local p = setmetatable({x = 1}, C)
out[#out + 1] = call(p)
setmetatable(p, {__index = {m = function() return 'new mt' end}})
out[#out + 1] = call(p)
-- two levels: Derived -> Base
local Base = {}; Base.__index = Base
function Base.m() return 'b' end
local Derived = setmetatable({}, Base); Derived.__index = Derived
local q = setmetatable({}, Derived)
out[#out + 1] = cat(call(q), call(q))
function Derived.m() return 'derived' end
out[#out + 1] = cat(call(q), call(q))
-- a holder in dict mode
local Big = {}; Big.__index = Big
for i = 1, 70 do Big['f' .. i] = i end
Big.m = function() return 'big' end
local r = setmetatable({}, Big)
out[#out + 1] = cat(call(r), call(r))
-- a string __index, and string receivers
local s = setmetatable({}, {__index = 'abc'})
out[#out + 1] = cat(s.len == string.len, ('xy'):upper(), ('xy'):upper())
-- _ENV with __index
local function env_get()
  local _ENV = setmetatable({}, {__index = _G})
  return type(print), type(print)
end
out[#out + 1] = cat(env_get())
-- missing methods and __index loops
local e = setmetatable({}, {__index = {}})
out[#out + 1] = cat((pcall(call, e)), (pcall(call, e)))
local L = {}; L.__index = L; setmetatable(L, L)
out[#out + 1] = tostring(select(2, pcall(get, setmetatable({}, L))):find('chain too long') ~= nil)
return table.concat(out, ' | ')
"#;
    assert_eq!(
        ok(src),
        "c1 c1 c1 c1 | c2 c2 | c2 | base base | c3 c3 | other other | fn:m fn:m | d | d | own own | d | new mt | b b | derived derived | big big | true XY XY | function function | false false | true"
    );
}
