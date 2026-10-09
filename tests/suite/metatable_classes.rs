//! Metatables with the same metamethods share a class, and a table whose
//! metatable didn't make its class keeps the metatable in a slot of its own.
//! Expected strings come from `lua` 5.5.1 running the same chunk.

use crate::common::ok;

/// Per-object metatables with one content: each table keeps its own
/// metatable, and the slot holding it shows in neither `pairs` nor `next`.
#[test]
fn shared_class_keeps_identity() {
    let src = r#"
local C = {get = function(self) return 'C' .. self.x end}
local objs, mts, same = {}, {}, true
for i = 1, 20 do
  mts[i] = {__index = C}
  objs[i] = setmetatable({x = i}, mts[i])
end
for i = 1, 20 do
  same = same and getmetatable(objs[i]) == mts[i] and objs[i]:get() == 'C' .. i
end
local ks = {}
for k, v in pairs(objs[5]) do ks[#ks + 1] = k .. '=' .. v end
return cat(same, getmetatable(objs[1]) ~= getmetatable(objs[2]), table.concat(ks, ','),
  next(objs[6], 'x'), rawget(objs[7], '(metatable)'))
"#;
    assert_eq!(ok(src), "true true x=5 nil nil");
}

/// A write to one member's `__index` reaches only that member's tables,
/// through the method-call caches, for the class's maker and the others.
#[test]
fn member_write_leaves_class() {
    let src = r#"
local C = {get = function(self) return 'C' .. self.x end}
local D = {get = function(self) return 'D' .. self.x end}
local objs, mts = {}, {}
for i = 1, 6 do
  mts[i] = {__index = C}
  objs[i] = setmetatable({x = i}, mts[i])
end
local out = {}
local function sweep()
  local s = {}
  for i = 1, 6 do s[#s + 1] = objs[i]:get() end
  out[#out + 1] = table.concat(s, ' ')
end
sweep()
mts[3].__index = D
sweep()
mts[1].__index = D
sweep()
mts[1].__index = C
mts[3].__index = C
for k = 1, 9 do mts[5].__index = (k % 2 == 0) and C or D end
sweep()
return table.concat(out, ' | ')
"#;
    assert_eq!(
        ok(src),
        "C1 C2 C3 C4 C5 C6 | C1 C2 D3 C4 C5 C6 | D1 C2 D3 C4 C5 C6 | C1 C2 C3 C4 D5 C6"
    );
}

/// The same for metamethods the arithmetic, length and `tostring` paths read.
#[test]
fn member_write_other_metamethods() {
    let src = r#"
local function add(a, b) return a.v + b.v end
local function len(a) return a.v * 10 end
local function tostr(a) return 'V' .. a.v end
local function newer() return 'new' end
local vs = {}
for i = 1, 5 do
  vs[i] = setmetatable({v = i}, {__add = add, __len = len, __tostring = tostr})
end
local before = cat(vs[1] + vs[2], #vs[3], tostring(vs[4]))
getmetatable(vs[2]).__add = newer
getmetatable(vs[3]).__len = nil
getmetatable(vs[4]).__tostring = newer
return cat(before, vs[1] + vs[2], vs[2] + vs[1], #vs[3], #vs[5], tostring(vs[4]), tostring(vs[5]))
"#;
    assert_eq!(ok(src), "3 30 V4 3 new 0 50 new V5");
}

/// A table that keeps its metatable itself, through removal, re-adding the
/// class's maker, and dict mode.
#[test]
fn own_slot_through_reset_and_dict() {
    let src = r#"
local C = {get = function(self) return 'C' .. self.x end}
local D = {get = function(self) return 'D' .. self.x end}
local a = setmetatable({x = 1}, {__index = C})
local p = setmetatable({x = 2}, {__index = C})
setmetatable(p, nil)
local r1 = cat(getmetatable(p), p.get)
setmetatable(p, getmetatable(a))
local r2 = cat(p:get(), getmetatable(p) == getmetatable(a))
for i = 1, 100 do p['k' .. i] = i end
local n = 0
for k in pairs(p) do n = n + 1 end
local r3 = cat(p:get(), n, getmetatable(p) == getmetatable(a))
setmetatable(p, {__index = D})
local r4 = p:get()
setmetatable(p, nil)
return cat(r1, '|', r2, '|', r3, '|', r4, getmetatable(p), p.get)
"#;
    assert_eq!(ok(src), "nil nil | C2 true | C2 101 true | D2 nil nil");
}

/// One member dropping its shared `__newindex` table.
#[test]
fn member_write_newindex() {
    let src = r#"
local store = {}
local ts = {}
for i = 1, 4 do ts[i] = setmetatable({}, {__newindex = store, __index = store}) end
for i = 1, 4 do ts[i]['k' .. i] = i end
getmetatable(ts[2]).__newindex = nil
ts[2].own = 1
ts[3].other = 2
local ks = {}
for k in pairs(store) do ks[#ks + 1] = k end
table.sort(ks)
return cat(table.concat(ks, ','), rawget(ts[2], 'own'), rawget(ts[3], 'other'), ts[1].k4)
"#;
    assert_eq!(ok(src), "k1,k2,k3,k4,other 1 nil 4");
}

/// A store cached at a site while its table was not a metatable yet stays
/// off the tables that are: their writes still reach the class.
#[test]
fn store_cached_before_adoption() {
    let src = r#"
local A = {get = function(s) return 'A' .. s.x end}
local B = {get = function(s) return 'B' .. s.x end}
local function set_index(t, v) t.__index = v end
local function add_len(t) t.__len = function() return 7 end end
local m1, m2, plain = {__index = A}, {__index = A}, {}
set_index(m1, A); set_index(m2, A); add_len(plain)
local o1, o2 = setmetatable({x = 1}, m1), setmetatable({x = 2}, m2)
set_index(m2, B)
local a = o1:get() .. o2:get()
set_index(m1, B)
add_len(m1)
return cat(a, o1:get(), o2:get(), #o1, #o2)
"#;
    assert_eq!(ok(src), "A1B2 B1 B2 7 0");
}

/// Metatables joining the class their shape last adopted, `__mode`
/// included; a member whose writes retired it keeps a class of its own.
#[test]
fn join_by_adopted_shape() {
    let src = r#"
local A = {get = function(s) return 'A' .. s.x end}
local B = {get = function(s) return 'B' .. s.x end}
local objs, mts = {}, {}
for i = 1, 6 do mts[i] = {__index = A, __mode = 'k'}; objs[i] = setmetatable({x = i}, mts[i]) end
for k = 1, 5 do mts[2].__index = (k % 2 == 0) and A or B end
local late = setmetatable({x = 7}, {__index = A, __mode = 'k'})
local s = {}
for i = 1, 6 do s[#s + 1] = objs[i]:get() end
objs[3] = setmetatable({x = 30}, mts[2])
return cat(table.concat(s, ' '), late:get(), objs[3]:get(), getmetatable(objs[3]) == mts[2], getmetatable(late).__mode)
"#;
    assert_eq!(ok(src), "A1 B2 A3 A4 A5 A6 A7 B30 true k");
}

/// An adopted metatable moving to dict mode keeps its class.
#[test]
fn adopted_metatable_goes_dict() {
    let src = r#"
local A = {get = function(s) return 'A' .. s.x end}
local B = {get = function(s) return 'B' .. s.x end}
local D = {__index = A}
local o = setmetatable({x = 1}, D)
for i = 1, 600 do D['k' .. i] = i end
local before = o:get()
D.__index = B
local p = setmetatable({x = 2}, D)
return cat(before, o:get(), p:get(), D.k600, getmetatable(p) == D)
"#;
    assert_eq!(ok(src), "A1 B1 B2 600 true");
}
