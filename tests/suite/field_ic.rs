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
