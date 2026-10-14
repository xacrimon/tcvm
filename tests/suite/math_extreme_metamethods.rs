//! Selection with Lua's `<`, including user comparisons and exact argument identity.
use crate::common::ok;

#[test]
fn table_order_and_identity() {
    assert_eq!(
        ok(r#"
local mt = {__lt = function(a, b) return a.n < b.n end}
local a, b, c = setmetatable({n=2}, mt), setmetatable({n=1}, mt), setmetatable({n=3}, mt)
local tie = setmetatable({n=1}, mt)
return cat(math.min(a,b,c) == b, math.max(a,b,c) == c, math.min(b,tie) == b)
"#),
        "true true true"
    );
}

#[test]
fn right_operand_method_and_mixed_types() {
    assert_eq!(
        ok(r#"
local t = setmetatable({}, {__lt = function(a, b) return type(a) == 'number' end})
return cat(math.min(t, 1), math.max(1, t) == t)
"#),
        "1 true"
    );
}

#[test]
fn comparison_result_truthiness() {
    assert_eq!(
        ok(r#"
local mt = {__lt = function() return 0, false end}
local a, b = setmetatable({}, mt), setmetatable({}, mt)
local x = math.min(a,b) == b
mt.__lt = function() return false, true end
local y = math.max(a,b) == a
mt.__lt = function() end
return cat(x, y, math.min(a,b) == a)
"#),
        "true true true"
    );
}

#[test]
fn metamethod_error_object() {
    assert_eq!(
        ok(r#"
local token = {}
local t = setmetatable({}, {__lt = function() error(token) end})
local a, b = pcall(math.min, t, t)
local c, d = pcall(math.max, t, t)
return cat(a, b == token, c, d == token)
"#),
        "false true false true"
    );
}

#[test]
fn collection_and_reentrant_selection() {
    assert_eq!(
        ok(r#"
local mt = {__lt = function(a, b)
  collectgarbage('collect')
  return math.min(a.n, b.n) == a.n and a.n ~= b.n
end}
local function t(n) return setmetatable({n=n}, mt) end
return cat(math.min(t(5), t(3), t(4), t(1)).n, math.max(t(5), t(3), t(9)).n)
"#),
        "1 9"
    );
}

#[test]
fn operand_order_and_live_metatable_changes() {
    assert_eq!(
        ok(r#"
local log = {}
local mt = {}
mt.__lt = function(a,b)
  log[#log+1] = a.n .. '<' .. b.n
  mt.__lt = function(c,d) log[#log+1] = c.n .. '!' .. d.n; return false end
  return a.n < b.n
end
local function t(n) return setmetatable({n=n}, mt) end
local v = math.min(t(3), t(2), t(1))
return cat(v.n, table.concat(log, ','))
"#),
        "2 2<3,1!2"
    );
}

#[test]
fn userdata_comparison() {
    assert_eq!(
        ok(r#"
local a, b = io.tmpfile(), io.tmpfile()
a:close(); b:close()
local mt = {__lt = function(x,y) return x == a and y == b end}
debug.setmetatable(a, mt); debug.setmetatable(b, mt)
return cat(math.min(b,a) == a, math.max(a,b) == b)
"#),
        "true true"
    );
}

#[test]
fn primitive_and_single_argument_controls() {
    assert_eq!(
        ok(r#"
local token = {}
local nan = 0/0
return cat(math.min('20','3'), math.max('20','3'), math.min(token) == token,
  math.max(false) == false, math.min(nil) == nil, math.min(nan, 1) ~= math.min(nan, 1),
  math.max(1, nan), math.type(math.min(1, 1.0)), math.type(math.max(1.0, 1)))
"#),
        "20 3 true true true true 1 integer float"
    );
}
