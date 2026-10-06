//! Binary arithmetic sites the slow path quickens (`ADD_NUM`, `ARITH_MM`,
//! `ARITH_MM_R`, `ARITH_MMI`), and back. Expected strings come from `lua`
//! 5.5.1 running the same chunk.

use crate::common::ok;

/// One `+` site fed tables, a table and a number, numbers, mixed numbers, then
/// tables again: it quickens, falls back for good, and still computes each.
#[test]
fn sites_change_operand_kinds() {
    let src = r#"
local V = {}
local function new(x) return setmetatable({x = x}, V) end
local function x(v) return type(v) == 'table' and v.x or v end
V.__add = function(a, b) return new(x(a) + x(b)) end
V.__sub = function(a, b) return new(x(a) - x(b)) end
local function add(a, b) return a + b end
local out = {}
for _, p in ipairs({{new(1), new(2)}, {new(3), 4}, {5, 6}, {1.5, 2}, {new(7), new(8)}, {9, new(10)}}) do
  local r = add(p[1], p[2])
  out[#out + 1] = tostring(x(r))
end
return table.concat(out, ' ')
"#;
    assert_eq!(ok(src), "3 7 11 3.5 15 19");
}

/// Every binary metamethod through register-immediate sites, with the constant
/// on either side, twice so the second pass runs the quickened forms.
#[test]
fn immediate_forms_both_sides() {
    let src = r#"
local V = {}
local function new(x) return setmetatable({x = x}, V) end
local log = {}
for _, k in ipairs({'add', 'sub', 'mul', 'div', 'mod', 'pow', 'idiv', 'band', 'bor', 'bxor', 'shl', 'shr'}) do
  V['__' .. k] = function(a, b)
    log[#log + 1] = k .. '(' .. (type(a) == 'table' and 'v' or tostring(a)) .. ',' .. (type(b) == 'table' and 'v' or tostring(b)) .. ')'
    return k
  end
end
local v = new(1)
for _ = 1, 2 do
  local _ = v + 1, 1 + v, v - 2, 10 - v, v * 0.5, 0.5 * v, v / 4, 4 / v, v % 3, 3 % v,
    v ^ 2, 2 ^ v, v // 2, 2 // v, v & 1, 1 & v, v | 2, 2 | v, v ~ 3, 3 ~ v, v << 1, 1 << v, v >> 1, 1 >> v
end
return table.concat(log, ' ', 1, 24) .. ' | ' .. tostring(#log)
"#;
    assert_eq!(
        ok(src),
        "add(v,1) add(1,v) sub(v,2) sub(10,v) mul(v,0.5) mul(0.5,v) div(v,4) div(4,v) mod(v,3) mod(3,v) pow(v,2) pow(2,v) idiv(v,2) idiv(2,v) band(v,1) band(1,v) bor(v,2) bor(2,v) bxor(v,3) bxor(3,v) shl(v,1) shl(1,v) shr(v,1) shr(1,v) | 48"
    );
}

/// Metamethods taken from the right operand, including after a number metatable
/// appears and goes away under sites quickened before it did.
#[test]
fn metamethod_from_the_right() {
    let src = r#"
local V = {__sub = function(a, b) return 'vsub' end}
local W = {__sub = function(a, b) return 'wsub' end}
local v, w = setmetatable({}, V), setmetatable({}, W)
local function sub(a, b) return a - b end
local function addk(a) return 1 + a end
local function subk(a) return 10 - a end
local out = {}
for i = 1, 2 do out[#out + 1] = sub(i, v) end
out[#out + 1] = sub({}, v)
out[#out + 1] = sub(w, v)
out[#out + 1] = sub(v, w)
out[#out + 1] = sub(3, v)
V.__add = function(a, b) return 'vadd' end
for _ = 1, 2 do out[#out + 1] = addk(v) end
debug.setmetatable(0, {__add = function(a, b) return 'numadd' end, __sub = function(a, b) return 'numsub' end})
out[#out + 1] = sub(1, v)
out[#out + 1] = addk(v)
out[#out + 1] = subk(v)
debug.setmetatable(0, nil)
out[#out + 1] = sub(1, v)
out[#out + 1] = addk(v)
out[#out + 1] = subk(v)
return table.concat(out, ' ')
"#;
    assert_eq!(
        ok(src),
        "vsub vsub vsub wsub vsub vsub vadd vadd numsub numadd numsub vsub vadd vsub"
    );
}

/// A quickened site whose metamethod is removed, replaced by a callable table or
/// a native, whose operand changes metatable, and one that yields.
#[test]
fn metatable_changes_under_a_site() {
    let src = r#"
local mt = {}
local v = setmetatable({}, mt)
local function add(a, b) return a + b end
local out = {}
mt.__add = function() return 'f' end
out[#out + 1] = add(v, 1)
out[#out + 1] = add(v, 1)
mt.__add = nil
out[#out + 1] = select(2, pcall(add, v, 1)):match('attempt to perform [^(]*[^ (]')
mt.__add = setmetatable({}, {__call = function(self, a, b) return 'callable' end})
out[#out + 1] = add(v, 1)
mt.__add = rawequal
out[#out + 1] = tostring(add(v, v))
setmetatable(v, {__add = function() return 'other' end})
out[#out + 1] = add(v, 1)
setmetatable(v, nil)
out[#out + 1] = select(2, pcall(add, v, 1)):match('attempt to perform [^(]*[^ (]')
local co = coroutine.wrap(function()
  local y = setmetatable({}, {__mul = function(a, b) return coroutine.yield('in') + b end})
  local r
  for i = 1, 2 do r = y * i end
  return r
end)
out[#out + 1] = co()
out[#out + 1] = co(10)
out[#out + 1] = co(20)
return table.concat(out, ' ')
"#;
    assert_eq!(
        ok(src),
        "f f attempt to perform arithmetic on a table value callable true other attempt to perform arithmetic on a table value in in 22"
    );
}

/// Mixed int/float sites quicken to their `_NUM` forms, which must still handle
/// every other operand pair: overflow, boxed ints, strings, zero divisors.
#[test]
fn mixed_numbers() {
    let src = r#"
local function g(x) return math.type(x) == 'float' and ('%.10g'):format(x) or tostring(x) end
local function f(a, b) return cat(g(a + b), g(a - b), g(a * b), g(a / b), g(a % b), g(a ^ b), g(a // b)) end
local out = {}
for _, p in ipairs({{3, 0.5}, {3, 2}, {2.5, 2.0}, {1 << 40, 3}, {7, -2}, {-7, 2.5}, {math.maxinteger, 1}, {'10', 3}, {5, math.huge}}) do
  out[#out + 1] = f(p[1], p[2])
end
out[#out + 1] = select(2, pcall(f, 3, 0)):match('attempt to perform [^(]*[^ (]')
out[#out + 1] = f(3.0, 0)
out[#out + 1] = select(2, pcall(f, 3, {})):match('attempt to perform [^(]*[^ (]')
return table.concat(out, ' | ')
"#;
    assert_eq!(
        ok(src),
        "3.5 2.5 1.5 6 0 1.732050808 6 | 5 1 6 1.5 1 9 1 | 4.5 0.5 5 1.25 0.5 6.25 1 | 1099511627779 1099511627773 3298534883328 3.665038759e+11 1 1.329227996e+36 366503875925 | 5 9 -14 -3.5 -1 0.02040816327 -4 | -4.5 -9.5 -17.5 -2.8 0.5 nan -3 | -9223372036854775808 9223372036854775806 9223372036854775807 9.223372037e+18 0 9.223372037e+18 9223372036854775807 | 13 7 30 3.333333333 1 1000 3 | inf -inf inf 0 5 inf 0 | attempt to perform 'n%0' | 3 3 0 inf nan 1 inf | attempt to perform arithmetic on a table value"
    );
}
