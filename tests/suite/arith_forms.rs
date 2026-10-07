//! The adaptive arithmetic forms: every form's guard failure (type
//! flips, boxed ints, zero divisors, metamethods on either side, string
//! coercion), a site that keeps flipping, edge values, and NaN payloads.
//! Expected strings come from `lua` 5.5.1 running the same chunks, with
//! the variable-name suffix of its error messages stripped.

use crate::common::ok;

#[test]
fn every_form_falls_back_correctly() {
    let src = r#"local function cat(...) local t = table.pack(...) for i = 1, t.n do t[i] = tostring(t[i]) end return table.concat(t, ' ') end
local function g(x) return math.type(x) == 'float' and ('%.10g'):format(x) or tostring(x) end
local out = {}
-- one site per op, fed int/int until specialized, then every other kind
local function run(f, pairs_)
  local r = {}
  for _, p in ipairs(pairs_) do
    local ok, v = pcall(f, p[1], p[2])
    r[#r + 1] = ok and g(v) or v:match('attempt to [^(]*[^ (]') or v:gsub("%(local '%a'%) ", ''):gsub('^.-:%d+: ', '')
  end
  return table.concat(r, ',')
end
local big = 1 << 40
local mt = {__add = function(a, b) return 'A' end, __sub = function() return 'S' end, __mul = function() return 'M' end,
  __div = function() return 'D' end, __mod = function() return 'Mo' end, __pow = function() return 'P' end,
  __idiv = function() return 'I' end, __band = function() return 'Ba' end, __shl = function() return 'Sh' end}
local v = setmetatable({}, mt)
local seq = {{1, 2}, {3, 4}, {2147483647, 1}, {big, 1}, {1, big}, {1.5, 2}, {2, 1.5}, {1.5, 2.5}, {'10', 3}, {3, '2'}, {v, 1}, {1, v}, {v, v}, {7, 0}, {7.0, 0}, {{}, 1}, {5, 6}}
out[#out + 1] = 'add ' .. run(function(a, b) return a + b end, seq)
out[#out + 1] = 'sub ' .. run(function(a, b) return a - b end, seq)
out[#out + 1] = 'mul ' .. run(function(a, b) return a * b end, seq)
out[#out + 1] = 'div ' .. run(function(a, b) return a / b end, seq)
out[#out + 1] = 'mod ' .. run(function(a, b) return a % b end, seq)
out[#out + 1] = 'pow ' .. run(function(a, b) return a ^ b end, seq)
out[#out + 1] = 'idiv ' .. run(function(a, b) return a // b end, seq)
out[#out + 1] = 'band ' .. run(function(a, b) return a & b end, seq)
out[#out + 1] = 'shl ' .. run(function(a, b) return a << b end, seq)
-- float-first sites
local fseq = {{1.5, 2.5}, {1.5, 2}, {2, 1.5}, {1, 2}, {big, 2.5}, {v, 2.5}}
out[#out + 1] = 'fadd ' .. run(function(a, b) return a + b end, fseq)
out[#out + 1] = 'fdiv ' .. run(function(a, b) return a / b end, fseq)
out[#out + 1] = 'fidiv ' .. run(function(a, b) return a // b end, fseq)
-- immediate sites: int, float, boxed, string, table, overflow
local iseq = {{1}, {2}, {2147483647}, {-2147483648}, {1.5}, {big}, {'4'}, {v}, {{}}, {3}}
local one = function(f) local r = {} for _, p in ipairs(iseq) do local ok, x = pcall(f, p[1]) r[#r + 1] = ok and g(x) or x:match('attempt to [^(]*[^ (]') end return table.concat(r, ',') end
out[#out + 1] = 'addi ' .. one(function(a) return a + 1 end)
out[#out + 1] = 'raddi ' .. one(function(a) return 1 + a end)
out[#out + 1] = 'subi ' .. one(function(a) return a - 1 end)
out[#out + 1] = 'rsubi ' .. one(function(a) return 10 - a end)
out[#out + 1] = 'muli ' .. one(function(a) return a * 2 end)
out[#out + 1] = 'mulif ' .. one(function(a) return a * 0.5 end)
out[#out + 1] = 'addif ' .. one(function(a) return a + 0.5 end)
out[#out + 1] = 'divi ' .. one(function(a) return a / 2 end)
out[#out + 1] = 'rdivi ' .. one(function(a) return 2 / a end)
out[#out + 1] = 'modi ' .. one(function(a) return a % 3 end)
out[#out + 1] = 'rmodi ' .. one(function(a) return 3 % a end)
out[#out + 1] = 'idivi ' .. one(function(a) return a // 2 end)
out[#out + 1] = 'ridivi ' .. one(function(a) return 2 // a end)
out[#out + 1] = 'powi ' .. one(function(a) return a ^ 2 end)
out[#out + 1] = 'rpowi ' .. one(function(a) return 2 ^ a end)
out[#out + 1] = 'bandi ' .. one(function(a) return a & 1 end)
out[#out + 1] = 'shli ' .. one(function(a) return a << 1 end)
out[#out + 1] = 'rshli ' .. one(function(a) return 1 << a end)
-- a site that flips kinds forever still computes
local function add(a, b) return a + b end
local acc = {}
for i = 1, 12 do
  local a = (i % 3 == 0) and i or (i % 3 == 1) and i + 0.5 or (1 << 40)
  local b = (i % 2 == 0) and 1 or 0.25
  acc[#acc + 1] = g(add(a, b))
end
out[#out + 1] = 'flip ' .. table.concat(acc, ',')
-- -0.0, inf, nan, MIN % -1, MIN // -1
out[#out + 1] = 'edge ' .. cat(g(-0.0 + 0), g(0.0 * -1), g(1/0 - 1/0), g((-2147483648) % -1), g((-2147483648) // -1), g(math.mininteger // -1), g(math.mininteger % -1), g(5 // 0.0), g(-5 % 0.0), g(2^63 // 1))
return table.concat(out, '\n')
"#;
    let expected = "add 3,7,2147483648,1099511627777,1099511627777,3.5,3.5,4,13,5,A,A,A,7,7,attempt to perform arithmetic on a table value,11
sub -1,-1,2147483646,1099511627775,-1099511627775,-0.5,0.5,-1,7,1,S,S,S,7,7,attempt to perform arithmetic on a table value,-1
mul 2,12,2147483647,1099511627776,1099511627776,3,3,3.75,30,6,M,M,M,0,0,attempt to perform arithmetic on a table value,30
div 0.5,0.75,2147483647,1.099511628e+12,9.094947018e-13,0.75,1.333333333,0.6,3.333333333,1.5,D,D,D,inf,inf,attempt to perform arithmetic on a table value,0.8333333333
mod 1,3,0,0,1,1.5,0.5,1.5,1,1,Mo,Mo,Mo,attempt to perform 'n%0',nan,attempt to perform arithmetic on a table value,5
pow 1,81,2147483647,1.099511628e+12,1,2.25,2.828427125,2.755675961,1000,9,P,P,P,1,1,attempt to perform arithmetic on a table value,15625
idiv 0,0,2147483647,1099511627776,0,0,1,0,3,1,I,I,I,attempt to divide by zero,inf,attempt to perform arithmetic on a table value,0
band 0,0,1,0,0,number has no integer representation,number has no integer representation,number has no integer representation,attempt to perform bitwise operation on a string value,attempt to perform bitwise operation on a string value,Ba,Ba,Ba,0,0,attempt to perform bitwise operation on a table value,4
shl 4,48,4294967294,2199023255552,0,number has no integer representation,number has no integer representation,number has no integer representation,attempt to perform bitwise operation on a string value,attempt to perform bitwise operation on a string value,Sh,Sh,Sh,7,7,attempt to perform bitwise operation on a table value,320
fadd 4,3.5,3.5,3,1.099511628e+12,A
fdiv 0.6,0.75,1.333333333,0.5,4.398046511e+11,D
fidiv 0,0,1,0,4.398046511e+11,I
addi 2,3,2147483648,-2147483647,2.5,1099511627777,5,A,attempt to perform arithmetic on a table value,4
raddi 2,3,2147483648,-2147483647,2.5,1099511627777,5,A,attempt to perform arithmetic on a table value,4
subi 0,1,2147483646,-2147483649,0.5,1099511627775,3,S,attempt to perform arithmetic on a table value,2
rsubi 9,8,-2147483637,2147483658,8.5,-1099511627766,6,S,attempt to perform arithmetic on a table value,7
muli 2,4,4294967294,-4294967296,3,2199023255552,8,M,attempt to perform arithmetic on a table value,6
mulif 0.5,1,1073741824,-1073741824,0.75,5.497558139e+11,2,M,attempt to perform arithmetic on a table value,1.5
addif 1.5,2.5,2147483648,-2147483648,2,1.099511628e+12,4.5,A,attempt to perform arithmetic on a table value,3.5
divi 0.5,1,1073741824,-1073741824,0.75,5.497558139e+11,2,D,attempt to perform arithmetic on a table value,1.5
rdivi 2,1,9.31322575e-10,-9.313225746e-10,1.333333333,1.818989404e-12,0.5,D,attempt to perform arithmetic on a table value,0.6666666667
modi 1,2,1,1,1.5,1,1,Mo,attempt to perform arithmetic on a table value,0
rmodi 0,1,3,-2147483645,0,3,3,Mo,attempt to perform arithmetic on a table value,0
idivi 0,1,1073741823,-1073741824,0,549755813888,2,I,attempt to perform arithmetic on a table value,1
ridivi 2,1,0,-1,1,0,0,I,attempt to perform arithmetic on a table value,0
powi 1,4,4.611686014e+18,4.611686018e+18,2.25,1.20892582e+24,16,P,attempt to perform arithmetic on a table value,9
rpowi 2,4,inf,0,2.828427125,inf,16,P,attempt to perform arithmetic on a table value,8
bandi 1,0,1,0,0,attempt to perform bitwise operation on a string value,Ba,attempt to perform bitwise operation on a table value,1
shli 2,4,4294967294,-4294967296,2199023255552,attempt to perform bitwise operation on a string value,Sh,attempt to perform bitwise operation on a table value,6
rshli 2,4,0,0,0,attempt to perform bitwise operation on a string value,Sh,attempt to perform bitwise operation on a table value,8
flip 1.75,1099511627777,3.25,5.5,1.099511628e+12,7,7.75,1099511627777,9.25,11.5,1.099511628e+12,13
edge 0 -0 nan 0 2147483648 -9223372036854775808 0 inf nan 9.223372037e+18";
    assert_eq!(ok(src), expected);
}

#[test]
fn nan_payloads_never_reach_box_space() {
    let src = r#"local function show(x) return tostring(x) .. ':' .. tostring(x ~= x) end
local t = {}
t[#t+1] = show(0/0)
t[#t+1] = show(math.huge - math.huge)
t[#t+1] = show(-(0/0))
t[#t+1] = show(math.abs(0/0))
t[#t+1] = show((0/0) % 1)
t[#t+1] = show(2 ^ (0/0))
t[#t+1] = show(string.unpack('<d', string.pack('<I8', 0xFFF9000000000001)))
t[#t+1] = show(string.unpack('<d', string.pack('<I8', 0xFFFC000000000123)))
t[#t+1] = show(string.unpack('<d', string.pack('<I8', 0x7FF0000000000001)))
local n = string.unpack('<d', string.pack('<I8', 0xFFFD000000000000))
t[#t+1] = show(n + 1) .. ' ' .. show(n * 2) .. ' ' .. show(-n) .. ' ' .. show(n // 1) .. ' ' .. show(1 / n)
local x = 0/0
t[#t+1] = tostring(type(x)) .. ' ' .. tostring(math.type(x)) .. ' ' .. tostring(x == x) .. ' ' .. tostring(rawequal(x, x))
local tbl = {} tbl[1] = 0/0 t[#t+1] = tostring(math.type(tbl[1]))
return table.concat(t, ' | ')
"#;
    assert_eq!(
        ok(src),
        "nan:true | nan:true | nan:true | nan:true | nan:true | nan:true | nan:true | nan:true | nan:true | nan:true nan:true nan:true nan:true nan:true | number float false false | float"
    );
}

/// A site that misses three times locks, and keeps computing every case.
#[test]
fn a_flipping_site_locks_and_still_computes() {
    let src = r#"
local function add(a, b) return a + b end
local out = {}
local kinds = {{1, 2}, {1.5, 2.5}, {1, 2.5}, {1.5, 2}, {1 << 40, 1}, {'3', 4}}
for round = 1, 3 do
  for _, p in ipairs(kinds) do out[#out + 1] = tostring(add(p[1], p[2])) end
end
return table.concat(out, ' ')
"#;
    let one = "3 4.0 3.5 3.5 1099511627777 7";
    assert_eq!(ok(src), format!("{one} {one} {one}"));
}
