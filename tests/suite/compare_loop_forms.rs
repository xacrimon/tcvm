//! The compare and loop forms: every compare form's guard failure
//! (mixed numbers, NaN, boxed ints, strings, metamethods), a flipping site,
//! numeric loops of every kind through one site, and the `next`/`ipairs`
//! steps of the generic for with holes, deletions, `__index` and `__pairs`.
//! Expected strings come from `lua` 5.5.1 running the same chunk.

use crate::common::ok;

#[test]
fn compare_and_loop_forms_fall_back_correctly() {
    let src = r#"local out = {}
local function g(x) return math.type(x) == 'float' and ('%.10g'):format(x) or tostring(x) end
local nan = 0/0
local big = 1 << 40
-- register compares: one site each, fed ints first, then everything
local mt = {__lt = function(a, b) return 'LT' == 'LT' end, __le = function() return false end, __eq = function() return true end}
local v, w = setmetatable({}, mt), setmetatable({}, mt)
local pairs_ = {{1, 2}, {2, 1}, {3, 3}, {1.5, 2}, {2, 1.5}, {1.5, 2.5}, {nan, 1}, {1, nan}, {big, 1}, {1, big}, {big, big}, {'a', 'b'}, {'b', 'a'}, {v, w}, {v, 1}, {1, 2}, {2147483647, -2147483648}}
local function run(f)
  local r = {}
  for _, p in ipairs(pairs_) do local ok, x = pcall(f, p[1], p[2]) r[#r + 1] = ok and tostring(x) or x:match('attempt to [^(]*[^ (]') end
  return table.concat(r, ',')
end
out[#out + 1] = 'lt ' .. run(function(a, b) return a < b end)
out[#out + 1] = 'le ' .. run(function(a, b) return a <= b end)
out[#out + 1] = 'nlt ' .. run(function(a, b) if a < b then return 'y' else return 'n' end end)
out[#out + 1] = 'nle ' .. run(function(a, b) if a <= b then return 'y' else return 'n' end end)
out[#out + 1] = 'eq ' .. run(function(a, b) return a == b end)
out[#out + 1] = 'neq ' .. run(function(a, b) if a ~= b then return 'y' else return 'n' end end)
-- float-first register sites: every guard failure of the _FF forms
local pairs_f = {{1.5, 2.5}, {2.5, 1.5}, {1.5, 1.5}, {nan, 1.5}, {1.5, nan}, {1, 2}, {1.5, 2}, {big, 1.5}, {'a', 'b'}, {v, w}, {v, 1.5}, {1.5, 2.5}}
local function runf(f)
  local r = {}
  for _, p in ipairs(pairs_f) do local ok, x = pcall(f, p[1], p[2]) r[#r + 1] = ok and tostring(x) or x:match('attempt to [^(]*[^ (]') end
  return table.concat(r, ',')
end
out[#out + 1] = 'ltf ' .. runf(function(a, b) return a < b end)
out[#out + 1] = 'lef ' .. runf(function(a, b) return a <= b end)
out[#out + 1] = 'nltf ' .. runf(function(a, b) if a < b then return 'y' else return 'n' end end)
out[#out + 1] = 'nlef ' .. runf(function(a, b) if a <= b then return 'y' else return 'n' end end)
out[#out + 1] = 'eqf ' .. runf(function(a, b) return a == b end)
out[#out + 1] = 'neqf ' .. runf(function(a, b) if a ~= b then return 'y' else return 'n' end end)
-- immediate compares: float-first sites then ints, boxed, strings, tables
local one = {{1.5}, {2.5}, {1}, {2}, {big}, {-big}, {nan}, {'x'}, {v}, {3}, {3.0}}
local function one_(f) local r = {} for _, p in ipairs(one) do local ok, x = pcall(f, p[1]) r[#r + 1] = ok and tostring(x) or x:match('attempt to [^(]*[^ (]') end return table.concat(r, ',') end
out[#out + 1] = 'lti ' .. one_(function(a) return a < 2 end)
out[#out + 1] = 'lei ' .. one_(function(a) return a <= 2 end)
out[#out + 1] = 'gti ' .. one_(function(a) return 2 < a end)
out[#out + 1] = 'gei ' .. one_(function(a) return 2 <= a end)
out[#out + 1] = 'nlti ' .. one_(function(a) if a < 2 then return 'y' else return 'n' end end)
out[#out + 1] = 'ngei ' .. one_(function(a) if 2 <= a then return 'y' else return 'n' end end)
out[#out + 1] = 'eqi ' .. one_(function(a) return a == 2 end)
out[#out + 1] = 'ltin ' .. one_(function(a) return a < -3 end)
-- a flipping compare site
local function lt(a, b) return a < b end
local acc = {}
for i = 1, 12 do local a = (i % 2 == 0) and i or i + 0.5 local b = (i % 3 == 0) and 7 or 6.5 acc[#acc + 1] = tostring(lt(a, b)) end
out[#out + 1] = 'flip ' .. table.concat(acc, ',')
-- numeric for: int, float, boxed, negative, zero-trip, huge
local function loop(a, b, c) local n, last = 0, nil for i = a, b, c do n = n + 1 last = i end return n .. ':' .. g(last) end
local loops = {{1, 10, 1}, {10, 1, -1}, {1, 10, 3}, {1.0, 2.0, 0.5}, {1, 2.5, 0.5}, {0, 1, 0.3}, {big, big + 3, 1}, {2147483646, 2147483648, 1}, {-2147483648, -2147483650, -1}, {math.maxinteger - 1, math.maxinteger, 1}, {1, 0, 1}, {1, 2^62, 2^60}, {5, 1, -2}, {1, 1, 1}}
for _, l in ipairs(loops) do out[#out + 1] = 'for ' .. loop(l[1], l[2], l[3]) end
out[#out + 1] = 'forerr ' .. tostring(select(2, pcall(loop, 1, 2, 0)):match("'for' step is zero"))
-- the same loop site run with ints then floats then boxed
local function loop2(a, b, c) local s = 0 for i = a, b, c do s = s + i end return g(s) end
out[#out + 1] = 'for2 ' .. loop2(1, 10, 1) .. ' ' .. loop2(1.0, 2.0, 0.25) .. ' ' .. loop2(big, big + 2, 1) .. ' ' .. loop2(1, 10, 1)
-- generic for: pairs over array, hash, dict (>64 keys), with assignment to existing keys, nil holes, ipairs with holes and __index
local function keys(t) local r = {} for k, v in pairs(t) do t[k] = v r[#r + 1] = tostring(k) end table.sort(r) return table.concat(r, ',') end
out[#out + 1] = 'pairs ' .. keys({1, 2, 3, x = 1, y = 2, [10] = 1, [1.5] = 1, [true] = 1})
local d = {} for i = 1, 70 do d['k' .. i] = i end d[1] = 1 d[2] = 2
out[#out + 1] = 'pairsd ' .. #keys(d)
local function ip(t) local r = {} for i, v in ipairs(t) do r[#r + 1] = i .. '=' .. tostring(v) end return table.concat(r, ',') end
out[#out + 1] = 'ipairs ' .. ip({1, 2, nil, 4}) .. ' | ' .. ip(setmetatable({1, 2}, {__index = function(t, i) if i < 5 then return i * 10 end end})) .. ' | [' .. ip({}) .. '] [' .. ip({n = 1}) .. ']'
local function custom(t) local r = {} for k, v in next, t do r[#r + 1] = tostring(k) end table.sort(r) return table.concat(r, ',') end
out[#out + 1] = 'next ' .. custom({5, 6, z = 1})
local function iter(t) local i = 0 return function() i = i + 1 if t[i] then return i, t[i] end end end
local r = {} for i, v in iter({7, 8, 9}) do r[#r + 1] = i * v end out[#out + 1] = 'iterfn ' .. table.concat(r, ',')
-- pairs loop that deletes the current key, and one on a table with __pairs
local t = {a = 1, b = 2, c = 3} local cnt = 0 for k in pairs(t) do t[k] = nil cnt = cnt + 1 end out[#out + 1] = 'del ' .. cnt .. ' ' .. tostring(next(t))
local pt = setmetatable({}, {__pairs = function(t) return function(_, k) if not k then return 1, 'one' end end, t, nil end})
local r2 = {} for k, v in pairs(pt) do r2[#r2 + 1] = k .. v end out[#out + 1] = '__pairs ' .. table.concat(r2, ',')
return table.concat(out, '\n')
"#;
    assert_eq!(ok(src), "lt true,false,false,true,false,true,false,false,false,true,false,true,false,true,true,true,false
le true,false,true,true,false,true,false,false,false,true,true,true,false,false,false,true,false
nlt y,n,n,y,n,y,n,n,n,y,n,y,n,y,y,y,n
nle y,n,y,y,n,y,n,n,n,y,y,y,n,n,n,y,n
eq false,false,true,false,false,false,false,false,false,false,true,false,false,true,false,false,false
neq y,y,n,y,y,y,y,y,y,y,n,y,y,n,y,y,y
ltf true,false,false,false,false,true,true,false,true,true,true,true
lef true,false,true,false,false,true,true,false,true,false,false,true
nltf y,n,n,n,n,y,y,n,y,y,y,y
nlef y,n,y,n,n,y,y,n,y,n,n,y
eqf false,false,true,false,false,false,false,false,false,true,false,false
neqf y,y,n,y,y,y,y,y,y,n,y,y
lti true,false,true,false,false,true,false,attempt to compare string with number,true,false,false
lei true,false,true,true,false,true,false,attempt to compare string with number,false,false,false
gti false,true,false,false,true,false,false,attempt to compare number with string,true,true,true
gei false,true,false,true,true,false,false,attempt to compare number with string,false,true,true
nlti y,n,y,n,n,y,n,attempt to compare string with number,y,n,n
ngei n,y,n,y,y,n,n,attempt to compare number with string,n,y,y
eqi false,false,false,true,false,false,false,false,false,false,false
ltin false,false,false,false,false,true,false,attempt to compare string with number,true,false,false
flip true,true,true,true,true,true,false,false,false,false,false,false
for 10:10
for 10:1
for 4:10
for 3:2
for 4:2.5
for 4:0.9
for 4:1099511627779
for 3:2147483648
for 3:-2147483650
for 2:9223372036854775807
for 0:nil
for 5:4.611686018e+18
for 3:1
for 1:1
forerr 'for' step is zero
for2 55 7.5 3298534883331 55
pairs 1,1.5,10,2,3,true,x,y
pairsd 274
ipairs 1=1,2=2 | 1=1,2=2,3=30,4=40 | [] []
next 1,2,z
iterfn 7,16,27
del 3 nil
__pairs 1one");
}
