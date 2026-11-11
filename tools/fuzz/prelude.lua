local OUT = {}
local function emit(s) OUT[#OUT + 1] = s end
local NAMES = setmetatable({}, { __mode = "k" })
local function ser(v)
  local t = type(v)
  if t == "number" then
    if v ~= v then return "nan" end
    if math.type(v) == "integer" then return "i" .. string.format("%d", v) end
    return "f" .. string.format("%a", v)
  elseif t == "string" then
    return "s'" .. v:gsub("%z", "\\0"):gsub("0x%x%x%x%x%x%x+", "ADDR") .. "'"
  elseif t == "nil" then return "nil"
  elseif t == "boolean" then return v and "T" or "F"
  elseif t == "table" then return "t:" .. (NAMES[v] or "new")
  elseif t == "function" then return "fn:" .. (NAMES[v] or "new")
  else return t end
end
local function sers(...)
  local n = select("#", ...)
  local p = { "#" .. n }
  for i = 1, n do p[#p + 1] = ser((select(i, ...))) end
  return table.concat(p, ",")
end
local function errn(e)
  if type(e) ~= "string" then return "E<" .. ser(e) .. ">" end
  return "E<" .. e:gsub("^[^:]*:%d+: ", "") .. ">"
end
local function rec(tag, ok, ...)
  if ok then emit(tag .. " " .. sers(...)) else emit(tag .. " " .. errn((...))) end
end
local function hist(tag, h)
  local ks = {}
  for k in pairs(h) do ks[#ks + 1] = k end
  table.sort(ks)
  local p = {}
  for _, k in ipairs(ks) do p[#p + 1] = k .. "=" .. h[k] end
  emit(tag .. " H " .. table.concat(p, " "))
end
-- value pool
local MT = {}
MT.__index = function(t, k) return "idx:" .. ser(k) end
MT.__newindex = function(t, k, v) rawset(t, "nw", ser(k) .. ser(v)) end
MT.__call = function(self, a, b) return "call", a, b end
for _, m in ipairs({ "add", "sub", "mul", "div", "mod", "pow", "unm", "idiv", "band", "bor", "bxor", "shl", "shr", "bnot", "concat" }) do
  MT["__" .. m] = function(x, y) return m .. ":" .. ser(x) .. ser(y) end
end
MT.__eq = function(x, y) return true end
MT.__lt = function(x, y) return ser(x) < ser(y) end
MT.__le = function(x, y) return ser(x) <= ser(y) end
MT.__len = function(x) return 42 end
local MTI = { __index = { x = 11, y = "yy", [1] = "one", z = 2.5 } }
local function mk(name, t, mt)
  if mt then setmetatable(t, mt) end
  NAMES[t] = name
  return t
end
local V = {}
V.si = { 0, 1, -1, 2, 3, 7, -7, 100, 5, 16 }
V.bi = { 2147483647, 2147483648, -2147483648, -2147483649, 1 << 40, math.maxinteger, math.mininteger, 4294967296, -(1 << 53), 1 << 62 }
V.fl = { 0.5, -2.5, 3.75, 1.0, -1.0, 2.0, 100.25, 1e15, 0.1, 7.0 }
V.fs = { 0.0, -0.0, 2.0 ^ 53, 2.0 ^ 63, -(2.0 ^ 63), 1 / 0, -1 / 0, 0 / 0, 1e308, 2.0 ^ 31 }
V.ns = { "10", "0x10", " 3 ", "1e2", "-7", "2.5", "9223372036854775808", "0x7fffffffffffffff" }
V.st = { "", "a", "abc", "hello world", "x\0y", "10a", "B" }
V.ot = { true, false }
V.tb = {
  mk("plain", {}), mk("arr", { 10, 20, 30, n = 3 }), mk("rec", { x = 1, y = "two", z = 3.5 }),
  mk("mt", { 1, 2, x = 5 }, MT), mk("mti", {}, MTI), mk("hole", { 1, nil, 3 }), mk("mix", { 5, 6, x = 7, [2.5] = "f", [-1] = "neg" }),
}
V.fn = { mk("f1", function(...) return select("#", ...) end), mk("f2", function(a) return a end), print == nil and 1 or math.abs }
NAMES[math.abs] = "abs"
V.co = { coroutine.create(function() end) }
local ALL = {}
for _, fam in ipairs({ "si", "bi", "fl", "fs", "ns", "st", "ot", "tb", "fn" }) do
  for _, v in ipairs(V[fam]) do ALL[#ALL + 1] = v end
end
V.nil1 = { nil }
local function pick(fam, i)
  local l = V[fam]
  if fam == "nil1" then return nil end
  return l[(i - 1) % #l + 1]
end
V.sp = { 1, 2, 3, 5, 7 }
V.sn = { -1, -2, -3 }
V.fp = { 0.5, 1.5, 2.25, 3.0 }
V.mn = { 1, 2.5, -3, 4.0, 7, -0.5 }
V.ib = { 1, 2147483647, -2147483648, 3, 1 << 40 }
V.tt = { V.tb[2], V.tb[3], V.tb[7] }
V.tm = { V.tb[4], V.tb[5] }
for k, l in pairs(V) do l.n = (k == "nil1") and 1 or #l end
-- Where a driver loop stands, so it resumes after an error instead of restarting.
local pos = 0
