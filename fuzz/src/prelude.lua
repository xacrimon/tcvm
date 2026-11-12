local function emit(s) io.write(s, "\n") end
local NAMES
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
-- A metamethod's view of a value: bounded, so no string grows by feeding one its own result.
local function sk(v) return ser(v):sub(1, 24) end
local function sers(...)
  local n = select("#", ...)
  local p = { "#" .. n }
  for i = 1, n do p[#p + 1] = ser((select(i, ...))) end
  return table.concat(p, ",")
end
local function errn(e)
  if type(e) ~= "string" then return "E<" .. ser(e) .. ">" end
  return "E<" .. e:gsub("^[^:]*:(%d+): ", "L%1: "):gsub("0x%x%x%x%x%x%x+", "ADDR") .. ">"
end
-- What a call computed on the way (`probe`), reported with its results.
local PR, PN = "", 0
local function probe(v)
  PN = PN + 1
  if PN <= 12 then PR = PR .. "|" .. ser(v) end
end
local function probes()
  local s = PN > 12 and PR .. "|+" .. PN or PR
  PR, PN = "", 0
  return s
end
local function rec(tag, ok, ...)
  if ok then emit(tag .. " " .. sers(...) .. probes()) else emit(tag .. " " .. errn((...)) .. probes()) end
end
local function hist(tag, h)
  local ks = {}
  for k in pairs(h) do ks[#ks + 1] = k end
  table.sort(ks)
  local p = {}
  for _, k in ipairs(ks) do p[#p + 1] = k .. "=" .. h[k] end
  emit(tag .. " H " .. table.concat(p, " "))
end
-- An integer of a value that depends on nothing but the value: sums of these do not depend on
-- the order `pairs` takes.
local function hk(v)
  local s, h = ser(v), 7
  for i = 1, #s do h = (h * 31 + s:byte(i)) & 0xffffffff end
  return h
end
-- A value equal to `v` that is another object: a computed copy, or with `k` > 0 the number of
-- the other subtype where it is exact (`k` = 2: only where a float's text is the same in tcvm).
local function twin(v, k)
  local t = math.type(v)
  if t == "integer" then
    local f = v + 0.0
    if k > 0 and f == v and (k == 1 or -1e14 < v and v < 1e14) then return f end
    return v + 0
  elseif t == "float" then
    local i = math.tointeger(v)
    if k > 0 and i then return i end
    return v * 1.0
  elseif type(v) == "string" then
    return (v .. "#"):sub(1, -2)
  end
  return v
end
-- Loops that are not bounded by their text spend fuel; a call gets a fixed amount.
local FUEL = 0
local function tick()
  FUEL = FUEL - 1
  if FUEL < 0 then error("fuel", 0) end
end
local V, MT, MTI, CL
local function mk(name, t, mt)
  if mt then setmetatable(t, mt) end
  NAMES[t] = name
  return t
end
-- Every unit starts from the same values, whatever the units before it did to theirs.
local function reset()
  NAMES = setmetatable({}, { __mode = "k" })
  PR, PN, FUEL = "", 0, 0
  G1, G2, G3 = nil, nil, nil
  MT = {}
  MT.__index = function(t, k) return "idx:" .. sk(k) end
  MT.__newindex = function(t, k, v) rawset(t, "nw", sk(k) .. sk(v)) end
  MT.__call = function(self, a, b) return "call", a, b end
  for _, m in ipairs({ "add", "sub", "mul", "div", "mod", "pow", "unm", "idiv", "band", "bor", "bxor", "shl", "shr", "bnot", "concat" }) do
    MT["__" .. m] = function(x, y) return m .. ":" .. sk(x) .. sk(y) end
  end
  -- Comparisons answer with a bool, so they report their operands: order and int or float.
  MT.__eq = function(x, y) probe("eq" .. sk(x) .. sk(y)) return rawlen(x) <= rawlen(y) end
  MT.__lt = function(x, y) probe("lt" .. sk(x) .. sk(y)) return sk(x) < sk(y) end
  MT.__le = function(x, y) probe("le" .. sk(x) .. sk(y)) return sk(x) <= sk(y) end
  MT.__len = function(x, y) probe("len" .. sk(x) .. sk(y)) return 42 end
  MTI = { __index = { x = 11, y = "yy", [1] = "one", z = 2.5 } }
  CL = setmetatable({}, { __close = function(o, e) probe(e == nil and "closed" or "closed-err") end })
  NAMES[CL] = "CL"
  V = {}
  V.si = { 0, 1, -1, 2, 3, 7, -7, 100, 5, 16 }
  V.bi = { 2147483647, 2147483648, -2147483648, -2147483649, 1 << 40, math.maxinteger, math.mininteger, 4294967296, -(1 << 53), 1 << 62 }
  V.fl = { 0.5, -2.5, 3.75, 1.0, -1.0, 2.0, 100.25, 1e15, 0.1, 7.0 }
  V.fs = { 0.0, -0.0, 2.0 ^ 53, 2.0 ^ 63, -(2.0 ^ 63), 1 / 0, -1 / 0, 0 / 0, 1e308, 2.0 ^ 31 }
  -- `fs` without the floats that `tostring` writes with more than 14 digits.
  V.fc = { 0.0, -0.0, 1 / 0, -1 / 0, 0 / 0, 1e308, 2.0 ^ 31, 4294967296.0 }
  V.ns = { "10", "0x10", " 3 ", "1e2", "-7", "2.5", "9223372036854775808", "0x7fffffffffffffff" }
  V.st = { "", "a", "abc", "hello world", "x\0y", "10a", "B" }
  V.ot = { true, false }
  V.tb = {
    mk("plain", {}), mk("arr", { 10, 20, 30, n = 3 }), mk("rec", { x = 1, y = "two", z = 3.5 }),
    mk("mt", { 1, 2, x = 5 }, MT), mk("mti", {}, MTI), mk("hole", { 1, nil, 3 }), mk("mix", { 5, 6, x = 7, [2.5] = "f", [-1] = "neg" }),
  }
  V.fn = { mk("f1", function(...) return select("#", ...) end), mk("f2", function(a) return a end), math.abs }
  NAMES[math.abs] = "abs"
  V.co = { coroutine.create(function() end) }
  V.nil1 = { nil }
  V.sp = { 1, 2, 3, 5, 7 }
  V.sn = { -1, -2, -3 }
  V.mx = { 1, 2.5, "10", true, 3, "abc", 7.0, false }
  V.ms = { 1, "2", 3, "4.5", 5, "0x10" }
  V.fp = { 0.5, 1.5, 2.25, 3.0 }
  V.mn = { 1, 2.5, -3, 4.0, 7, -0.5 }
  V.ib = { 1, 2147483647, -2147483648, 3, 1 << 40 }
  V.tt = { V.tb[2], V.tb[3], V.tb[7] }
  V.tm = { V.tb[4], V.tb[5] }
  -- Tables without holes, for code that takes lengths.
  V.sq = { mk("sq0", {}), mk("sq3", { 4, 5, 6 }), mk("sqs", { "p", "q" }), mk("sq1", { 9 }) }
  for k, l in pairs(V) do l.n = (k == "nil1") and 1 or #l end
end
-- Where a driver loop stands, so it resumes after an error instead of restarting.
local pos = 0
