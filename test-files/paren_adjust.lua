-- Parenthesized multi-value RHS adjusts to one value (#119, #48); bare
-- trailing call/vararg still expands across targets.
local function f()
  return 10, 20
end

-- Bare call expands: a=10, b=20 (CALL ret=3).
local a, b = f()

-- Parenthesized call adjusts to one: c=10, d=nil (CALL ret=2 + LOADNIL).
local c, d = (f())

-- Bare method call expands: m=10, n=20 (SELF + CALL ret=3).
local tbl = {}
tbl.meth = function(self)
  return 10, 20
end
local m, n = tbl:meth()
-- Parenthesized method call adjusts to one: o=10, p=nil.
local o, p = (tbl:meth())

-- Method-call expander with the receiver as an UPVALUE inside a nested
-- function whose value-block base is R0: SELF lands the func above the
-- receiver temp, so the expander must MOVE the results down to R0/R1.
local function up_recv()
  local x, y = tbl:meth()
  return x, y
end

local function g(...)
  -- Bare vararg expands (VARARG count=3).
  local p, q = ...
  -- Parenthesized vararg adjusts to one (VARARG count=2 + LOADNIL).
  local r, s = (...)
  -- Nested parens still adjust to one.
  local t, u = ((...))
  return p, q, r, s, t, u
end

-- `return (f())` must NOT tail-call; adjust to one value (CALL + RETURN).
local function h()
  return (f())
end

-- Bare `return f()` still tail-calls.
local function k()
  return f()
end

return a, b, c, d, m, n, o, p, g, h, k, up_recv
