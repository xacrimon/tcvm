global print
local function f() return 10, 20 end
global a, b = f()
print(a, b)

local function g(...)
  global x, y, z = ...
  print(x, y, z)
end
g(10, 20, 30)

global p, q, r = 1
print(p, q, r)

-- Parentheses force exactly one value: the trailing (call)/(...) must NOT
-- expand across the remaining targets, so the extras stay nil.
local function h(...)
  global s, t = (f())
  print(s, t)
  global u, v, w = (...)
  print(u, v, w)
end
h(7, 8, 9)
