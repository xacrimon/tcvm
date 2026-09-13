-- Same function, compiled against `a`'s shape, then fed `b` — same field, but
-- reached by a different transition, so a different shape. Every `b` call enters
-- native code, fails the shape guard on the first iteration, and finishes in the
-- interpreter. The total is what says the deopt landed on a frame that could
-- actually be resumed.
local function sum_field(t, n)
  local s = 0
  for i = 1, n do
    s = s + t.x
  end
  return s
end

local a = { x = 7 }
local b = { y = 1, x = 7 }

local total = 0
for i = 1, 200 do
  total = total + sum_field(a, 10)
end
for i = 1, 50 do
  total = total + sum_field(b, 10)
end
return total, sum_field
