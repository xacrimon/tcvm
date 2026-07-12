local function sum_field(t, n)
  local s = 0
  for i = 1, n do
    s = s + t.x
  end
  return s
end

-- Run it, so the GETFIELD inline cache is filled with the shape of `t` before
-- the JIT ever looks at this function.
local t = { x = 7 }
sum_field(t, 3)

return sum_field
