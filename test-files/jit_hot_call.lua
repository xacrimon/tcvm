-- Calls `sum_field` far past the compile threshold, so most of these calls run
-- as native code. The answer is the point: 200 * (10 * 7).
local function sum_field(t, n)
  local s = 0
  for i = 1, n do
    s = s + t.x
  end
  return s
end

local t = { x = 7 }
local total = 0
for i = 1, 200 do
  total = total + sum_field(t, 10)
end
return total, sum_field
