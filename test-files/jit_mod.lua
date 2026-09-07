-- `%` and `//` in native code, over every sign combination.
--
-- Lua rounds these toward negative infinity; `sdiv` truncates toward zero. The
-- two agree on positive operands and disagree on every other quadrant, so a
-- benchmark full of positive numbers would never notice a wrong correction term.
-- Hence the negatives, and hence checking against the reference implementation
-- rather than against a number I worked out by hand.
local function modops(a, b)
  return a % b, a // b
end

-- Well past the compile threshold, so most of these run natively.
local h = 0
for a = -30, 30 do
  for b = -9, 9 do
    if b ~= 0 then
      local m, d = modops(a, b)
      h = (h * 31 + m) % 1000003
      h = (h * 31 + d) % 1000003
    end
  end
end
print(h)

-- The one input where the hardware and the interpreter could each be excused for
-- disagreeing: `sdiv` wraps rather than trapping, and so does `wrapping_div`.
local min = -9223372036854775807 - 1
print(modops(min, -1))
print(modops(min, 3))
print(modops(7, 3))
print(modops(-7, 3))
print(modops(7, -3))
print(modops(-7, -3))

return h, modops
