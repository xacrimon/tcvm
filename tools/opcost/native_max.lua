-- live: 0,4,8
local max = math.max
local r
--@decl
for i = 1, N do
  --@live
  r = max(i, 3)
end
--@use
if r == nil then print(r) end
