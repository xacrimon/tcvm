-- live: 0,4,8
-- n: 500000
local format = string.format
local r
--@decl
for i = 1, N do
  --@live
  r = format("%d", i)
end
--@use
if r == nil then print(r) end
