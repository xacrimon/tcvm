-- live: 0,4,8
-- lj: no
local ult = math.ult
local r
--@decl
for i = 1, N do
  --@live
  r = ult(i, 3)
end
--@use
if r == nil then print(r) end
