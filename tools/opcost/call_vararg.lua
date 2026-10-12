-- live: 0,4,8
-- env: TCVM_JIT_INLINE=0
local function f(...) local x = ... return x end
local r
--@decl
for i = 1, N do
  --@live
  r = f(i)
end
--@use
if r == nil then print(r) end
