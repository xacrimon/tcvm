-- live: 0,4,8
-- env: TCVM_JIT_INLINE=0
local pcall = pcall
local function f(x) return x end
local ok, r
--@decl
for i = 1, N do
  --@live
  ok, r = pcall(f, i)
end
--@use
if r == nil then print(r) end
