-- live: 0,4,8
-- env: TCVM_JIT_INLINE=0
local function f(x) return x, x end
local function g(a, b) return b end
local r
--@decl
for i = 1, N do
  --@live
  r = g(f(i))
end
--@use
if r == nil then print(r) end
