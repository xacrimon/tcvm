-- live: 0,4,8
-- env: TCVM_JIT_INLINE=0
local function run(f)
  local r
  --@decl
  for i = 1, N do
    --@live
    r = f(i)
  end
  --@use
  return r
end
if run(function(x) return x end) == nil then print("wrong") end
