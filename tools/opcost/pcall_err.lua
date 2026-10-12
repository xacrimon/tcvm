-- live: 0,4,8
-- n: 500000
local pcall, error = pcall, error
local e = {}
local function f() error(e) end
local ok
--@decl
for _ = 1, N do
  --@live
  ok = pcall(f)
end
--@use
if ok then print(ok) end
