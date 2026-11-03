-- live: 0,4,8
local select = select
local function run(...)
  local r
  --@decl
  for _ = 1, N do
    --@live
    r = select('#', ...)
  end
  --@use
  return r
end
if run(1, 2, 3) ~= 3 then print("wrong") end
