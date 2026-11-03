-- live: 0,4,8
local insert = table.insert
--@decl
for _ = 1, math.floor(N / 1024) do
  local t = {}
  for i = 1, 1024 do
    --@live
    insert(t, i)
  end
end
--@use
