-- live: 0,8
local objs = {}
local mt = {__index = function(t, key) return 1 end}
for k = 1, 1024 do objs[k] = setmetatable({x = k}, mt) end
local r
--@decl
for _ = 1, math.floor(N / 1024) do
  for k = 1, 1024 do
    local o = objs[k]
    --@live
    r = o.y
  end
end
--@use
if r == 0.5 then print(r) end
