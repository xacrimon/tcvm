local objs = {}
local Base = {y = 2}
local mt = {__index = Base}
for k = 1, 1024 do objs[k] = setmetatable({x = k}, mt) end
local r
--@decl
for _ = 1, math.floor(N / 1024) do
  for k = 1, 1024 do
    local o = objs[k]
    --@live
    r = o.w
  end
end
--@use
if r == 0.5 then print(r) end
