local objs = {}
for k = 1, 1024 do objs[k] = {k} end
local r
--@decl
for _ = 1, math.floor(N / 1024) do
  for k = 1, 1024 do
    local o = objs[k]
    --@live
    r = o[2]
  end
end
--@use
if r == 0.5 then print(r) end
