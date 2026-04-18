local a, b, c, d = 1, 2, 3, 4

if a < b and c < d then
    return 1
end

if a < b or c < d then
    return 2
end

local x = a and b
local y = a or b
local z = not (a < b)
return x, y, z
