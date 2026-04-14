local function add(a, b)
    return a + b
end

local function sub(a, b)
    return a - b
end

local function mul(a, b)
    return a * b
end

local function div(a, b)
    return a / b
end

local a = mul(3, 4)
local b = sub(3, 1)
local c = div(6, b)
local d = add(a, c)
print(d)
