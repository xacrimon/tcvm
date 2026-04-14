local function is_prime(x)
    for i=2, x-1 do
        if x % i == 0 then
            return false
        end
    end

    return true
end

local x = 2
local found = 0
local s = ""
while found < 1000 do
    if is_prime(x) then
        found = found + 1
        s = s .. x .. "\n"
    end

    x = x + 1
end

print(s)
