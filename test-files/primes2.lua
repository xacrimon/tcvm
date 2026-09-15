local function is_prime(n)
    if n < 2 then return false end
    if n % 2 == 0 then return n == 2 end
    local i = 3
    while i * i <= n do        -- trial division up to sqrt(n)
        if n % i == 0 then return false end
        i = i + 2
    end
    return true
end
local count = 0
for n = 1, 3000000 do if is_prime(n) then count = count + 1 end end
print(count)
