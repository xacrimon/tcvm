local function collatz(n)
    local steps = 0
    while n ~= 1 do
        if n % 2 == 0 then n = n // 2 else n = 3 * n + 1 end
        steps = steps + 1
    end
    return steps
end
local total = 0
for i = 1, 1500000 do total = total + collatz(i) end
print(total)
