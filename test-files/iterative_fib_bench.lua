for i=0,100000000 do
    local n = 35
    local a = 0
    local b = 1
    local count = 2

    while count < n + 1 do
        local c = b
        b = a + b
        a = c
        count = count + 1
    end
end
