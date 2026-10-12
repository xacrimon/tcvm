-- n: 1000000
-- lj: no
local t = {0}
local big = 1099511627776
for i = 1, N do t[1] = big + i end
if t[1] ~= big + N then print("wrong") end
