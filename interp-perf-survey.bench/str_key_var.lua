local t = {alpha = 1, beta = 2, gamma = 3, delta = 4, eps = 5}
local k1, k2 = "gamma", "eps"
local acc = 0
for i = 1, 30000000 do acc = acc + t[k1] + t[k2] end
print(acc)
