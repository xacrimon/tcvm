-- n: 1000000
local k = 1
local cnt = 0
local r
for _ = 1, N do
  cnt = cnt + 1
  r = function() return k + cnt end
end
if r() ~= N + 1 then print("wrong") end
