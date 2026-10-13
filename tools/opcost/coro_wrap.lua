-- n: 1000000
local yield = coroutine.yield
local co = coroutine.wrap(function() while true do yield(1) end end)
local r
for _ = 1, N do r = co() end
if r ~= 1 then print(r) end
