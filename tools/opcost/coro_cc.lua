-- n: 1000000
local yield, resume = coroutine.yield, coroutine.resume
local co = coroutine.create(function() while true do yield(1) end end)
local ok, r
for _ = 1, N do ok, r = resume(co) end
if not ok then print(r) end
