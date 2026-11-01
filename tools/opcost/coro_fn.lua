-- n: 1000000
local yield, resume = coroutine.yield, coroutine.resume
local co = coroutine.create(function() while true do yield(1) end end)
local function step() local _, v = resume(co) return v end
local s = 0
for _ = 1, N do s = s + step() end
if s ~= N then print(s) end
