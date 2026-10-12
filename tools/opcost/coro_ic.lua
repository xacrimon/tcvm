-- n: 1000000
-- env: TCVM_JIT_ONLY=@FILE
local yield, resume = coroutine.yield, coroutine.resume
local co = coroutine.create(function() while true do yield(1) end end)
load("local resume, co, N = ... for _ = 1, N do resume(co) end", "=interp")(resume, co, N)
