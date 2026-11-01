-- n: 1000000
-- env: TCVM_JIT_ONLY=@FILE
local yield, resume = coroutine.yield, coroutine.resume
local body = load("local yield = ... return function() while true do yield(1) end end", "=interp")(yield)
local co = coroutine.create(body)
load("local resume, co, N = ... for _ = 1, N do resume(co) end", "=interp")(resume, co, N)
