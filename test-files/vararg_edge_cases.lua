-- Edge cases that previously broke or are subtle.

-- Zero varargs.
local function zero(...)
    local t = {...}
    return #t
end
print(zero())                     -- 0

-- More varargs than fixed params (with leading fixed).
local function leading(a, b, ...)
    local t = {...}
    return a + b + #t
end
print(leading(1, 2, 9, 9, 9))     -- 1 + 2 + 3 = 6

-- Vararg in middle (forces single result).
local function sum3(a, b, c) return a + b + c end
local function mid(...) return sum3(..., 99, 100) end
print(mid(7, 999, 999))           -- 7 (first vararg only) + 99 + 100 = 206

-- Forwarding chain: anonymous to anonymous.
local function forward(...) return ... end
local function chain(...) return forward(...) end
local function consume(a, b, c) return a + b + c end
print(consume(chain(11, 22, 33))) -- 66

-- Named vararg forwarding to anonymous, accessed via .n.
local function tagged(...args)
    return args[1] + args.n
end
print(tagged(50, 1, 2, 3, 4))     -- 50 + 5 = 55

-- Multi-return spread across multiple levels (#37 regression).
local function quad() return 1, 2, 3, 4 end
local function sum4(a, b, c, d) return a + b + c + d end
print(sum4(quad()))               -- 10

-- Multi-return as last arg of nested call.
local function f(n) return n end
print(f(f(f(7))))                 -- 7

-- Vararg in TAILCALL position.
local function tail(...) return forward(...) end
print(consume(tail(100, 200, 300))) -- 600
