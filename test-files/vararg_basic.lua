-- Anonymous `...` covering each multires consumer site.

-- Pick first via local-assignment.
local function first(...)
    local a = ...
    return a
end
print(first(7, 8, 9))             -- 7

-- Pick second via local-assignment that requests two values.
local function second(...)
    local a, b = ...
    return b
end
print(second(7, 8, 9))            -- 8

-- Length of `{...}` — exercises SETLIST count=0.
local function len(...)
    local t = {...}
    return #t
end
print(len("x", "y", "z", "w"))    -- 4

-- `...` as last arg of an outer call.
local function sum3(a, b, c) return a + b + c end
local function fwd(...) return sum3(...) end
print(fwd(11, 22, 33))            -- 66

-- `...` in last-arg position mixed with leading fixed args.
local function fwd_mixed(...) return sum3(1, ...) end
print(fwd_mixed(20, 30))          -- 1 + 20 + 30 = 51

-- Multi-return call as last arg (fixes #37).
local function pair() return 100, 200 end
print(sum3(1, pair()))            -- 301
