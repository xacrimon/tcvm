-- Named vararg, materialized path: `args` is used as a value (passed
-- to a function), forcing VARARGPREP to build a real vararg table.

local function unwrap_first(t)
    return t[1]
end

local function passthrough(...args)
    -- `args` here escapes — used as a value (passed to unwrap_first).
    -- This flips `used_as_non_base`, so `needs_vararg_table` is set and
    -- the VARARGPREP handler materializes the table at run time.
    return unwrap_first(args)
end

print(passthrough(42, 99, 7))   -- 42

-- Mixed: some accesses are index-style (args[1]) AND args is used as a
-- value. Materialization wins, and the index sites are rewritten from
-- VARARGGET to GETTABLE at the epilogue so they read the same table.
local function both(...args)
    local first = args[1]
    return unwrap_first(args) + first
end

print(both(5, 6, 7))            -- 5 + 5 = 10

-- Captured by a nested closure: also forces materialization.
local function build_getter(...args)
    return function() return args[1] + args[2] end
end

local g = build_getter(100, 200, 300)
print(g())                       -- 300
