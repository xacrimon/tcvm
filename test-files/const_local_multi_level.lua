-- Three-level nesting: const declared at the top scope is inlinable in
-- the innermost function with no upvalue chain at any level.

local k <const> = 7

local function outer()
    local function inner()
        return k + 1
    end
    return inner
end

return outer
