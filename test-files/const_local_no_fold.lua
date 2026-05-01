-- `<const>` whose initializer doesn't fold to a const expdesc:
-- string literals (no Str variant), table constructors, length, etc.
-- These bind a const local but reference sites still load from the
-- register. Assignment is rejected (covered separately).

local s <const> = "hello"        -- string: not folded by current scope
local t <const> = {}             -- table: never foldable
local h <const> = #"abc"         -- length op never folds

a = s
b = t
c = h
