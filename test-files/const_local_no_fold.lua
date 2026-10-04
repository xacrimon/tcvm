-- `<const>` whose initializer doesn't fold to a const expdesc: table
-- constructors, length, etc. These bind a const local but reference
-- sites still load from the register. Assignment is rejected (covered
-- separately).

local t <const> = {}             -- table: never foldable
local h <const> = #"abc"         -- length op never folds

b = t
c = h
