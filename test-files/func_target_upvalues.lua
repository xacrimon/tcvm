-- A function statement's target is captured before its body's upvalues, as
-- luac orders them: `up` before `z`, and `_ENV` before `w`.
local up, z, w
local function outer() function up() return z end end
local function outer_global() global function gg() return w end end
