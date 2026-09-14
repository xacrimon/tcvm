print "hello"
print [[long]]
local function f(t) return #t end
print(f {1, 2, 3})
local s = setmetatable({}, {__index = {m = function(self, x) return x end}})
print(s:m "str", s:m {4, 5}, f {} + 1)
print(type "x", f {1})
local g = function(a) return function(b) return a .. b end end
print(g "a" "b")
