-- Both anonymous and named vararg parameter forms parse.
local function f(a, b, ...) return a, b end
local function g(a, ...args) return args end
local function h(...) return ... end
local function k(...rest) return rest end
return f(1, 2, 3), g(10, 20), h(true, false), k()
