-- Exercises the SELF opcode emitted by `obj:m(...)`. Method calls only
-- parse correctly in expression position today (rhs of local/return/
-- assignment, or as function arguments) — the statement-position parser
-- doesn't recognize them yet.

local t = {}
function t.id(self) return self end
function t.add(self, a, b) return a + b end
function t.chain(self) return self end

-- Method call as the initializer of a local.
local a = t:id()

-- Method call with multiple positional args.
local b = t:add(1, 2)

-- Method call passed as a function argument.
local function use(x) return x end
local c = use(t:id())

-- Chained method calls — the receiver is itself a method-call result.
local d = t:chain():id()

-- Tail-call form: `return obj:m()` should emit TAILCALL, not CALL.
local function tail(o) return o:id() end
tail(t)

return b
