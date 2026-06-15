-- `global a, b = <multi>` expands a trailing call/vararg across targets
-- (#113); a parenthesized trailing value adjusts to one and is nil-padded.
global print

local function f()
  return 10, 20
end

local function w(...)
  -- Bare call expands: a=10, b=20 (CALL ret=3 + 2x SETTABUP).
  global a, b = f()
  -- Parenthesized call adjusts to one: c=10, d=nil (CALL ret=2 + LOADNIL).
  global c, d = (f())
  -- Bare vararg expands (VARARG count=3).
  global e, g = ...
  -- Parenthesized vararg adjusts to one (VARARG count=2 + LOADNIL).
  global h, i = (...)
  -- Bare method call expands: j=10, k=20 (SELF + CALL ret=3 + 2x SETTABUP).
  local t = {}
  t.m = function(self)
    return 10, 20
  end
  global j, k = t:m()
  -- Parenthesized method call adjusts to one: l=10, m=nil.
  global l, n = (t:m())
  print(a, b, c, d, e, g, h, i, j, k, l, n)
  -- Method-call expander where the receiver is an UPVALUE inside a nested
  -- function whose value-block base is R0: SELF lands the func above the
  -- receiver temp, so the expander MOVEs the results down to R0/R1.
  -- The `global` decl is lexically scoped to `up_recv`; print there too.
  -- (Ordered after the outer print on purpose — declaring new `o`/`p`
  -- globals transitions `_ENV`'s shape.)
  local function up_recv()
    global o, p = t:m()
    print("up_recv", o, p)
  end
  up_recv()
end

w(100, 200)
