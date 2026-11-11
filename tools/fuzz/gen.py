#!/usr/bin/env python3
"""gen.py SEED [KIND]: print one fuzz program (see run.py).

A program is prelude.lua plus a few instances. An instance is one small function f, called many
times with values of one family (so it compiles speculating on them), then with values of other
families, then with the first family again. Every result goes through the prelude's serializer,
which keeps integers and floats apart. KIND restricts the instances to one template; `limits` is
integer loops with bounds and steps near the integer limits, the rest take (a, b, c).
"""
import os
import random
import sys

PRELUDE = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "prelude.lua")).read()

FAMS = ["si", "bi", "fl", "fs", "ns", "st", "ot", "tb", "fn", "nil1", "sp", "sn", "fp", "mn", "ib", "tt", "tm"]
NUMF = ["si", "sp", "fl", "fp", "mn", "ib", "bi"]
POOL = [("si", 10), ("bi", 10), ("fl", 10), ("fs", 10), ("ns", 8), ("st", 7), ("ot", 2), ("tb", 7), ("fn", 3), ("co", 1)]

KONST = ["0", "1", "-1", "2", "3", "8", "0.0", "-0.0", "1.0", "0.5", "-2.0", "2147483648", "2147483647",
         "-2147483648", "1099511627776", "9223372036854775807", "math.mininteger", "2^53", "1e300", "63", "64", "-64", "65", "0x7fffffff", '"10"', '"abc"', '"0x10"', "4.0", "-3"]
INTK = ["0", "1", "-1", "2", "3", "7", "-5", "2147483647", "2147483648", "-2147483648", "1099511627776",
        "9223372036854775807", "math.mininteger", "0x7fffffff"]
FLK = ["0.0", "-0.0", "1.0", "0.5", "-2.0", "2.5", "2^53", "1e300", "3.0", "-1.5"]

BINOPS = ["+", "-", "*", "/", "//", "%", "^", "&", "|", "~", "<<", ">>", ".."]
CMPOPS = ["==", "~=", "<", "<=", ">", ">="]

BUILTINS = [("math.floor", 1), ("math.ceil", 1), ("math.abs", 1), ("math.sqrt", 1), ("math.min", 3), ("math.max", 3),
            ("math.fmod", 2), ("math.sin", 1), ("math.cos", 1), ("math.tan", 1), ("math.exp", 1), ("math.log", 2),
            ("math.atan", 2), ("math.tointeger", 1), ("type", 1), ("tostring", 1), ("tonumber", 2), ("rawget", 2),
            ("rawset", 3), ("rawequal", 2), ("rawlen", 1), ("getmetatable", 1), ("select", 3),
            ("assert", 3), ("string.len", 1), ("string.sub", 3), ("string.byte", 3), ("table.insert", 3),
            ("table.remove", 2), ("next", 2), ("math.ult", 2), ("math.type", 1), ("string.rep", 2),
            ("string.char", 2), ("string.upper", 1), ("table.unpack", 3), ("ipairs", 1), ("pairs", 1)]

# The limits template: what its loop computes from the index i and the argument k.
LIMIT_EXPRS = ["i + k", "i - k", "i * k", "i // k", "i % k", "i << k", "i >> k", "-i", "i + 1", "i - 1", "i * 2",
               "i * i", "(i + k) - k", "i ~ k", "i & k", "i | k", "k - i", "i / k", "i ^ 2", "math.abs(i)", "i + 0.5",
               "i // 2", "i % 3", "math.min(i, k)", "math.max(i, k)", "math.tointeger(i)", "math.ult(i, k) and 1 or 0",
               "i + k + 1", "(i * k) // 1", "math.fmod(i, k)", "math.floor(i / 2)"]
LIMIT_TESTS = ["x > i", "x < k", "x == i", "x <= 0", "x ~= x", "i + 1 > i", "i - 1 < i", "x >= k",
               "math.type(x) == 'float'", "x == i + k", "i == hi", "i * 2 > i"]
LIMIT_ACCS = ["s = s ~ math.tointeger(x // 1 | 0)", "s = s + x", "s = x", "s = math.max(s, x)",
              "s = s + (x > 0 and 1 or 0)"]
LIMIT_BOUNDS = ["0", "1", "-1", "10", "-10", "math.maxinteger", "math.mininteger", "math.maxinteger - 5",
                "math.mininteger + 5", "2147483647", "2147483648", "-2147483648", "-2147483649", "2147483640",
                "1 << 62", "(1 << 53) - 3"]
LIMIT_STEPS = ["1", "-1", "2", "-2", "3", "1 << 62", "math.mininteger", "math.maxinteger", "2147483647"]
LIMIT_KS = ["1", "-1", "0", "2", "3", "63", "64", "-64", "1 << 40", "math.maxinteger", "math.mininteger",
            "2147483647", "0.5", "-2", "2^53", "7"]
LIMIT_WARM = ["1, 10, 1, 2", "1, 20, 1, 3", "-5, 5, 1, 1", "10, 1, -1, 2", "0, 30, 3, 7"]


class G:
    def __init__(self, seed):
        self.r = random.Random(seed)
        self.n = 0

    def c(self, l):
        return self.r.choice(l)

    def k(self):
        return self.c(KONST)

    def operand(self):
        return self.c(["a", "b", "c", self.k()])

    # random expression trees
    def expr(self, d):
        r = self.r
        if d <= 0 or r.random() < 0.25:
            return self.c(["a", "b", "c", "a", "b", self.k()])
        x = r.random()
        if x < 0.45:
            return "(%s %s %s)" % (self.expr(d - 1), self.c(BINOPS), self.expr(d - 1))
        if x < 0.55:
            return "(%s %s %s)" % (self.expr(d - 1), self.c(CMPOPS), self.expr(d - 1))
        if x < 0.62:
            return "(%s %s %s)" % (self.expr(d - 1), self.c(["and", "or"]), self.expr(d - 1))
        if x < 0.70:
            return "(%s%s)" % (self.c(["-", "~", "not ", "#"]), self.expr(d - 1))
        if x < 0.88:
            name, ar = self.c(BUILTINS)
            if name in ("rawset", "table.insert", "table.remove", "ipairs", "pairs"):
                name, ar = "math.max", 3
            na = r.randint(1, ar)
            return "%s(%s)" % (name, ", ".join(self.expr(d - 1) for _ in range(na)))
        if x < 0.95:
            return "%s[%s]" % (self.c(["a", "b", "c"]), self.expr(d - 1))
        return "%s.%s" % (self.c(["a", "b", "c"]), self.c(["x", "y", "z", "n"]))

    def tpl_binop(self):
        op = self.c(BINOPS)
        f = self.r.random()
        if f < 0.4:
            return "return a %s b" % op
        if f < 0.7:
            return "return a %s %s" % (op, self.k())
        if f < 0.85:
            return "return %s %s a" % (self.k(), op)
        return "local x = a %s b\n    return x %s %s, math.type(x)" % (op, self.c(BINOPS), self.operand())

    def tpl_unop(self):
        u = self.c(["-", "~", "not ", "#", "- -", "-~"])
        return "local x = %sa\n    return x, %s" % (u, self.c(["x == x", "x + 0", "math.type(x)", "x == -0.0", "x == 0", "x < 1"]))

    def tpl_cmp(self):
        op = self.c(CMPOPS)
        lhs = self.c(["a", "a", self.k()])
        rhs = self.c(["b", self.k(), "c"])
        f = self.r.random()
        if f < 0.35:
            return "return %s %s %s" % (lhs, op, rhs)
        if f < 0.7:
            return "if %s %s %s then return 1 else return 2 end" % (lhs, op, rhs)
        if f < 0.85:
            return "local x = a %s b\n    if x %s %s then return x end\n    return not x" % (self.c(BINOPS[:7]), op, self.k())
        return "if not (%s %s %s) then return 'n' end return 'y'" % (lhs, op, rhs)

    def tpl_logic(self):
        return "return %s" % self.expr(2).replace("..", self.c(["+", "..", "and"]))

    def tpl_builtin(self):
        name, ar = self.c(BUILTINS)
        na = self.r.randint(0, ar + 1)
        args = ", ".join(self.c(["a", "b", "c", "a", "b", self.k()]) for _ in range(na))
        call = "%s(%s)" % (name, args)
        f = self.r.random()
        if f < 0.4:
            return "return %s" % call
        if f < 0.55:
            return "local x = %s\n    return x" % call
        if f < 0.7:
            return "return select('#', %s), %s" % (call, call)
        if f < 0.8:
            return "return {%s}" % call
        if f < 0.9:
            return "return %s, 'tail'" % call
        return "local t = {%s, %s}\n    return #t, t[1], t[2]" % (call, call)

    def tpl_tget(self):
        key = self.c(["b", "1", "0", "-1", "1.0", "2", "3", "4", "2^53", "'x'", "'y'", "'n'", "1.5", "c", "b + 1", "#a"])
        f = self.r.random()
        if f < 0.5:
            return "return a[%s]" % key
        if f < 0.7:
            return "return a.%s, a.%s" % (self.c(["x", "y", "z", "n"]), self.c(["x", "y", "z", "n"]))
        if f < 0.85:
            return "local v = a[%s]\n    return v, type(v), v == nil" % key
        return "return rawget(a, %s), #a" % key

    def tpl_tset(self):
        f = self.r.random()
        key = self.c(["b", "1", "0", "-1", "1.0", "4", "'x'", "'q'", "c", "#a + 1", "2.5"])
        if f < 0.3:
            return "local t = {}\n    t[%s] = c\n    return t[%s], #t, next(t)" % (key, key)
        if f < 0.55:
            # shape: new fields in different orders, types from other tables
            fl = ["x", "y", "z", "w"]
            self.r.shuffle(fl)
            src = self.c(["a", "b", "c", "a.x", "b[1]", "c.y", "a[b]"])
            lines = ["local o = {}"]
            for nm in fl[: self.r.randint(1, 4)]:
                lines.append("o.%s = %s" % (nm, self.c([src, "a", "b", "c", self.k()])))
            lines.append("return o.x, o.y, o.z, o.w, (o.x or 0) %s 1" % self.c(["+", "..", "<", "=="]))
            return "\n    ".join(lines)
        if f < 0.75:
            return "local t = {x = 1, y = 2}\n    t.%s = a\n    t[%s] = b\n    return t.x, t.y, t[%s]" % (self.c(["x", "y", "z"]), key, key)
        return "local t = {a, b, c}\n    t[%s] = nil\n    return #t, t[1], t[2], t[3]" % self.c(["1", "2", "3", "b", "4"])

    def tpl_for(self):
        def bound():
            return self.c(["a", "b", "c", self.c(INTK), self.c(FLK), "a", "b", "c"])
        init, lim, step = bound(), bound(), bound()
        if self.r.random() < 0.3:
            hdr = "for i = %s, %s do" % (init, lim)
        else:
            hdr = "for i = %s, %s, %s do" % (init, lim, step)
        body = self.c(["acc = acc + i", "acc = acc .. ','", "acc = i", "fs[#fs + 1] = function() return i end",
                       "acc = acc + (i // 1)", "if i > 3 then acc = acc + 1 end", "acc = math.max(acc, i)"])
        inner = """local n, first, last, acc, fs = 0, nil, nil, 0, {}
    %s
      n = n + 1
      if first == nil then first = i end
      last = i
      %s
      if n >= 25 then break end
    end
    local fv = fs[1] and fs[#fs]()
    return n, first, last, math.type(last), acc, fv""" % (hdr, body)
        if self.r.random() < 0.25:
            # nested inside outer loop, maybe in a not-taken branch
            inner = """local tot = 0
    for o = 1, 3 do
      if %s then
        %s
        local r1, r2, r3 = (function() %s end)()
        tot = tot + 1
      end
    end
    return tot""" % (self.c(["o == 2", "o == 5", "a == 'never'", "true"]), "", inner.replace("\n    ", "\n          "))
        return inner

    def tpl_while(self):
        f = self.r.random()
        op = self.c(CMPOPS[2:])
        if f < 0.35:
            return """local n = 0
    local x = a
    while x %s b and n < 20 do
      x = x %s %s
      n = n + 1
    end
    return x, n""" % (op, self.c(["+", "-", "*", ".."]), self.c(["1", "c", "0.5", "2"]))
        if f < 0.65:
            return """local n, x = 0, a
    repeat
      n = n + 1
      x = x %s c
      if n > 15 then break end
    until x %s b
    return n, x""" % (self.c(["+", "*", "-", "//"]), op)
        return """local n, x = 0, a
    ::top::
    n = n + 1
    if n < 10 and x %s b then
      x = x + %s
      goto top
    end
    return n, x""" % (op, self.c(["1", "c", "0.25"]))

    def tpl_gfor(self):
        f = self.r.random()
        if f < 0.3:
            return """local n, s = 0, {}
    for k, v in pairs(a) do
      n = n + 1
      s[#s + 1] = ser(k) .. '=' .. ser(v)
      %s
    end
    table.sort(s)
    return n, table.concat(s, ';')""" % self.c(["", "a[k] = nil", "if n > 2 then break end", "b = k"])
        if f < 0.55:
            return """local n, s = 0, 0
    for i, v in ipairs(a) do
      n = n + 1
      s = %s
      if n > 30 then break end
    end
    return n, s""" % self.c(["s + i", "v", "s + v", "i + b"])
        if f < 0.75:
            return """local n, k, v = 0, nil, nil
    repeat
      k, v = next(a, k)
      n = n + 1
    until k == nil or n > 20
    return n"""
        return """local function it(s, i)
      i = i + %s
      if i <= s then return i, i * 2 end
    end
    local n, last = 0, nil
    for i, j in it, b, %s do n = n + 1; last = j; if n > 20 then break end end
    return n, last""" % (self.c(["1", "c", "0.5"]), self.c(["0", "a", "0.0"]))

    def tpl_vararg(self):
        self.helpers.append("local function va(...) return select('#', ...), ... end")
        self.helpers.append("local function vb(x, ...) local t = {...} return #t, x, ... end")
        f = self.r.random()
        if f < 0.25:
            return "return va(a, b, c)"
        if f < 0.45:
            return "return select(%s, a, b, c)" % self.c(["a", "b", "1", "-1", "2", "'#'", "4", "-4"])
        if f < 0.65:
            return "return vb(va(a, b))"
        if f < 0.8:
            return "return %s(a, b, c)" % self.c(["vb", "va", "math.max", "select", "string.byte", "table.unpack"])
        return "local x, y, z, w = va(a)\n    return x, y, z, w"

    def tpl_closure(self):
        f = self.r.random()
        if f < 0.5:
            return """local fs = {}
    local u = a
    for i = 1, 3 do
      fs[i] = function() u = u %s b return u end
    end
    return fs[1](), fs[2](), u""" % self.c(["+", "..", "*", "//"])
        self.helpers.append("local up = 0")
        return "up = %s\n    return up, (function() return up %s c end)()" % (self.c(["a", "up + 1", "b"]), self.c(BINOPS))

    def tpl_rec(self):
        self.helpers.append("local function rf(x, d) if d <= 0 then return x end return rf(x %s 1, d - 1) end" % self.c(["+", "-", "*", "//", ".."]))
        self.helpers.append("local function tf(x, d) if d <= 0 then return x end return tf(x, d - 1) end")
        return self.c(["return rf(a, 5)", "return tf(a, 10)", "return rf(b, 3), tf(c, 2)", "return math.max(rf(a, 2), b)"])

    def tpl_tail(self):
        name, ar = self.c(BUILTINS)
        na = self.r.randint(0, ar + 1)
        return "return %s(%s)" % (name, ", ".join(self.c(["a", "b", "c"]) for _ in range(na)))

    def tpl_str(self):
        return self.c([
            "return a .. b", "return #a", "return string.sub(a, b, c)", "return a:sub(b)", "return string.byte(a, b, c)",
            "return a .. %s" % self.k(), "return %s .. a .. b" % self.k(), "return string.len(a), #a",
            "local s = a .. ''\n    return s, #s", "return (a .. b) == 'ab'", "return string.byte(a, -1)",
            "return a:byte(1, -1)", "return string.sub(a, %s, %s)" % (self.c(INTK), self.c(INTK)),
        ])

    def tpl_expr(self):
        return "return %s" % self.expr(self.r.randint(2, 4))

    def tpl_coerce(self):
        return self.c([
            "return a + 0, a * 1, a - 0", "return a // 1, a % 1", "return math.tointeger(a)", "return a | 0",
            "return tonumber(a)", "return tonumber(a, %s)" % self.c(["10", "16", "2", "36", "b"]), "return a == a + 0",
            "return a + 0.0", "return a // %s, a %% %s" % (self.k(), self.k()), "return math.floor(a), math.ceil(a)",
            "local x = a\n    if math.type(x) == 'integer' then return x + 1 end\n    return x",
            "return math.fmod(a, b), math.fmod(a, %s)" % self.k(),
        ])

    def tpl_mt(self):
        f = self.r.random()
        if f < 0.4:
            return "return a.x, a[1], a.zz"
        if f < 0.7:
            return "return a %s b" % self.c(BINOPS + CMPOPS)
        return "return #a, -a, a(b)"


    def tpl_call(self):
        self.helpers.append("local function g0() return 'g0' end")
        self.helpers.append("local function g2(x, y) return y, x end")
        self.helpers.append("local function gv(...) return ... end")
        self.helpers.append("local obj = { m = function(self, x) return self, x end }")
        return self.c([
            "return g2(a)", "return g2(a, b, c, a)", "return g0(a, b)", "return gv(a, b, c)", "return (gv(a, b, c))",
            "return {gv(a, b)}, #{gv(a, b, c)}", "return b(a)", "return a(b, c)", "return obj:m(a)", "return obj.m(a, b)",
            "return g2(gv(a, b, c))", "local x, y = g2(a, b)\n    return y %s x" % self.c(BINOPS + CMPOPS),
            "return select(-1, gv(a, b, c))", "return select(2, gv(a, b, c)), select('#', gv())",
            "return pcall(g2, a, b)", "return pcall(a, b)", "return string.format('%%s', a)",
        ])

    def tpl_coro(self):
        return self.c([
            """local co = coroutine.wrap(function(x)
      for i = 1, 3 do x = x %s coroutine.yield(x) end
      return x
    end)
    co(a); co(b); co(c)
    return co(b)""" % self.c(["+", "..", "*", "-"]),
            """local co = coroutine.create(function(x, y)
      local ok, v = pcall(function() return coroutine.yield(x %s y) end)
      return ok, v
    end)
    local r1 = { coroutine.resume(co, a, b) }
    local r2 = { coroutine.resume(co, c) }
    return #r1, r1[2], #r2, r2[2], r2[3]""" % self.c(BINOPS),
            """local n = 0
    local co = coroutine.wrap(function()
      for i = a, b do n = n + 1; coroutine.yield(i) if n > 10 then break end end
      return "end"
    end)
    local last
    for j = 1, 12 do last = co() if last == "end" then break end end
    return n, last""",
        ])

    def tpl_meta2(self):
        return self.c([
            "return a.x, a.y, a[1]", "a.q = b\n    return rawget(a, 'q'), a.q", "return a == b, a ~= b, rawequal(a, b)",
            "return a < b, a <= b", "return #a, rawlen(a)", "return a .. b", "return -a", "return a(b, c)",
            "return a + b, b + a", "return a[b], a[c]", "return getmetatable(a)", "local x = a.x\n    return x %s 1" % self.c(BINOPS),
        ])

    def tpl_close(self):
        self.helpers.append("local CL = setmetatable({}, { __close = function(o, e) pos = pos end })")
        return "local x <close> = %s\n    return a %s b" % (self.c(["CL", "nil", "false"]), self.c(BINOPS))


    def tpl_tarr(self):
        fill = self.c(["t[i] = a", "t[#t + 1] = a", "t[i] = i", "rawset(t, i, b)", "table.insert(t, a)", "t[i * 1.0] = i"])
        upd = self.c(["t[b] = c", "t[b] = nil", "t[#t] = nil", "t[#t + 1] = c", "table.insert(t, 1, c)", "table.remove(t, b)",
                      "t[b + 0.0] = c", "t[c] = b", "rawset(t, b, c)"])
        rd = self.c(["t[b]", "t[#t]", "t[1]", "rawget(t, b)", "t[b // 1]", "t[c]", "select('#', table.unpack(t))"])
        return """local t = {}
    for i = 1, %s do %s end
    %s
    local s = 0
    for k, v in ipairs(t) do s = s + 1 if s > 40 then break end end
    return #t, s, %s, next(t)""" % (self.c(["8", "3", "1", "0", "17"]), fill, upd, rd)

    def tpl_upv(self):
        self.helpers.append("local cnt = 0")
        self.helpers.append("local function inc(d) cnt = cnt + d return cnt end")
        self.helpers.append("local function mkc(x) return function(y) x = x %s y return x end end" % self.c(["+", "..", "*", "-"]))
        return self.c([
            "cnt = 0\n    return inc(a), inc(b), cnt",
            "local g = mkc(a)\n    return g(b), g(c)",
            "local x = a\n    local function h() return x end\n    x = b\n    return h(), x",
            "local fs = {}\n    for i = 1, 3 do local j = i + a fs[i] = function() return j, i end end\n    return fs[1](), fs[3]()",
        ])

    def tpl_strmeth(self):
        return self.c([
            "return a:len(), a:upper()", "return (a):sub(b, c)", "return a:byte(b)", "return a:rep(2), #a:rep(3, ',')",
            "return ('x'):rep(b)", "return a:find(b, 1, true)", "return tostring(a):len()", "return a:sub(-b)",
        ])

    TEMPLATES = [("binop", 6), ("unop", 2), ("cmp", 5), ("logic", 1), ("builtin", 6), ("tget", 3), ("tset", 3),
                 ("for", 5), ("while", 2), ("gfor", 2), ("vararg", 2), ("closure", 1), ("rec", 1), ("tail", 2),
                 ("str", 2), ("expr", 4), ("coerce", 3), ("mt", 1), ("call", 3), ("coro", 2), ("meta2", 3),
                 ("close", 1), ("tarr", 4), ("upv", 2), ("strmeth", 1), ("limits", 5)]

    def warm_fams(self):
        # mostly numeric families for warmup; sometimes tables or strings
        r = self.r.random()
        if r < 0.55:
            return [self.c(NUMF) for _ in range(3)]
        if r < 0.7:
            return [self.c(["tt", "tm", "tb"]), self.c(NUMF), self.c(NUMF)]
        if r < 0.85:
            return [self.c(["st", "ns"]), self.c(["sp", "si", "sn"]), self.c(["sp", "si", "sn"])]
        return [self.c(FAMS) for _ in range(3)]

    def pval(self):
        fam, n = self.c(POOL + [("nil", 1)] + [("si", 10)] * 2 + [("fl", 10), ("bi", 10), ("fs", 10)])
        if fam == "nil":
            return "nil"
        return "V.%s[%d]" % (fam, self.r.randint(1, n))

    def limits(self, tag):
        r = self.r
        step = """n = n + 1
      local x = %s
      %s
      if %s then t = t + 1 end""" % (self.c(LIMIT_EXPRS), self.c(LIMIT_ACCS), self.c(LIMIT_TESTS))
        form = r.random()
        if form < 0.6:
            loop = """for i = lo, hi, st do
      %s
      if n >= 40 then break end
    end""" % step
        elif form < 0.8:
            loop = """local i = lo
    while n < 40 and (st > 0 and i <= hi or st < 0 and i >= hi) do
      %s
      i = i + st
    end""" % step
        else:
            loop = """for j = 1, 12 do
      local i = lo + (j - 1) * st
      %s
      if i == hi then break end
    end""" % step
        body = "local s, t, n = 0, 0, 0\n    %s\n    return s, t, n, math.type(s)" % loop
        warm = self.c(LIMIT_WARM)
        out = ["do -- %s limits" % tag, "  local function f(lo, hi, st, k)\n    %s\n  end" % body,
               "  for w = 1, %d do pcall(f, %s) end" % (self.c([30, 300, 1000]), warm),
               '  rec("%sw", pcall(f, %s))' % (tag, warm)]
        for p in range(r.randint(5, 12)):
            args = ", ".join([self.c(LIMIT_BOUNDS), self.c(LIMIT_BOUNDS), self.c(LIMIT_STEPS), self.c(LIMIT_KS)])
            if r.random() < 0.5:
                out.append('  rec("%sp%d", pcall(f, %s))' % (tag, p, args))
            else:
                # The same call from a compiled loop of its own.
                out.append('  do local ok, a, b, c, d for q = 1, 3 do ok, a, b, c, d = pcall(f, %s) end '
                           'rec("%sp%d", ok, a, b, c, d) end' % (args, tag, p))
        out += ['  rec("%sz", pcall(f, %s))' % (tag, warm), "end"]
        return "\n".join(out), body

    def instance(self, idx, kind=None):
        """(source, kind, f's body) of instance `idx`; its output lines start with I<idx>."""
        self.helpers = []
        if kind is None:
            kind = self.c([t for t, w in self.TEMPLATES for _ in range(w)])
        if kind == "limits":
            src, body = self.limits("I%d" % idx)
            return src, kind, body
        body = getattr(self, "tpl_" + kind)()
        wf = self.warm_fams()
        if kind == "for" and self.r.random() < 0.8:
            wf = [self.c(["si", "sp", "sn", "ib", "bi"]) for _ in range(3)]
        if kind == "meta2":
            wf = [self.c(["tm", "tb", "tt"]), self.c(NUMF + ["tm", "st"]), self.c(NUMF + ["tm"])]
        tag = "I%d" % idx
        coro = self.r.random() < 0.12
        nwarm = self.c([300, 600, 1000, 2000])
        event = ""
        restore = ""
        er = self.r.random()
        if er < 0.06 and "math.floor" in body:
            event = "local sv = math.floor; math.floor = function(...) return 'FL', ... end"
            restore = "math.floor = sv"
        elif er < 0.06 and "math." in body:
            event = "local sv = math.max; math.max = math.min"
            restore = "math.max = sv"
        elif er < 0.1:
            event = "setmetatable(V.tb[%d], %s)" % (self.r.randint(1, 3), self.c(["MT", "MTI", "nil"]))
        elif er < 0.2:
            event = self.c(["MTI.__index.x = 'changed'", "MT.__add = nil; MT.__index = MTI.__index", "MTI.__index = function(t, k) return 'fn' end",
                            "MT.__eq = nil; MT.__lt = nil", "MT.__len = nil", "MT.__call = nil", "MTI.__index.x = nil",
                            "V.tb[4].x = nil; V.tb[4][1] = nil", "setmetatable(V.tb[5], MT)"])
        elif er < 0.23:
            event = "V.tb[3].x = %s; V.tb[3].q = 1; V.tb[2][4] = 'four'" % self.c(["'str'", "nil", "2.5", "{}"])
        if coro:
            loop = """local co = coroutine.wrap(function()
      for i = pos + 1, n do
        pos = i
        local a, b, c = LA[(i - 1) % LA.n + 1], LB[i % LB.n + 1], LC[(i + 1) % LC.n + 1]
        local s = sers(f(a, b, c))
        h[s] = (h[s] or 0) + 1
        if i % 7 == 0 then coroutine.yield() end
      end
      return "done"
    end)
    while co() ~= "done" do end"""
        else:
            loop = """for i = pos + 1, n do
      pos = i
      local a, b, c = LA[(i - 1) % LA.n + 1], LB[i % LB.n + 1], LC[(i + 1) % LC.n + 1]
      local s = sers(f(a, b, c))
      h[s] = (h[s] or 0) + 1
    end"""
        out = ["do -- %s %s" % (tag, kind)]
        out += ["  " + h for h in self.helpers]
        out.append("  local function f(a, b, c)\n    %s\n  end" % body)
        out.append("""  local function run(LA, LB, LC, n, h)
    %s
  end
  local function drive(LA, LB, LC, n, tag)
    local h, errs = {}, 0
    pos = 0
    while pos < n do
      local ok, e = pcall(run, LA, LB, LC, n, h)
      if ok then break end
      local k = errn(e)
      h[k] = (h[k] or 0) + 1
      errs = errs + 1
      if errs > 300 then break end
    end
    hist(tag, h)
  end""" % loop)
        out.append('  drive(V.%s, V.%s, V.%s, %d, "%sw")' % (wf[0], wf[1], wf[2], nwarm, tag))
        if event:
            out.append("  " + event)
        nph = self.r.randint(6, 16)
        for j in range(nph):
            vs = [self.pval() for _ in range(3)]
            # often keep two slots in warm family, change one
            if self.r.random() < 0.5:
                keep = self.r.randint(0, 2)
                for s in range(3):
                    if s != keep:
                        vs[s] = "V.%s[%d]" % (wf[s], self.r.randint(1, 3)) if wf[s] != "nil1" else "nil"
            if self.r.random() < 0.6:
                out.append('  drive({n = 1, %s}, {n = 1, %s}, {n = 1, %s}, 2, "%sp%d")' % (vs[0], vs[1], vs[2], tag, j))
            else:
                out.append('  rec("%sq%d", pcall(f, %s, %s, %s))' % (tag, j, vs[0], vs[1], vs[2]))
        out.append('  drive(V.%s, V.%s, V.%s, 200, "%sz")' % (wf[0], wf[1], wf[2], tag))
        if restore:
            out.append("  " + restore)
        out.append("end")
        return "\n".join(out), kind, body

    def program(self, kind=None):
        """(source, {instance number: (kind, f's body)})."""
        parts, insts = [PRELUDE], {}
        for i in range(1, self.r.randint(2, 5) + 1):
            src, k, body = self.instance(i, kind)
            parts.append(src)
            insts[i] = (k, body)
        # The last line shows the program ran to its end.
        parts.append('io.write(table.concat(OUT, "\\n"), "\\nCHECK ", #OUT, "\\n")\n')
        return "\n".join(parts), insts


KINDS = [t for t, _ in G.TEMPLATES]

if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3 or (len(sys.argv) == 3 and sys.argv[2] not in KINDS):
        sys.exit(__doc__ + "\nkinds: " + " ".join(KINDS))
    sys.stdout.write(G(int(sys.argv[1])).program(sys.argv[2] if len(sys.argv) == 3 else None)[0])
