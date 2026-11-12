//! Templates that stay where tcvm and reference Lua are meant to agree, so their units are
//! strict: no float or table reaches a string, no length is taken of a table that may have
//! holes, no libm function whose last bit may differ, no `pairs` order. They aim at the
//! compiler (constants in conditions), the native call protocol (result counts) and what the
//! JIT assumes of a value that went through memory.

use super::{CMPOPS, INTK};
use crate::generate::Gen;

/// The templates of this file.
pub(super) const STRICT: &[&str] = &["konst", "tailnat", "flow", "arity"];

/// Atoms of a condition: values that fold, and functions as values.
const CX_K: &[&str] = &[
    "nil",
    "true",
    "false",
    "0",
    "1",
    "-1",
    "2",
    "8",
    "65",
    "0.0",
    "-0.0",
    "0.5",
    "2.5",
    "1e300",
    "2^53",
    "100.25",
    "1099511627776",
    "2147483648",
    "9223372036854775807",
    "math.mininteger",
    "\"\"",
    "\"x\"",
    "\"10\"",
    "\"x\\0y\"",
    "#\"\"",
    "#\"a\"",
    "#\"abc\"",
    "G1",
    "G2",
    "math.max",
    "type",
    "rawequal",
];
/// Numbers that compare without raising.
const CX_NUM: &[&str] = &[
    "0",
    "1",
    "-1",
    "2",
    "7",
    "8",
    "65",
    "0.5",
    "2.5",
    "-0.0",
    "1e300",
    "2^53",
    "100.25",
    "1099511627776",
    "2147483648",
    "9223372036854775807",
    "math.mininteger",
    "#\"\"",
    "#\"a\"",
];

/// Constants for a native's arguments: none whose text could differ.
const NAT_K: &[&str] = &[
    "0",
    "1",
    "-1",
    "2",
    "3",
    "7",
    "-5",
    "0.5",
    "2.5",
    "\"abc\"",
    "\"10\"",
    "\"\"",
    "nil",
    "true",
    "2147483648",
    "math.mininteger",
];

/// Natives and the fewest and most arguments a call gives them (one fewer is tried too).
const NATS: &[(&str, i64, i64)] = &[
    ("math.max", 1, 3),
    ("math.min", 1, 3),
    ("math.abs", 1, 1),
    ("math.floor", 1, 1),
    ("math.ceil", 1, 1),
    ("math.fmod", 2, 2),
    ("math.tointeger", 1, 1),
    ("math.type", 1, 1),
    ("math.ult", 2, 2),
    ("math.sqrt", 1, 1),
    ("string.sub", 2, 3),
    ("string.byte", 1, 3),
    ("string.len", 1, 1),
    ("string.upper", 1, 1),
    ("string.char", 1, 2),
    ("select", 1, 3),
    ("type", 1, 1),
    ("rawequal", 2, 2),
    ("rawget", 2, 2),
    ("assert", 1, 3),
    ("tonumber", 1, 2),
    ("getmetatable", 1, 1),
];
const STR_METHS: &[&str] = &[
    "a:sub(b, c)",
    "a:sub(b)",
    "a:byte(b)",
    "a:byte(b, c)",
    "a:len()",
    "a:upper()",
];

/// How a value of the unit's `v` reaches memory and back.
const ROUTES: &[&str] = &[
    "local o = {}\no.x = v\nv = o.x",
    "local o = {}\no.x = v\nv = (o.x or K)",
    "local o = {}\no.y = K\no.x = v\nv = o.x",
    "local t = {}\nt[1] = v\nv = t[1]",
    "v = ({v})[1]",
    "v = ({x = v}).x",
    "v = rawget({v, K}, 1)",
    "local function id(x) return x end\nv = id(v)",
    "v = (function(...) return ... end)(v)",
    "v = select(1, v)",
    "v = select(-1, K, v)",
    "local ok, r = pcall(function(x) return x end, v)\nv = r",
    "local up = v\nv = (function() return up end)()",
    "local function h() v = v end\nh()",
    "local w = b\nv, w = w, v\nv, w = w, v",
    "local co = coroutine.wrap(function(x) local y = coroutine.yield(x) return y end)\nco(v)\nv = co(v)",
    "G1 = v\nv = G1",
    "local o = setmetatable({}, {__index = function(t, k) return v end})\nv = o.x",
    "local u = {v}\nlocal function g() return u[1] end\nv = g()",
];
/// What consumes `v`, by a typed operation.
const SINKS: &[&str] = &[
    "v + K",
    "K - v",
    "v * 2",
    "v // 1",
    "v % 3",
    "v / 2",
    "-v",
    "v | 0",
    "v << 1",
    "~v",
    "v < K",
    "K <= v",
    "v == K",
    "v ~= v",
    "v[1]",
    "v.x",
    "v(1)",
    "math.floor(v)",
    "math.max(v, 1)",
    "type(v)",
    "math.type(v)",
    "v and 1 or 2",
    "not v",
    "({10, 20})[v]",
    "(v or K) + 1",
    "(v or K) < 1",
    "(v or 0) == 1",
    "(v or K) * K",
];

const ARITY_HELPERS: &[&str] = &[
    "local function r0() end",
    "local function r1(x) return x end",
    "local function r2(x, y) return x, y end",
    "local function rv(...) return ... end",
    "local function rn(...) return select('#', ...) end",
    "local function rt(...) return rv(...) end",
    "local function rp(x) return (rv(x)) end",
];
const ARITY_CALLS: &[&str] = &[
    "r0()",
    "r0(a)",
    "r1(a)",
    "r1()",
    "r2(a, b)",
    "r2(a)",
    "rv(a, b, c)",
    "rv()",
    "rv(nil, nil)",
    "rt(a, b)",
    "rt()",
    "rp(a)",
    "string.byte(\"abc\", 10)",
    "string.byte(\"\", 1)",
    "select(5, a, b)",
    "select(2, a, b)",
    "select(-1, a, b, c)",
    "(function() end)()",
    "pcall(r0)",
    "pcall(r2, a, b)",
    "pcall(error)",
    "coroutine.wrap(r0)()",
    "coroutine.wrap(r2)(a, b)",
    "next({})",
    "math.max(a, b)",
    "rawequal(a, b)",
    "rawset({}, 1, a)",
    "table.insert({}, a)",
    "table.remove({})",
];
const ARITY_USES: &[&str] = &[
    "return X",
    "return (X)",
    "local p, q, r = X\nreturn p, q, r",
    "local p = X\nreturn p",
    "return select('#', X)",
    "return rn(X)",
    "return r2(X)",
    "return r2(X, a)",
    "return r2(a, X)",
    "return X, X",
    "local t = {X}\nreturn t[1], t[2], t[3]",
    "local p, q = X, a\nreturn p, q",
    "return rv(X, b)",
    "local p = X\nlocal q = X\nreturn p == q, p, q",
    "for i = 1, 2 do local p = X\n  if i == 2 then return p, i end\nend",
];

impl Gen {
    fn cx_const(&mut self) -> &'static str {
        self.tc(CX_K)
    }

    fn cx_atom(&mut self) -> String {
        let x = self.r.f64();
        if x < 0.4 {
            return self.tc(&["a", "b", "c"]).to_string();
        }
        if x < 0.55 {
            return self
                .tc(&[
                    "rawequal(a, b)",
                    "(type(a) == \"table\")",
                    "(math.type(a) == \"float\")",
                    "(a == b)",
                    "(a ~= nil)",
                    "(type == \"function\")",
                    "(math.type == \"float\")",
                ])
                .to_string();
        }
        self.cx_const().to_string()
    }

    /// A condition over `a`, `b`, `c` and constants, with a constant often on the left of an
    /// `and`/`or`: what the compiler folds.
    fn t_cx(&mut self, d: i32) -> String {
        let x = self.r.f64();
        if d <= 0 || x < 0.2 {
            return self.cx_atom();
        }
        if x < 0.65 {
            let op = self.tc(&["and", "or"]);
            let l = if self.r.chance(0.45) {
                self.cx_const().to_string()
            } else {
                self.t_cx(d - 1)
            };
            let r = self.t_cx(d - 1);
            return format!("({l} {op} {r})");
        }
        if x < 0.78 {
            return format!("(not {})", self.t_cx(d - 1));
        }
        let op = self.tc(CMPOPS);
        let l = if self.r.chance(0.5) {
            self.tc(CX_NUM)
        } else {
            self.tc(&["a", "b", "c"])
        };
        let r = self.tc(CX_NUM);
        format!("({l} {op} {r})")
    }

    pub(super) fn t_konst(&mut self) -> String {
        let c = self.t_cx(3);
        let c2 = self.t_cx(2);
        let tail = self.tc(&[
            "return 1",
            "return a",
            "return b, c",
            "return math.type(a)",
            "return select('#', a, b, c)",
            "return math.max(a, b)",
        ]);
        let opts = [
            format!("if {c} then end\n{tail}"),
            format!("if {c} then elseif {c2} then end\n{tail}"),
            format!(
                "local x = 0\nif {c} then x = 1 elseif {c2} then x = 2 else x = 3 end\nreturn x"
            ),
            format!(
                "local n = 0\nwhile {c} do\n  n = n + 1\n  if n > 5 then break end\nend\nreturn n"
            ),
            format!(
                "local n = 0\nrepeat\n  n = n + 1\n  if n > 5 then break end\nuntil {c}\nreturn n"
            ),
            format!("local n = 0\nrepeat n = n + 1 until {c} or n > 3\nreturn n"),
            format!("return {c}"),
            format!("local x = {c}\nreturn x, type(x)"),
            format!("return {c}, {c2}"),
            format!("local t = {{{c}, {c2}}}\nreturn t[1], t[2]"),
            format!("local n = 0\nfor i = 1, 4 do\n  if {c} then n = n + i end\nend\nreturn n"),
            format!("return select('#', {c}), not {c2}"),
            format!("local x = {c}\nlocal y = {c2}\nreturn x == y, x and 1, y or 2"),
            format!("if {c} then return 'A' end\nif {c2} then return 'B' end\nreturn 'C'"),
        ];
        self.r.pick(&opts).clone()
    }

    fn nat_call(&mut self) -> String {
        if self.r.chance(0.15) {
            return self.tc(STR_METHS).to_string();
        }
        let &(name, lo, hi) = self.r.pick(NATS);
        let n = self.r.range((lo - 1).max(0), hi);
        let mut args: Vec<&str> = (0..n)
            .map(|_| {
                let k = self.tc(NAT_K);
                self.tc(&["a", "b", "c", "v", "v", "a", "b", k])
            })
            .collect();
        // A `string` function turns a number into text, and a float's text differs.
        if name.starts_with("string.") && name != "string.char" && !args.is_empty() {
            args[0] = self.tc(&["\"abc\"", "\"hello world\"", "\"\"", "\"x\\0y\"", "\"10\""]);
        }
        format!("{name}({})", args.join(", "))
    }

    /// A native in tail position, in a function that holds a closure over one of its locals.
    pub(super) fn t_tailnat(&mut self) -> String {
        let (decl, call_h) = *self.r.pick(&[
            ("local v = a", false),
            ("local v = a\nlocal function h() v = v end", true),
            ("local v = b\nlocal function h() v = a end", true),
            (
                "local v = a\nlocal h = function() local g = function() v = v end return 2 end",
                true,
            ),
            ("local function h() c = a end\nlocal v = b", true),
            (
                "local v = a\nlocal fs = {}\nfor i = 1, 2 do fs[i] = function() v = i end end\nlocal function h() fs[1]() end",
                true,
            ),
            ("local v = c\nlocal function h() return v end", true),
        ]);
        let pre = if !call_h {
            ""
        } else {
            *self.r.pick(&["", "", "h()\n", "if a == b then h() end\n"])
        };
        let call = self.nat_call();
        let call2 = self.nat_call();
        let tail = match self.r.below(9) {
            0 => format!("return {call}"),
            1 => format!("return ({call})"),
            2 => format!("return {call}, v"),
            3 => format!("return v, {call}"),
            4 => format!("local x, y, z = {call}\nreturn x, y, z"),
            5 => format!("return math.abs({call})"),
            6 => format!("return select('#', {call})"),
            7 => format!("return {call2}, {call}"),
            _ => format!(
                "return pcall({}, a, b)",
                self.tc(&["math.max", "math.min", "string.sub", "select", "rawequal"])
            ),
        };
        format!("{decl}\n{pre}{tail}")
    }

    /// A value through memory, then a typed operation on it.
    pub(super) fn t_flow(&mut self) -> String {
        let k = self.tc(INTK);
        let mut out = format!("local v = {}\n", self.tc(&["a", "b", "c"]));
        for _ in 0..self.r.range(1, 3) {
            out.push_str(&self.tc(ROUTES).replace('K', k));
            out.push('\n');
        }
        let s1 = self.tc(SINKS).replace('K', k);
        if self.r.chance(0.3) {
            let s2 = self.tc(SINKS).replace('K', k);
            out.push_str(&format!("return v, {s1}, {s2}"));
        } else if self.r.chance(0.5) {
            out.push_str(&format!("return v, {s1}"));
        } else {
            out.push_str(&format!("local r = {s1}\nreturn r, v"));
        }
        out
    }

    /// Calls that return no, one or many values, taken in different ways.
    pub(super) fn t_arity(&mut self) -> String {
        for h in ARITY_HELPERS {
            self.helper_stmt(h);
        }
        let call = self.tc(ARITY_CALLS);
        let call2 = self.tc(ARITY_CALLS);
        let u = self.tc(ARITY_USES);
        let first = u.replacen('X', call, 1);
        // The second `X`, if any, is another call.
        first.replacen('X', call2, 1)
    }
}
