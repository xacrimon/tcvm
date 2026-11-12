//! Hand-written templates: small bodies of `f` that aim at one thing each, ported from the old
//! Python generator (`tools/fuzz/gen.py`).

mod stress;
mod strict;

use super::Gen;
use crate::ast::{Block, Stmt};

/// Whether a template's units stay within what reference Lua agrees on, so are compared with it.
pub(super) fn is_strict(kind: &str) -> bool {
    strict::STRICT.contains(&kind) || stress::STRICT.contains(&kind)
}

/// Whether a template's call is slow, so the driver makes fewer.
pub(super) fn is_heavy(kind: &str) -> bool {
    stress::HEAVY.contains(&kind)
}

const FAMS: &[&str] = &[
    "si", "bi", "fl", "fs", "ns", "st", "ot", "tb", "fn", "nil1", "sp", "sn", "fp", "mn", "ib",
    "tt", "tm", "mx", "ms",
];
const NUMF: &[&str] = &["si", "sp", "fl", "fp", "mn", "ib", "bi"];

const KONST: &[&str] = &[
    "0",
    "1",
    "-1",
    "2",
    "3",
    "8",
    "0.0",
    "-0.0",
    "1.0",
    "0.5",
    "-2.0",
    "2147483648",
    "2147483647",
    "-2147483648",
    "1099511627776",
    "9223372036854775807",
    "math.mininteger",
    "2^53",
    "1e300",
    "63",
    "64",
    "-64",
    "65",
    "0x7fffffff",
    "\"10\"",
    "\"abc\"",
    "\"0x10\"",
    "4.0",
    "-3",
];
const INTK: &[&str] = &[
    "0",
    "1",
    "-1",
    "2",
    "3",
    "7",
    "-5",
    "2147483647",
    "2147483648",
    "-2147483648",
    "1099511627776",
    "9223372036854775807",
    "math.mininteger",
    "0x7fffffff",
];
const FLK: &[&str] = &[
    "0.0", "-0.0", "1.0", "0.5", "-2.0", "2.5", "2^53", "1e300", "3.0", "-1.5",
];

const BINOPS: &[&str] = &[
    "+", "-", "*", "/", "//", "%", "^", "&", "|", "~", "<<", ">>", "..",
];
const CMPOPS: &[&str] = &["==", "~=", "<", "<=", ">", ">="];
const BINCMP: &[&str] = &[
    "+", "-", "*", "/", "//", "%", "^", "&", "|", "~", "<<", ">>", "..", "==", "~=", "<", "<=",
    ">", ">=",
];

const BUILTINS: &[(&str, i64)] = &[
    ("math.floor", 1),
    ("math.ceil", 1),
    ("math.abs", 1),
    ("math.sqrt", 1),
    ("math.min", 3),
    ("math.max", 3),
    ("math.fmod", 2),
    ("math.sin", 1),
    ("math.cos", 1),
    ("math.tan", 1),
    ("math.exp", 1),
    ("math.log", 2),
    ("math.atan", 2),
    ("math.tointeger", 1),
    ("type", 1),
    ("tostring", 1),
    ("tonumber", 2),
    ("rawget", 2),
    ("rawset", 3),
    ("rawequal", 2),
    ("rawlen", 1),
    ("getmetatable", 1),
    ("select", 3),
    ("assert", 3),
    ("string.len", 1),
    ("string.sub", 3),
    ("string.byte", 3),
    ("table.insert", 3),
    ("table.remove", 2),
    ("next", 2),
    ("math.ult", 2),
    ("math.type", 1),
    ("string.char", 2),
    ("string.upper", 1),
    ("table.unpack", 3),
    ("ipairs", 1),
    ("pairs", 1),
];

const LIMIT_EXPRS: &[&str] = &[
    "i + k",
    "i - k",
    "i * k",
    "i // k",
    "i % k",
    "i << k",
    "i >> k",
    "-i",
    "i + 1",
    "i - 1",
    "i * 2",
    "i * i",
    "(i + k) - k",
    "i ~ k",
    "i & k",
    "i | k",
    "k - i",
    "i / k",
    "i ^ 2",
    "math.abs(i)",
    "i + 0.5",
    "i // 2",
    "i % 3",
    "math.min(i, k)",
    "math.max(i, k)",
    "math.tointeger(i)",
    "math.ult(i, k) and 1 or 0",
    "i + k + 1",
    "(i * k) // 1",
    "math.fmod(i, k)",
    "math.floor(i / 2)",
];
const LIMIT_TESTS: &[&str] = &[
    "x > i",
    "x < k",
    "x == i",
    "x <= 0",
    "x ~= x",
    "i + 1 > i",
    "i - 1 < i",
    "x >= k",
    "math.type(x) == 'float'",
    "x == i + k",
    "i == hi",
    "i * 2 > i",
];
const LIMIT_ACCS: &[&str] = &[
    "s = s ~ math.tointeger(x // 1 | 0)",
    "s = s + x",
    "s = x",
    "s = math.max(s, x)",
    "s = s + (x > 0 and 1 or 0)",
];
const LIMIT_BOUNDS: &[&str] = &[
    "0",
    "1",
    "-1",
    "10",
    "-10",
    "math.maxinteger",
    "math.mininteger",
    "math.maxinteger - 5",
    "math.mininteger + 5",
    "2147483647",
    "2147483648",
    "-2147483648",
    "-2147483649",
    "2147483640",
    "1 << 62",
    "(1 << 53) - 3",
];
const LIMIT_STEPS: &[&str] = &[
    "1",
    "-1",
    "2",
    "-2",
    "3",
    "1 << 62",
    "math.mininteger",
    "math.maxinteger",
    "2147483647",
];
const LIMIT_KS: &[&str] = &[
    "1",
    "-1",
    "0",
    "2",
    "3",
    "63",
    "64",
    "-64",
    "1 << 40",
    "math.maxinteger",
    "math.mininteger",
    "2147483647",
    "0.5",
    "-2",
    "2^53",
    "7",
];
const LIMIT_WARM: &[&str] = &[
    "1, 10, 1, 2",
    "1, 20, 1, 3",
    "-5, 5, 1, 1",
    "10, 1, -1, 2",
    "0, 30, 3, 7",
];

/// The templates (all but `limits`) and how often a unit takes each.
const TEMPLATES: &[(&str, u32)] = &[
    ("binop", 6),
    ("unop", 2),
    ("cmp", 5),
    ("logic", 1),
    ("builtin", 6),
    ("tget", 3),
    ("tset", 3),
    ("for", 5),
    ("while", 2),
    ("gfor", 2),
    ("vararg", 2),
    ("closure", 1),
    ("rec", 1),
    ("tail", 2),
    ("str", 2),
    ("expr", 4),
    ("coerce", 3),
    ("mt", 1),
    ("call", 3),
    ("coro", 2),
    ("meta2", 3),
    ("close", 1),
    ("tarr", 4),
    ("upv", 2),
    ("strmeth", 1),
    ("konst", 6),
    ("tailnat", 6),
    ("flow", 6),
    ("arity", 4),
    ("chain", 3),
    ("recur", 3),
    ("callslot", 3),
    ("hoist", 4),
];

impl Gen {
    fn tc(&mut self, l: &[&'static str]) -> &'static str {
        self.r.pick(l)
    }

    fn tk(&mut self) -> &'static str {
        self.tc(KONST)
    }

    fn toperand(&mut self) -> &'static str {
        let k = self.tk();
        self.tc(&["a", "b", "c", k])
    }

    /// A helper definition ahead of `f`, once per unit.
    fn helper_stmt(&mut self, text: &str) {
        let s = Stmt::Raw(text.to_string());
        if !self.pre.contains(&s) {
            self.pre.push(s);
        }
    }

    fn t_builtin_pick(&mut self) -> (&'static str, i64) {
        *self.r.pick(BUILTINS)
    }

    /// A random expression tree over `a`, `b`, `c`.
    fn t_expr(&mut self, d: i32) -> String {
        if d <= 0 || self.r.f64() < 0.25 {
            let k = self.tk();
            return self.tc(&["a", "b", "c", "a", "b", k]).to_string();
        }
        let x = self.r.f64();
        if x < 0.45 {
            let l = self.t_expr(d - 1);
            let op = self.tc(BINOPS);
            let r = self.t_expr(d - 1);
            return format!("({l} {op} {r})");
        }
        if x < 0.55 {
            let l = self.t_expr(d - 1);
            let op = self.tc(CMPOPS);
            let r = self.t_expr(d - 1);
            return format!("({l} {op} {r})");
        }
        if x < 0.62 {
            let l = self.t_expr(d - 1);
            let op = self.tc(&["and", "or"]);
            let r = self.t_expr(d - 1);
            return format!("({l} {op} {r})");
        }
        if x < 0.70 {
            let op = self.tc(&["-", "~", "not ", "#"]);
            let e = self.t_expr(d - 1);
            // The space keeps `-` off a negative constant: `--` starts a comment.
            return format!("({op} {e})");
        }
        if x < 0.88 {
            let (mut name, mut ar) = self.t_builtin_pick();
            if matches!(
                name,
                "rawset" | "table.insert" | "table.remove" | "ipairs" | "pairs"
            ) {
                (name, ar) = ("math.max", 3);
            }
            let na = self.r.range(1, ar);
            let args: Vec<String> = (0..na).map(|_| self.t_expr(d - 1)).collect();
            return format!("{name}({})", args.join(", "));
        }
        if x < 0.95 {
            let t = self.tc(&["a", "b", "c"]);
            let e = self.t_expr(d - 1);
            return format!("{t}[{e}]");
        }
        let t = self.tc(&["a", "b", "c"]);
        let f = self.tc(&["x", "y", "z", "n"]);
        format!("{t}.{f}")
    }

    fn t_binop(&mut self) -> String {
        let op = self.tc(BINOPS);
        let f = self.r.f64();
        if f < 0.4 {
            return format!("return a {op} b");
        }
        if f < 0.7 {
            let k = self.tk();
            return format!("return a {op} {k}");
        }
        if f < 0.85 {
            let k = self.tk();
            return format!("return {k} {op} a");
        }
        let op2 = self.tc(BINOPS);
        let o = self.toperand();
        format!("local x = a {op} b\nreturn x {op2} {o}, math.type(x)")
    }

    fn t_unop(&mut self) -> String {
        let u = self.tc(&["-", "~", "not ", "#", "- -", "-~"]);
        let t = self.tc(&[
            "x == x",
            "x + 0",
            "math.type(x)",
            "x == -0.0",
            "x == 0",
            "x < 1",
        ]);
        format!("local x = {u}a\nreturn x, {t}")
    }

    fn t_cmp(&mut self) -> String {
        let op = self.tc(CMPOPS);
        let k1 = self.tk();
        let lhs = self.tc(&["a", "a", k1]);
        let k2 = self.tk();
        let rhs = self.tc(&["b", k2, "c"]);
        let f = self.r.f64();
        if f < 0.35 {
            return format!("return {lhs} {op} {rhs}");
        }
        if f < 0.7 {
            return format!("if {lhs} {op} {rhs} then return 1 else return 2 end");
        }
        if f < 0.85 {
            let b = self.tc(&BINOPS[..7]);
            let k = self.tk();
            return format!("local x = a {b} b\nif x {op} {k} then return x end\nreturn not x");
        }
        format!("if not ({lhs} {op} {rhs}) then return 'n' end return 'y'")
    }

    fn t_logic(&mut self) -> String {
        let e = self.t_expr(2);
        let rep = self.tc(&["+", "..", "and"]);
        format!("return {}", e.replace("..", rep))
    }

    fn t_builtin(&mut self) -> String {
        let (name, ar) = self.t_builtin_pick();
        let na = self.r.range(0, ar + 1);
        let args: Vec<&str> = (0..na)
            .map(|_| {
                let k = self.tk();
                self.tc(&["a", "b", "c", "a", "b", k])
            })
            .collect();
        let call = format!("{name}({})", args.join(", "));
        let f = self.r.f64();
        if f < 0.4 {
            format!("return {call}")
        } else if f < 0.55 {
            format!("local x = {call}\nreturn x")
        } else if f < 0.7 {
            format!("return select('#', {call}), {call}")
        } else if f < 0.8 {
            format!("return {{{call}}}")
        } else if f < 0.9 {
            format!("return {call}, 'tail'")
        } else {
            format!("local t = {{{call}, {call}}}\nreturn #t, t[1], t[2]")
        }
    }

    fn t_tget(&mut self) -> String {
        let key = self.tc(&[
            "b", "1", "0", "-1", "1.0", "2", "3", "4", "2^53", "'x'", "'y'", "'n'", "1.5", "c",
            "b + 1", "#a",
        ]);
        let f = self.r.f64();
        if f < 0.5 {
            return format!("return a[{key}]");
        }
        if f < 0.7 {
            let f1 = self.tc(&["x", "y", "z", "n"]);
            let f2 = self.tc(&["x", "y", "z", "n"]);
            return format!("return a.{f1}, a.{f2}");
        }
        if f < 0.85 {
            return format!("local v = a[{key}]\nreturn v, type(v), v == nil");
        }
        format!("return rawget(a, {key}), #a")
    }

    fn t_tset(&mut self) -> String {
        let f = self.r.f64();
        let key = self.tc(&[
            "b", "1", "0", "-1", "1.0", "4", "'x'", "'q'", "c", "#a + 1", "2.5",
        ]);
        if f < 0.3 {
            return format!("local t = {{}}\nt[{key}] = c\nreturn t[{key}], #t, next(t)");
        }
        if f < 0.55 {
            let mut fl = ["x", "y", "z", "w"];
            for i in (1..fl.len()).rev() {
                let j = self.r.below(i + 1);
                fl.swap(i, j);
            }
            let src = self.tc(&["a", "b", "c", "a.x", "b[1]", "c.y", "a[b]"]);
            let mut lines = vec!["local o = {}".to_string()];
            let n = self.r.range(1, 4) as usize;
            for nm in &fl[..n] {
                let k = self.tk();
                let v = self.tc(&[src, "a", "b", "c", k]);
                lines.push(format!("o.{nm} = {v}"));
            }
            let op = self.tc(&["+", "..", "<", "=="]);
            lines.push(format!("return o.x, o.y, o.z, o.w, (o.x or 0) {op} 1"));
            return lines.join("\n");
        }
        if f < 0.75 {
            let fld = self.tc(&["x", "y", "z"]);
            return format!(
                "local t = {{x = 1, y = 2}}\nt.{fld} = a\nt[{key}] = b\nreturn t.x, t.y, t[{key}]"
            );
        }
        let k = self.tc(&["1", "2", "3", "b", "4"]);
        format!("local t = {{a, b, c}}\nt[{k}] = nil\nreturn #t, t[1], t[2], t[3]")
    }

    fn t_for(&mut self) -> String {
        let bound = |g: &mut Gen| {
            let i = g.tc(INTK);
            let f = g.tc(FLK);
            g.tc(&["a", "b", "c", i, f, "a", "b", "c"])
        };
        let init = bound(self);
        let lim = bound(self);
        let step = bound(self);
        let hdr = if self.r.f64() < 0.3 {
            format!("for i = {init}, {lim} do")
        } else {
            format!("for i = {init}, {lim}, {step} do")
        };
        let body = self.tc(&[
            "acc = acc + i",
            "acc = acc .. ','",
            "acc = i",
            "fs[#fs + 1] = function() return i end",
            "acc = acc + (i // 1)",
            "if i > 3 then acc = acc + 1 end",
            "acc = math.max(acc, i)",
        ]);
        let inner = format!(
            "local n, first, last, acc, fs = 0, nil, nil, 0, {{}}\n{hdr}\n  n = n + 1\n  if first == nil then first = i end\n  last = i\n  {body}\n  if n >= 25 then break end\nend\nlocal fv = fs[1] and fs[#fs]()\nreturn n, first, last, math.type(last), acc, fv"
        );
        if self.r.f64() < 0.25 {
            let cond = self.tc(&["o == 2", "o == 5", "a == 'never'", "true"]);
            return format!(
                "local tot = 0\nfor o = 1, 3 do\n  if {cond} then\n    local r1, r2, r3 = (function()\n{inner}\n    end)()\n    tot = tot + 1\n  end\nend\nreturn tot"
            );
        }
        inner
    }

    fn t_while(&mut self) -> String {
        let f = self.r.f64();
        let op = self.tc(&CMPOPS[2..]);
        if f < 0.35 {
            let a = self.tc(&["+", "-", "*", ".."]);
            let b = self.tc(&["1", "c", "0.5", "2"]);
            return format!(
                "local n = 0\nlocal x = a\nwhile x {op} b and n < 20 do\n  x = x {a} {b}\n  n = n + 1\nend\nreturn x, n"
            );
        }
        if f < 0.65 {
            let a = self.tc(&["+", "*", "-", "//"]);
            return format!(
                "local n, x = 0, a\nrepeat\n  n = n + 1\n  x = x {a} c\n  if n > 15 then break end\nuntil x {op} b\nreturn n, x"
            );
        }
        let a = self.tc(&["1", "c", "0.25"]);
        format!(
            "local n, x = 0, a\n::top::\nn = n + 1\nif n < 10 and x {op} b then\n  x = x + {a}\n  goto top\nend\nreturn n, x"
        )
    }

    fn t_gfor(&mut self) -> String {
        let f = self.r.f64();
        if f < 0.3 {
            let x = self.tc(&["", "a[k] = nil", "if n > 2 then break end", "b = k"]);
            return format!(
                "local n, s = 0, {{}}\nfor k, v in pairs(a) do\n  n = n + 1\n  s[#s + 1] = ser(k) .. '=' .. ser(v)\n  {x}\nend\ntable.sort(s)\nreturn n, table.concat(s, ';')"
            );
        }
        if f < 0.55 {
            let x = self.tc(&["s + i", "v", "s + v", "i + b"]);
            return format!(
                "local n, s = 0, 0\nfor i, v in ipairs(a) do\n  n = n + 1\n  s = {x}\n  if n > 30 then break end\nend\nreturn n, s"
            );
        }
        if f < 0.75 {
            return "local n, k, v = 0, nil, nil\nrepeat\n  k, v = next(a, k)\n  n = n + 1\nuntil k == nil or n > 20\nreturn n".to_string();
        }
        let s1 = self.tc(&["1", "c", "0.5"]);
        let s2 = self.tc(&["0", "a", "0.0"]);
        format!(
            "local function it(s, i)\n  i = i + {s1}\n  if i <= s then return i, i * 2 end\nend\nlocal n, last = 0, nil\nfor i, j in it, b, {s2} do n = n + 1; last = j; if n > 20 then break end end\nreturn n, last"
        )
    }

    fn t_vararg(&mut self) -> String {
        self.helper_stmt("local function va(...) return select('#', ...), ... end");
        self.helper_stmt("local function vb(x, ...) local t = {...} return #t, x, ... end");
        let f = self.r.f64();
        if f < 0.25 {
            return "return va(a, b, c)".to_string();
        }
        if f < 0.45 {
            let s = self.tc(&["a", "b", "1", "-1", "2", "'#'", "4", "-4"]);
            return format!("return select({s}, a, b, c)");
        }
        if f < 0.65 {
            return "return vb(va(a, b))".to_string();
        }
        if f < 0.8 {
            let n = self.tc(&[
                "vb",
                "va",
                "math.max",
                "select",
                "string.byte",
                "table.unpack",
            ]);
            return format!("return {n}(a, b, c)");
        }
        "local x, y, z, w = va(a)\nreturn x, y, z, w".to_string()
    }

    fn t_closure(&mut self) -> String {
        let f = self.r.f64();
        if f < 0.5 {
            let op = self.tc(&["+", "..", "*", "//"]);
            return format!(
                "local fs = {{}}\nlocal u = a\nfor i = 1, 3 do\n  fs[i] = function() u = u {op} b return u end\nend\nreturn fs[1](), fs[2](), u"
            );
        }
        self.helper_stmt("local up = 0");
        let v = self.tc(&["a", "up + 1", "b"]);
        let op = self.tc(BINOPS);
        format!("up = {v}\nreturn up, (function() return up {op} c end)()")
    }

    fn t_rec(&mut self) -> String {
        let op = self.tc(&["+", "-", "*", "//", ".."]);
        self.helper_stmt(&format!(
            "local function rf(x, d) if d <= 0 then return x end return rf(x {op} 1, d - 1) end"
        ));
        self.helper_stmt(
            "local function tf(x, d) if d <= 0 then return x end return tf(x, d - 1) end",
        );
        self.tc(&[
            "return rf(a, 5)",
            "return tf(a, 10)",
            "return rf(b, 3), tf(c, 2)",
            "return math.max(rf(a, 2), b)",
        ])
        .to_string()
    }

    fn t_tail(&mut self) -> String {
        let (name, ar) = self.t_builtin_pick();
        let na = self.r.range(0, ar + 1);
        let args: Vec<&str> = (0..na).map(|_| self.tc(&["a", "b", "c"])).collect();
        format!("return {name}({})", args.join(", "))
    }

    fn t_str(&mut self) -> String {
        let k1 = self.tk();
        let k2 = self.tk();
        let i1 = self.tc(INTK);
        let i2 = self.tc(INTK);
        let opts = [
            "return a .. b".to_string(),
            "return #a".to_string(),
            "return string.sub(a, b, c)".to_string(),
            "return a:sub(b)".to_string(),
            "return string.byte(a, b, c)".to_string(),
            format!("return a .. {k1}"),
            format!("return {k2} .. a .. b"),
            "return string.len(a), #a".to_string(),
            "local s = a .. ''\nreturn s, #s".to_string(),
            "return (a .. b) == 'ab'".to_string(),
            "return string.byte(a, -1)".to_string(),
            "return a:byte(1, -1)".to_string(),
            format!("return string.sub(a, {i1}, {i2})"),
        ];
        self.r.pick(&opts).clone()
    }

    fn t_expr_tpl(&mut self) -> String {
        let d = self.r.range(2, 4) as i32;
        format!("return {}", self.t_expr(d))
    }

    fn t_coerce(&mut self) -> String {
        let b = self.tc(&["10", "16", "2", "36", "b"]);
        let k1 = self.tk();
        let k2 = self.tk();
        let k3 = self.tk();
        let opts = [
            "return a + 0, a * 1, a - 0".to_string(),
            "return a // 1, a % 1".to_string(),
            "return math.tointeger(a)".to_string(),
            "return a | 0".to_string(),
            "return tonumber(a)".to_string(),
            format!("return tonumber(a, {b})"),
            "return a == a + 0".to_string(),
            "return a + 0.0".to_string(),
            format!("return a // {k1}, a % {k2}"),
            "return math.floor(a), math.ceil(a)".to_string(),
            "local x = a\nif math.type(x) == 'integer' then return x + 1 end\nreturn x".to_string(),
            format!("return math.fmod(a, b), math.fmod(a, {k3})"),
        ];
        self.r.pick(&opts).clone()
    }

    fn t_mt(&mut self) -> String {
        let f = self.r.f64();
        if f < 0.4 {
            return "return a.x, a[1], a.zz".to_string();
        }
        if f < 0.7 {
            let op = self.tc(BINCMP);
            return format!("return a {op} b");
        }
        "return #a, -a, a(b)".to_string()
    }

    fn t_call(&mut self) -> String {
        self.helper_stmt("local function g0() return 'g0' end");
        self.helper_stmt("local function g2(x, y) return y, x end");
        self.helper_stmt("local function gv(...) return ... end");
        self.helper_stmt("local obj = { m = function(self, x) return self, x end }");
        let op = self.tc(BINCMP);
        let opts = [
            "return g2(a)".to_string(),
            "return g2(a, b, c, a)".to_string(),
            "return g0(a, b)".to_string(),
            "return gv(a, b, c)".to_string(),
            "return (gv(a, b, c))".to_string(),
            "return {gv(a, b)}, #{gv(a, b, c)}".to_string(),
            "return b(a)".to_string(),
            "return a(b, c)".to_string(),
            "return obj:m(a)".to_string(),
            "return obj.m(a, b)".to_string(),
            "return g2(gv(a, b, c))".to_string(),
            format!("local x, y = g2(a, b)\nreturn y {op} x"),
            "return select(-1, gv(a, b, c))".to_string(),
            "return select(2, gv(a, b, c)), select('#', gv())".to_string(),
            "return pcall(g2, a, b)".to_string(),
            "return pcall(a, b)".to_string(),
            "return string.format('%s', a)".to_string(),
        ];
        self.r.pick(&opts).clone()
    }

    fn t_coro(&mut self) -> String {
        let o1 = self.tc(&["+", "..", "*", "-"]);
        let o2 = self.tc(BINOPS);
        let opts = [
            format!("local co = coroutine.wrap(function(x)\n  for i = 1, 3 do x = x {o1} coroutine.yield(x) end\n  return x\nend)\nco(a); co(b); co(c)\nreturn co(b)"),
            format!("local co = coroutine.create(function(x, y)\n  local ok, v = pcall(function() return coroutine.yield(x {o2} y) end)\n  return ok, v\nend)\nlocal r1 = {{ coroutine.resume(co, a, b) }}\nlocal r2 = {{ coroutine.resume(co, c) }}\nreturn #r1, r1[2], #r2, r2[2], r2[3]"),
            "local n = 0\nlocal co = coroutine.wrap(function()\n  for i = a, b do n = n + 1; coroutine.yield(i) if n > 10 then break end end\n  return \"end\"\nend)\nlocal last\nfor j = 1, 12 do last = co() if last == \"end\" then break end end\nreturn n, last".to_string(),
        ];
        self.r.pick(&opts).clone()
    }

    fn t_meta2(&mut self) -> String {
        let op = self.tc(BINOPS);
        let opts = [
            "return a.x, a.y, a[1]".to_string(),
            "a.q = b\nreturn rawget(a, 'q'), a.q".to_string(),
            "return a == b, a ~= b, rawequal(a, b)".to_string(),
            "return a < b, a <= b".to_string(),
            "return #a, rawlen(a)".to_string(),
            "return a .. b".to_string(),
            "return -a".to_string(),
            "return a(b, c)".to_string(),
            "return a + b, b + a".to_string(),
            "return a[b], a[c]".to_string(),
            "return getmetatable(a)".to_string(),
            format!("local x = a.x\nreturn x {op} 1"),
        ];
        self.r.pick(&opts).clone()
    }

    fn t_close(&mut self) -> String {
        let v = self.tc(&["CL", "nil", "false"]);
        let op = self.tc(BINOPS);
        format!("local x <close> = {v}\nreturn a {op} b")
    }

    fn t_tarr(&mut self) -> String {
        let fill = self.tc(&[
            "t[i] = a",
            "t[#t + 1] = a",
            "t[i] = i",
            "rawset(t, i, b)",
            "table.insert(t, a)",
            "t[i * 1.0] = i",
        ]);
        let upd = self.tc(&[
            "t[b] = c",
            "t[b] = nil",
            "t[#t] = nil",
            "t[#t + 1] = c",
            "table.insert(t, 1, c)",
            "table.remove(t, b)",
            "t[b + 0.0] = c",
            "t[c] = b",
            "rawset(t, b, c)",
        ]);
        let rd = self.tc(&[
            "t[b]",
            "t[#t]",
            "t[1]",
            "rawget(t, b)",
            "t[b // 1]",
            "t[c]",
            "select('#', table.unpack(t))",
        ]);
        let n = self.tc(&["8", "3", "1", "0", "17"]);
        format!(
            "local t = {{}}\nfor i = 1, {n} do {fill} end\n{upd}\nlocal s = 0\nfor k, v in ipairs(t) do s = s + 1 if s > 40 then break end end\nreturn #t, s, {rd}, next(t)"
        )
    }

    fn t_upv(&mut self) -> String {
        let op = self.tc(&["+", "..", "*", "-"]);
        self.helper_stmt("local cnt = 0");
        self.helper_stmt("local function inc(d) cnt = cnt + d return cnt end");
        self.helper_stmt(&format!(
            "local function mkc(x) return function(y) x = x {op} y return x end end"
        ));
        self.tc(&[
            "cnt = 0\nreturn inc(a), inc(b), cnt",
            "local g = mkc(a)\nreturn g(b), g(c)",
            "local x = a\nlocal function h() return x end\nx = b\nreturn h(), x",
            "local fs = {}\nfor i = 1, 3 do local j = i + a fs[i] = function() return j, i end end\nreturn fs[1](), fs[3]()",
        ])
        .to_string()
    }

    fn t_strmeth(&mut self) -> String {
        self.tc(&[
            "return a:len(), a:upper()",
            "return (a):sub(b, c)",
            "return a:byte(b)",
            "return a:rep(2), #a:rep(3, ',')",
            "return ('x'):rep(b % 64)",
            "return a:find(b, 1, true)",
            "return tostring(a):len()",
            "return a:sub(-b)",
        ])
        .to_string()
    }

    /// Three warm-up families, mostly numeric.
    fn t_warm_fams(&mut self) -> [&'static str; 3] {
        let r = self.r.f64();
        if r < 0.55 {
            [self.tc(NUMF), self.tc(NUMF), self.tc(NUMF)]
        } else if r < 0.7 {
            [self.tc(&["tt", "tm", "tb"]), self.tc(NUMF), self.tc(NUMF)]
        } else if r < 0.85 {
            [
                self.tc(&["st", "ns"]),
                self.tc(&["sp", "si", "sn"]),
                self.tc(&["sp", "si", "sn"]),
            ]
        } else {
            [self.tc(FAMS), self.tc(FAMS), self.tc(FAMS)]
        }
    }

    /// A template: (its kind, the body of `f`, the families `f` is warmed up with).
    pub(super) fn template(
        &mut self,
        only: Option<&str>,
    ) -> (&'static str, String, Option<[&'static str; 3]>) {
        let kind = match only.and_then(|k| TEMPLATES.iter().find(|(n, _)| *n == k)) {
            Some(&(n, _)) => n,
            None => {
                let w: Vec<u32> = TEMPLATES.iter().map(|&(_, w)| w).collect();
                TEMPLATES[self.r.weighted(&w)].0
            }
        };
        let body = match kind {
            "binop" => self.t_binop(),
            "unop" => self.t_unop(),
            "cmp" => self.t_cmp(),
            "logic" => self.t_logic(),
            "builtin" => self.t_builtin(),
            "tget" => self.t_tget(),
            "tset" => self.t_tset(),
            "for" => self.t_for(),
            "while" => self.t_while(),
            "gfor" => self.t_gfor(),
            "vararg" => self.t_vararg(),
            "closure" => self.t_closure(),
            "rec" => self.t_rec(),
            "tail" => self.t_tail(),
            "str" => self.t_str(),
            "expr" => self.t_expr_tpl(),
            "coerce" => self.t_coerce(),
            "mt" => self.t_mt(),
            "call" => self.t_call(),
            "coro" => self.t_coro(),
            "meta2" => self.t_meta2(),
            "close" => self.t_close(),
            "tarr" => self.t_tarr(),
            "upv" => self.t_upv(),
            "strmeth" => self.t_strmeth(),
            "konst" => self.t_konst(),
            "tailnat" => self.t_tailnat(),
            "flow" => self.t_flow(),
            "arity" => self.t_arity(),
            "chain" => self.t_chain(),
            "recur" => self.t_recur(),
            "callslot" => self.t_callslot(),
            "hoist" => self.t_hoist(),
            _ => unreachable!(),
        };
        let mut wf = self.t_warm_fams();
        if kind == "for" && self.r.f64() < 0.8 {
            let p = &["si", "sp", "sn", "ib", "bi"];
            wf = [self.tc(p), self.tc(p), self.tc(p)];
        }
        // The calls that exit come after the warm-up: `b` a number until then.
        if kind == "callslot" {
            wf[1] = self.tc(&["si", "sp", "fl", "mn"]);
        }
        if kind == "hoist" && self.r.f64() < 0.7 {
            let i = &["si", "sp", "ib", "bi", "sn"];
            wf = [
                self.tc(i),
                self.tc(&["sp", "bi", "ib"]),
                self.tc(&["sp", "sn", "ot", "si"]),
            ];
        }
        if kind == "meta2" {
            let mut m = NUMF.to_vec();
            m.extend(["tm", "st"]);
            let mut n = NUMF.to_vec();
            n.push("tm");
            wf = [self.tc(&["tm", "tb", "tt"]), self.tc(&m), self.tc(&n)];
        }
        (kind, body, Some(wf))
    }

    /// The whole code of a `limits` unit: integer loops with bounds and steps near the limits.
    pub(super) fn limits_unit(&mut self, tag: &str) -> Block {
        let e = self.tc(LIMIT_EXPRS);
        let acc = self.tc(LIMIT_ACCS);
        let test = self.tc(LIMIT_TESTS);
        let step = format!("n = n + 1\n  local x = {e}\n  {acc}\n  if {test} then t = t + 1 end");
        let form = self.r.f64();
        let lp = if form < 0.6 {
            format!("for i = lo, hi, st do\n  {step}\n  if n >= 40 then break end\nend")
        } else if form < 0.8 {
            format!(
                "local i = lo\nwhile n < 40 and (st > 0 and i <= hi or st < 0 and i >= hi) do\n  {step}\n  i = i + st\nend"
            )
        } else {
            format!(
                "for j = 1, 12 do\n  local i = lo + (j - 1) * st\n  {step}\n  if i == hi then break end\nend"
            )
        };
        let body = format!("local s, t, n = 0, 0, 0\n{lp}\nreturn s, t, n, math.type(s)");
        let warm = self.tc(LIMIT_WARM);
        let nw = *self.r.pick(&[30, 300, 1000]);
        let mut out = format!(
            "local function f(lo, hi, st, k)\n{body}\nend\nfor w = 1, {nw} do pcall(f, {warm}) end\nrec(\"{tag}w\", pcall(f, {warm}))\n"
        );
        let np = self.r.range(5, 12);
        for p in 0..np {
            let a = [
                self.tc(LIMIT_BOUNDS),
                self.tc(LIMIT_BOUNDS),
                self.tc(LIMIT_STEPS),
                self.tc(LIMIT_KS),
            ];
            let args = a.join(", ");
            if self.r.f64() < 0.5 {
                out.push_str(&format!("rec(\"{tag}p{p}\", pcall(f, {args}))\n"));
            } else {
                out.push_str(&format!(
                    "do local ok, a, b, c, d for q = 1, 3 do ok, a, b, c, d = pcall(f, {args}) end rec(\"{tag}p{p}\", ok, a, b, c, d) end\n"
                ));
            }
        }
        out.push_str(&format!("rec(\"{tag}z\", pcall(f, {warm}))"));
        vec![Stmt::Raw(out)]
    }
}
