//! Hand-written templates: small bodies of `f` that aim at one thing each.

mod lang;
mod stress;
mod strict;

use super::driver::{FAMS, NUMF};
use super::{Build, Gen};
use crate::ast::{Block, Stmt};

pub(super) struct Template {
    pub name: &'static str,
    /// How often a unit takes it.
    weight: u32,
    /// The body of `f`.
    body: Build,
    /// Its units stay within what reference Lua agrees on, so are compared with it: no float or
    /// table reaches a string, no length of a table that may have holes, no libm function whose
    /// last bit may differ, no `pairs` order.
    pub strict: bool,
    /// Its call is slow, so the driver makes fewer.
    pub heavy: bool,
}

const fn t(name: &'static str, weight: u32, body: Build) -> Template {
    Template {
        name,
        weight,
        body,
        strict: false,
        heavy: false,
    }
}

/// A strict template.
const fn s(name: &'static str, weight: u32, body: Build) -> Template {
    Template {
        strict: true,
        ..t(name, weight, body)
    }
}

const fn heavy(t: Template) -> Template {
    Template { heavy: true, ..t }
}

/// Constants by class: a template takes a class first, so a rare class is not rare by count.
const KONST: &[&[&str]] = &[
    &["0", "1", "-1", "2", "3", "8", "63", "64", "-64", "65", "-3"],
    &[
        "2147483648",
        "2147483647",
        "-2147483648",
        "1099511627776",
        "9223372036854775807",
        "math.mininteger",
        "0x7fffffff",
    ],
    &["0.0", "-0.0", "1.0", "0.5", "-2.0", "4.0", "2^53", "1e300"],
    &["\"10\"", "\"abc\"", "\"0x10\""],
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

/// The templates (all but `limits`).
pub(super) const TEMPLATES: &[Template] = &[
    t("binop", 6, Gen::t_binop),
    t("unop", 2, Gen::t_unop),
    t("cmp", 5, Gen::t_cmp),
    t("logic", 1, Gen::t_logic),
    t("builtin", 6, Gen::t_builtin),
    t("tget", 3, Gen::t_tget),
    t("tset", 3, Gen::t_tset),
    t("for", 5, Gen::t_for),
    t("while", 2, Gen::t_while),
    t("gfor", 2, Gen::t_gfor),
    t("vararg", 2, Gen::t_vararg),
    t("closure", 1, Gen::t_closure),
    t("rec", 1, Gen::t_rec),
    t("tail", 2, Gen::t_tail),
    t("str", 2, Gen::t_str),
    t("expr", 4, Gen::t_expr_tpl),
    t("coerce", 3, Gen::t_coerce),
    t("mt", 1, Gen::t_mt),
    t("call", 3, Gen::t_call),
    t("coro", 2, Gen::t_coro),
    t("meta2", 3, Gen::t_meta2),
    t("close", 1, Gen::t_close),
    t("tarr", 4, Gen::t_tarr),
    t("upv", 2, Gen::t_upv),
    t("strmeth", 1, Gen::t_strmeth),
    s("konst", 6, Gen::t_konst),
    s("tailnat", 6, Gen::t_tailnat),
    s("flow", 6, Gen::t_flow),
    s("arity", 4, Gen::t_arity),
    heavy(s("chain", 3, Gen::t_chain)),
    heavy(s("recur", 3, Gen::t_recur)),
    t("callslot", 3, Gen::t_callslot),
    s("hoist", 4, Gen::t_hoist),
    s("pattern", 4, Gen::t_pattern),
    s("strfmt", 3, Gen::t_strfmt),
    s("pack", 2, Gen::t_pack),
    s("utf8", 2, Gen::t_utf8),
    s("lex", 3, Gen::t_lex),
    // Not strict: the manual leaves the order of assignment undefined.
    t("assign", 2, Gen::t_assign),
    t("ctor", 3, Gen::t_ctor),
    s("vatab", 3, Gen::t_vatab),
    s("env", 2, Gen::t_env),
    s("load", 2, Gen::t_load),
    s("tbc", 3, Gen::t_tbc),
    s("goto", 2, Gen::t_goto),
    s("xpcall", 3, Gen::t_xpcall),
    // Not strict: tcvm yields across natives on purpose.
    t("coro2", 3, Gen::t_coro2),
    s("oop", 3, Gen::t_oop),
    s("tablib", 3, Gen::t_tablib),
    s("numedge", 4, Gen::t_numedge),
];

impl Gen {
    fn tc(&mut self, l: &[&'static str]) -> &'static str {
        self.r.pick(l)
    }

    /// A class, then a member of it.
    fn tclass(&mut self, classes: &[&[&'static str]]) -> &'static str {
        let class = *self.r.pick(classes);
        self.tc(class)
    }

    fn tk(&mut self) -> &'static str {
        self.tclass(KONST)
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
        match self.r.below(4) {
            0 => format!("return a {op} b"),
            1 => format!("return a {op} {}", self.tk()),
            2 => format!("return {} {op} a", self.tk()),
            _ => {
                let op2 = self.tc(BINOPS);
                let o = self.toperand();
                format!("local x = a {op} b\nreturn x {op2} {o}, math.type(x)")
            }
        }
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
        match self.r.below(4) {
            0 => format!("return {lhs} {op} {rhs}"),
            1 => format!("if {lhs} {op} {rhs} then return 1 else return 2 end"),
            2 => {
                let b = self.tc(&BINOPS[..7]);
                let k = self.tk();
                format!("local x = a {b} b\nif x {op} {k} then return x end\nreturn not x")
            }
            _ => format!("if not ({lhs} {op} {rhs}) then return 'n' end return 'y'"),
        }
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
        match self.r.below(6) {
            0 => format!("return {call}"),
            1 => format!("local x = {call}\nreturn x"),
            2 => format!("return select('#', {call}), {call}"),
            3 => format!("return {{{call}}}"),
            4 => format!("return {call}, 'tail'"),
            _ => format!("local t = {{{call}, {call}}}\nreturn #t, t[1], t[2]"),
        }
    }

    fn t_tget(&mut self) -> String {
        let key = self.tc(&[
            "b", "1", "0", "-1", "1.0", "2", "3", "4", "2^53", "'x'", "'y'", "'n'", "1.5", "c",
            "b + 1", "#a",
        ]);
        match self.r.below(4) {
            0 => format!("return a[{key}]"),
            1 => {
                let f1 = self.tc(&["x", "y", "z", "n"]);
                let f2 = self.tc(&["x", "y", "z", "n"]);
                format!("return a.{f1}, a.{f2}")
            }
            2 => format!("local v = a[{key}]\nreturn v, type(v), v == nil"),
            _ => format!("return rawget(a, {key}), #a"),
        }
    }

    fn t_tset(&mut self) -> String {
        let key = self.tc(&[
            "b", "1", "0", "-1", "1.0", "4", "'x'", "'q'", "c", "#a + 1", "2.5",
        ]);
        match self.r.below(4) {
            0 => format!("local t = {{}}\nt[{key}] = c\nreturn t[{key}], #t, next(t)"),
            1 => {
                let mut fl = ["x", "y", "z", "w"];
                self.r.shuffle(&mut fl);
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
                lines.join("\n")
            }
            2 => {
                let fld = self.tc(&["x", "y", "z"]);
                format!(
                    "local t = {{x = 1, y = 2}}\nt.{fld} = a\nt[{key}] = b\nreturn t.x, t.y, t[{key}]"
                )
            }
            _ => {
                let k = self.tc(&["1", "2", "3", "b", "4"]);
                format!("local t = {{a, b, c}}\nt[{k}] = nil\nreturn #t, t[1], t[2], t[3]")
            }
        }
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
        let op = self.tc(&CMPOPS[2..]);
        match self.r.below(3) {
            0 => {
                let a = self.tc(&["+", "-", "*", ".."]);
                let b = self.tc(&["1", "c", "0.5", "2"]);
                format!(
                    "local n = 0\nlocal x = a\nwhile x {op} b and n < 20 do\n  x = x {a} {b}\n  n = n + 1\nend\nreturn x, n"
                )
            }
            1 => {
                let a = self.tc(&["+", "*", "-", "//"]);
                format!(
                    "local n, x = 0, a\nrepeat\n  n = n + 1\n  x = x {a} c\n  if n > 15 then break end\nuntil x {op} b\nreturn n, x"
                )
            }
            _ => {
                let a = self.tc(&["1", "c", "0.25"]);
                format!(
                    "local n, x = 0, a\n::top::\nn = n + 1\nif n < 10 and x {op} b then\n  x = x + {a}\n  goto top\nend\nreturn n, x"
                )
            }
        }
    }

    fn t_gfor(&mut self) -> String {
        match self.r.below(4) {
            0 => {
                let x = self.tc(&["", "a[k] = nil", "if n > 2 then break end", "b = k"]);
                format!(
                    "local n, s = 0, {{}}\nfor k, v in pairs(a) do\n  n = n + 1\n  s[#s + 1] = ser(k) .. '=' .. ser(v)\n  {x}\nend\ntable.sort(s)\nreturn n, table.concat(s, ';')"
                )
            }
            1 => {
                let x = self.tc(&["s + i", "v", "s + v", "i + b"]);
                format!(
                    "local n, s = 0, 0\nfor i, v in ipairs(a) do\n  n = n + 1\n  s = {x}\n  if n > 30 then break end\nend\nreturn n, s"
                )
            }
            2 => "local n, k, v = 0, nil, nil\nrepeat\n  k, v = next(a, k)\n  n = n + 1\nuntil k == nil or n > 20\nreturn n".to_string(),
            _ => {
                let s1 = self.tc(&["1", "c", "0.5"]);
                let s2 = self.tc(&["0", "a", "0.0"]);
                format!(
                    "local function it(s, i)\n  i = i + {s1}\n  if i <= s then return i, i * 2 end\nend\nlocal n, last = 0, nil\nfor i, j in it, b, {s2} do n = n + 1; last = j; if n > 20 then break end end\nreturn n, last"
                )
            }
        }
    }

    fn t_vararg(&mut self) -> String {
        self.helper_stmt("local function va(...) return select('#', ...), ... end");
        self.helper_stmt("local function vb(x, ...) local t = {...} return #t, x, ... end");
        match self.r.below(5) {
            0 => "return va(a, b, c)".to_string(),
            1 => {
                let s = self.tc(&["a", "b", "1", "-1", "2", "'#'", "4", "-4"]);
                format!("return select({s}, a, b, c)")
            }
            2 => "return vb(va(a, b))".to_string(),
            3 => {
                let n = self.tc(&[
                    "vb",
                    "va",
                    "math.max",
                    "select",
                    "string.byte",
                    "table.unpack",
                ]);
                format!("return {n}(a, b, c)")
            }
            _ => "local x, y, z, w = va(a)\nreturn x, y, z, w".to_string(),
        }
    }

    fn t_closure(&mut self) -> String {
        if self.r.chance(0.5) {
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
        const FIXED: &[&str] = &[
            "return a .. b",
            "return #a",
            "return string.sub(a, b, c)",
            "return a:sub(b)",
            "return string.byte(a, b, c)",
            "return string.len(a), #a",
            "local s = a .. ''\nreturn s, #s",
            "return (a .. b) == 'ab'",
            "return string.byte(a, -1)",
            "return a:byte(1, -1)",
        ];
        match self.r.below(FIXED.len() + 3) {
            0 => format!("return a .. {}", self.tk()),
            1 => format!("return {} .. a .. b", self.tk()),
            2 => {
                let i = self.tc(INTK);
                let j = self.tc(INTK);
                format!("return string.sub(a, {i}, {j})")
            }
            n => FIXED[n - 3].to_string(),
        }
    }

    fn t_expr_tpl(&mut self) -> String {
        let d = self.r.range(2, 4) as i32;
        format!("return {}", self.t_expr(d))
    }

    fn t_coerce(&mut self) -> String {
        const FIXED: &[&str] = &[
            "return a + 0, a * 1, a - 0",
            "return a // 1, a % 1",
            "return math.tointeger(a)",
            "return a | 0",
            "return tonumber(a)",
            "return a == a + 0",
            "return a + 0.0",
            "return math.floor(a), math.ceil(a)",
            "local x = a\nif math.type(x) == 'integer' then return x + 1 end\nreturn x",
        ];
        match self.r.below(FIXED.len() + 3) {
            0 => {
                let b = self.tc(&["10", "16", "2", "36", "b"]);
                format!("return tonumber(a, {b})")
            }
            1 => {
                let k1 = self.tk();
                let k2 = self.tk();
                format!("return a // {k1}, a % {k2}")
            }
            2 => format!("return math.fmod(a, b), math.fmod(a, {})", self.tk()),
            n => FIXED[n - 3].to_string(),
        }
    }

    fn t_mt(&mut self) -> String {
        match self.r.below(3) {
            0 => "return a.x, a[1], a.zz".to_string(),
            1 => format!("return a {} b", self.tc(BINCMP)),
            _ => "return #a, -a, a(b)".to_string(),
        }
    }

    fn t_call(&mut self) -> String {
        self.helper_stmt("local function g0() return 'g0' end");
        self.helper_stmt("local function g2(x, y) return y, x end");
        self.helper_stmt("local function gv(...) return ... end");
        self.helper_stmt("local obj = { m = function(self, x) return self, x end }");
        const FIXED: &[&str] = &[
            "return g2(a)",
            "return g2(a, b, c, a)",
            "return g0(a, b)",
            "return gv(a, b, c)",
            "return (gv(a, b, c))",
            "return {gv(a, b)}, #{gv(a, b, c)}",
            "return b(a)",
            "return a(b, c)",
            "return obj:m(a)",
            "return obj.m(a, b)",
            "return g2(gv(a, b, c))",
            "return select(-1, gv(a, b, c))",
            "return select(2, gv(a, b, c)), select('#', gv())",
            "return pcall(g2, a, b)",
            "return pcall(a, b)",
            "return string.format('%s', a)",
        ];
        match self.r.below(FIXED.len() + 1) {
            0 => format!("local x, y = g2(a, b)\nreturn y {} x", self.tc(BINCMP)),
            n => FIXED[n - 1].to_string(),
        }
    }

    fn t_coro(&mut self) -> String {
        match self.r.below(3) {
            0 => {
                let o1 = self.tc(&["+", "..", "*", "-"]);
                format!("local co = coroutine.wrap(function(x)\n  for i = 1, 3 do x = x {o1} coroutine.yield(x) end\n  return x\nend)\nco(a); co(b); co(c)\nreturn co(b)")
            }
            1 => {
                let o2 = self.tc(BINOPS);
                format!("local co = coroutine.create(function(x, y)\n  local ok, v = pcall(function() return coroutine.yield(x {o2} y) end)\n  return ok, v\nend)\nlocal r1 = {{ coroutine.resume(co, a, b) }}\nlocal r2 = {{ coroutine.resume(co, c) }}\nreturn #r1, r1[2], #r2, r2[2], r2[3]")
            }
            _ => "local n = 0\nlocal co = coroutine.wrap(function()\n  for i = a, b do n = n + 1; coroutine.yield(i) if n > 10 then break end end\n  return \"end\"\nend)\nlocal last\nfor j = 1, 12 do last = co() if last == \"end\" then break end end\nreturn n, last".to_string(),
        }
    }

    fn t_meta2(&mut self) -> String {
        const FIXED: &[&str] = &[
            "return a.x, a.y, a[1]",
            "a.q = b\nreturn rawget(a, 'q'), a.q",
            "return a == b, a ~= b, rawequal(a, b)",
            "return a < b, a <= b",
            "return #a, rawlen(a)",
            "return a .. b",
            "return -a",
            "return a(b, c)",
            "return a + b, b + a",
            "return a[b], a[c]",
            "return getmetatable(a)",
        ];
        match self.r.below(FIXED.len() + 1) {
            0 => format!("local x = a.x\nreturn x {} 1", self.tc(BINOPS)),
            n => FIXED[n - 1].to_string(),
        }
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

    /// A template, `only` if it names one: (it, the body of `f`, the families `f` is warmed up
    /// with).
    pub(super) fn template(
        &mut self,
        only: Option<&str>,
    ) -> (&'static Template, String, [&'static str; 3]) {
        let t = match only.and_then(|k| TEMPLATES.iter().find(|t| t.name == k)) {
            Some(t) => t,
            None => self.r.pick_by(TEMPLATES, |t| t.weight),
        };
        let (kind, body) = (t.name, (t.body)(self));
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
        if lang::STRINGY.contains(&kind) {
            wf = [
                self.tc(&["st", "st", "ns"]),
                self.tc(&["si", "sp", "sn"]),
                self.tc(&["st", "si"]),
            ];
        }
        if kind == "meta2" {
            let mut m = NUMF.to_vec();
            m.extend(["tm", "st"]);
            let mut n = NUMF.to_vec();
            n.push("tm");
            wf = [self.tc(&["tm", "tb", "tt"]), self.tc(&m), self.tc(&n)];
        }
        (t, body, wf)
    }

    /// The whole code of a `limits` unit: integer loops with bounds and steps near the limits.
    pub(super) fn limits_unit(&mut self, tag: &str) -> Block {
        let e = self.tc(LIMIT_EXPRS);
        let acc = self.tc(LIMIT_ACCS);
        let test = self.tc(LIMIT_TESTS);
        let step = format!("n = n + 1\n  local x = {e}\n  {acc}\n  if {test} then t = t + 1 end");
        let lp = match self.r.below(3) {
            0 => format!("for i = lo, hi, st do\n  {step}\n  if n >= 40 then break end\nend"),
            1 => format!(
                "local i = lo\nwhile n < 40 and (st > 0 and i <= hi or st < 0 and i >= hi) do\n  {step}\n  i = i + st\nend"
            ),
            _ => format!(
                "for j = 1, 12 do\n  local i = lo + (j - 1) * st\n  {step}\n  if i == hi then break end\nend"
            ),
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
