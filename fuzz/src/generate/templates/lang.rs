//! Templates over the language's corners, by the sections of the manual: string patterns,
//! formats, `pack`, `utf8`, lexical forms, assignment and constructor order, vararg tables,
//! `_ENV` and global declarations, `load`, to-be-closed variables, `goto`, `xpcall` and error
//! levels, coroutines, inheritance, the table library and numeric edges.

use crate::generate::Gen;

/// Templates whose `a` is a string, `b` a small integer.
pub(super) const STRINGY: &[&str] = &["pattern", "strfmt", "pack", "utf8", "lex"];

/// Subjects for patterns.
const SUBJ: &[&str] = &[
    "a",
    "a",
    "\"key = val, k2 = v2; (nested (parens)) [x]\"",
    "\"THE (quick) fox jumps over 12 lazy dogs\"",
    "\"  trim me  \"",
    "\"aaa bbb aaa\"",
    "\"x\\0y\\0z\"",
    "\"0x1F, 077, -3.5e2\"",
    "(a .. a)",
];
const PATS: &[&str] = &[
    "%a+",
    "(%w+)=(%w+)",
    "(%w+)%s*=%s*(%w+)",
    "^%s*(.-)%s*$",
    "^%s*",
    "%b()",
    "%b[]",
    "%f[%a]%a+",
    "%f[%A]",
    "()",
    "()a()",
    "(%d+)",
    "[%a_][%w_]*",
    "[^%s]+",
    ".-",
    "(.)%1",
    "%x+",
    "[%]%-]",
    "%p",
    "^(%a)(.*)$",
    "(h)(e)(l)(l)(o)",
    "%u",
    "%c",
    "[a-c]+",
    "[^a-c]",
    "a*",
    "a-b",
    "a?",
    "%.",
    "$",
    "^",
    "%",
    "(",
    "[a",
    "%b",
    "%f",
    "(()",
    "%1",
];
const REPLS: &[&str] = &[
    "\"%1-%0\"",
    "\"%%\"",
    "\"<%2>\"",
    "\"\"",
    "\"%0%0\"",
    "{a = \"A\", key = 1, x = false, the = \"t\"}",
    "string.upper",
    "function(m) return nil end",
    "function(m, n) return n end",
    "function(m) return #m end",
    "function(m) return {} end",
    "function(...) return select(\"#\", ...) end",
];

const FMTS: &[&str] = &[
    "%d", "%5d", "%-5d", "%+d", "% d", "%05d", "%x", "%X", "%#x", "%o", "%c", "%s", "%10s",
    "%-10s", "%.2s", "%q", "%i", "%u", "%%", "%a", "%.3f", "%g", "%e", "%10.4g", "%5.1f", "%.0f",
    "%#o", "%.20s", "%100d", "%-+5d", "%.3d",
];

const PACKF: &[&str] = &[
    "i4", "<i4", ">i8", "=I2", "b", "B", "h", "H", "j", "J", "T", "d", "n", "s1", "s2", "z", "x",
    "!4 i1 i4", "<!8 b d", "i3", "I7", "i16", "c3", ">s4", "<I1 I1", "!2 h b h", "Xi4", "i0", "c",
    "f", "<j z", "!x",
];

const UTFS: &[&str] = &[
    "a",
    "\"h\\u{e4}llo\"",
    "\"\\xff\\xfe\"",
    "\"\\u{7FFFFFFF}\"",
    "\"\\xe4\"",
    "\"a\\u{10FFFF}b\"",
    "\"\\u{D800}\"",
    "\"\\u{20AC}\\u{20AC}\"",
    "\"\\xc0\\x80\"",
    "\"\"",
];

/// Lexical forms, each a whole expression.
const LEXEMES: &[&str] = &[
    "[==[a]]b]==]",
    "[[\nthe first newline goes]]",
    "[=[\n]=]",
    "\"\\z\n      x\"",
    "\"\\x41\\x62\"",
    "\"\\u{48}\\u{20AC}\\u{7FFFFFFF}\"",
    "\"\\65\\066\\0067\"",
    "'\\''",
    "\"\\\\\"",
    "\"a\\\nb\"",
    "\"\\a\\b\\f\\v\\r\\t\"",
    "0x1p4",
    "0x.8p1",
    "0xA.8",
    "1e-2",
    ".5",
    "5.",
    "3e0",
    "0xffffffffffffffff",
    "0x10000000000000000",
    "9007199254740993",
    "9223372036854775807",
    "9223372036854775808",
    "-9223372036854775808",
    "1E+2",
    "0X1P-2",
    "--[==[ a\ncomment ]==] 7",
    "#[[\n\n]]",
    "type\"x\"",
    "string.len\"abc\"",
    "rawlen{1, 2, nil}",
    "(\"x\"):rep(2)",
    "(\"%d\"):format(3):rep(2)",
    "({f = function(s, x) return x end}):f\"y\"",
    "select('#', table.unpack{1, 2, 3})",
    "#{n = 1}",
    "(\"a\"):byte()",
    "- - -2",
    "not not nil",
    "2^-1",
    "-2^2",
    "2^3^2",
    "1 .. 2",
    "\"a\" .. 1 .. 2",
    "7 // 2 * 2",
    "1 < 2 == true",
    "\"10\" + 1",
    "\"0x10\" * 1",
    "3 | 1 ~ 2 & 7",
    "1 << 2 + 1",
    "~5 >> 1",
];

const VA_HELPERS: &[&str] = &[
    "local function va1(...t) return t.n, t[1], select('#', ...) end",
    "local function va2(x, ...t) t[1] = x return ... end",
    "local function va3(...t) t.n = 1 return ... end",
    "local function va4(...t) t.n = 4 return ... end",
    "local function va5(...t) local g = function() return t.n, t[2] end return g(), ... end",
    "local function va6(...t) return table.unpack(t, 1, t.n) end",
    "local function va7(...t) t.n = \"x\" return ... end",
    "local function va8(...t) for i = 1, t.n do t[i] = t[i] == nil end return ... end",
    "local function va9(...) return select(-1, ...) end",
    "local function va10(...t) t.n = t.n - 1 return select('#', ...), ... end",
    "local function va11(...t) local u = t u[2] = 'w' return ... end",
    "local function va12(a, ...t) return va1(...), va2(a, ...) end",
];

const CLOSER: &str = "local function closer(tag)\n  return setmetatable({}, {__close = function(o, e) probe(tag .. \":\" .. (e == nil and \"ok\" or type(e) == \"string\" and e:gsub(\"^[^:]*:%d+: \", \"\") or type(e))) end})\nend";

/// What `xpcall` runs, and what handles its errors.
const XFS: &[&str] = &[
    "function(x, y) return x + y end",
    "function(x) error(x) end",
    "function(x) error(x, 2) end",
    "function(x) error(x, 0) end",
    "function() error() end",
    "function(x) error(setmetatable({}, {__tostring = function() return \"TS\" end})) end",
    "function(x) local t = nil return t.x end",
    "function(x) return pcall(error, x) end",
    "error",
    "function(x, y) return select('#', x, y), y end",
    "function(x) return xpcall(error, function(e) return \"in:\" .. type(e) end, x) end",
    "function(x) local ok, e = pcall(error, {x}) error(e) end",
];
const XHS: &[&str] = &[
    "function(e) return \"H:\" .. type(e) end",
    "function(e) return e end",
    "function(e) return e, 2 end",
    "type",
    "function(e) return select('#', e) end",
    "function(e) return type(e) == \"table\" and e[1] or e end",
];

const NUMEDGE: &[&str] = &[
    "math.modf(a)",
    "math.fmod(a, b)",
    "math.ult(a, b)",
    "math.tointeger(a)",
    "math.abs(math.mininteger)",
    "math.mininteger // -1",
    "math.mininteger % -1",
    "math.maxinteger + 1 == math.mininteger",
    "a // 0.0",
    "-0.0 == 0",
    "1 / (-0.0)",
    "2^53 == 2^53 + 1",
    "math.maxinteger + 0.0 == math.maxinteger",
    "math.maxinteger < math.maxinteger + 0.0",
    "math.mininteger == -2^63",
    "a < b",
    "a == b + 0.0",
    "a <= b * 1.0",
    "\"0x10\" | 0",
    "\"1e2\" // 1",
    "\" 10 \" + 0",
    "a % -3",
    "-7 // 2",
    "-7 % 2",
    "7 % -2.0",
    "-7 % math.huge",
    "7 % -math.huge",
    "math.floor(-0.5)",
    "math.ceil(-0.5)",
    "math.huge // 1",
    "math.floor(math.huge)",
    "3 // 0",
    "3 % 0",
    "1 << 64",
    "1 << -1",
    "-1 >> 1",
    "-1 >> 63",
    "~0",
    "5 // 0.0",
    "-5 // 0.0",
    "0/0 ~= 0/0",
    "math.tointeger(\"8\")",
    "math.tointeger(2^63)",
    "math.tointeger(-2^63)",
    "a // b",
    "a % b",
    "a ^ b",
    "a | b",
    "b >> a",
    "math.max(a, b, -0.0)",
    "math.min(0.0, -0.0)",
    "math.type(a + b)",
    "(a + 0.0) // 1 == a",
    "math.abs(a)",
    "-a",
    "9007199254740993 < 9007199254740992.0",
    "9007199254740993 == 2^53 + 1",
    "math.ult(math.maxinteger, math.mininteger)",
    "string.format(\"%d\", 3.0)",
    "string.format(\"%x\", a)",
    "tostring(a // 1 | 0)",
    "math.fmod(math.mininteger, -1)",
    "math.fmod(5, 0)",
    "math.fmod(5.0, 0)",
];

impl Gen {
    fn subj(&mut self) -> &'static str {
        self.tc(SUBJ)
    }

    /// `find`, `match`, `gmatch`, `gsub`, with captures, frontiers, balances and bad patterns.
    pub(super) fn t_pattern(&mut self) -> String {
        let s = self.subj();
        let p = format!("{:?}", self.tc(PATS));
        let init = self.tc(&["1", "b", "-3", "0", "100", "b + 1"]);
        match self.r.below(7) {
            0 => format!("return string.find({s}, {p})"),
            1 => format!("return string.find({s}, {p}, {init})"),
            2 => format!("return string.find({s}, {p}, {init}, true)"),
            3 => format!("return string.match({s}, {p}, {init})"),
            4 => format!(
                "local r = {{}}\nfor x, y in string.gmatch({s}, {p}) do\n  r[#r + 1] = tostring(x) .. \"|\" .. tostring(y)\n  if #r > 20 then break end\nend\nreturn #r, table.concat(r, \",\")"
            ),
            5 => {
                let repl = self.tc(REPLS);
                let n = self.tc(&["", ", b", ", 1", ", 0", ", -1"]);
                format!("return string.gsub({s}, {p}, {repl}{n})")
            }
            _ => {
                let repl = self.tc(REPLS);
                format!("local x = ({s}):gsub({p}, {repl})\nreturn x, ({s}):match({p})")
            }
        }
    }

    /// `string.format` with flags, widths and precisions; `rep`, `byte`, `char`, comparisons.
    pub(super) fn t_strfmt(&mut self) -> String {
        let n = self.r.range(1, 3);
        let mut fmt = String::new();
        let mut args = Vec::new();
        for _ in 0..n {
            fmt.push_str(self.tc(FMTS));
            fmt.push_str(self.tc(&["", " ", "|", "x"]));
            args.push(self.tc(&["a", "b", "c", "65", "-1", "2.25", "\"q\\n\\0\"", "255"]));
        }
        let call = format!("string.format({fmt:?}, {})", args.join(", "));
        match self.r.below(6) {
            0 => format!("return {call}"),
            1 => format!("return pcall(string.format, {fmt:?}, {})", args.join(", ")),
            2 => self
                .tc(&[
                    "return string.rep(a, 3, \",\"), string.rep(a, 0, \"x\"), string.rep(\"\", 5)",
                    "return a:rep(2, \"\\0\"), #a:rep(3, a)",
                    "return string.char(string.byte(a, 1, -1))",
                    "return string.byte(a, b, c)",
                    "return pcall(string.char, b, 256)",
                ])
                .to_string(),
            3 => self
                .tc(&[
                    "return a < \"a\\0\", \"a\\0b\" < \"a\\0c\", \"\" < a, a <= a .. \"\\0\"",
                    "return a:upper(), a:lower(), a:reverse(), #a",
                    "return (\"%s|%s\"):format(a, b)",
                    "return string.format(\"%q\", b), string.format(\"%q\", math.mininteger)",
                ])
                .to_string(),
            4 => format!("local s = {call}\nreturn #s, s:byte(1, 3)"),
            _ => format!("return select('#', {call}), {call} == {call}"),
        }
    }

    /// `string.pack`, `unpack`, `packsize`: sizes, endianness, alignment, round trips.
    pub(super) fn t_pack(&mut self) -> String {
        let n = self.r.range(1, 3);
        let mut f = String::new();
        let mut vals = Vec::new();
        for _ in 0..n {
            f.push_str(self.tc(PACKF));
            f.push(' ');
            vals.push(self.tc(&[
                "a", "b", "c", "-1", "255", "65536", "2.5", "\"str\"", "1 << 40",
            ]));
        }
        let f = format!("{:?}", f.trim_end());
        let v = vals.join(", ");
        match self.r.below(5) {
            0 => format!("local s = string.pack({f}, {v})\nreturn #s, string.unpack({f}, s)"),
            1 => format!("return pcall(string.pack, {f}, {v})"),
            2 => format!("return pcall(string.packsize, {f})"),
            3 => format!("return pcall(string.unpack, {f}, a, b)"),
            _ => format!(
                "local ok, s = pcall(string.pack, {f}, {v})\nif not ok then return s end\nreturn string.unpack({f}, s, #s // 2 + 1)"
            ),
        }
    }

    /// The `utf8` library on valid and invalid sequences.
    pub(super) fn t_utf8(&mut self) -> String {
        let s = self.tc(UTFS);
        let lax = self.tc(&["", ", true"]);
        match self.r.below(7) {
            0 => "return pcall(utf8.char, b, c, 0x10FFFF, 0x7FFFFFFF)".to_string(),
            1 => format!("return utf8.len({s}, 1, -1{lax})"),
            2 => format!("return pcall(utf8.codepoint, {s}, 1, -1{lax})"),
            3 => format!(
                "local r = {{}}\nlocal ok, e = pcall(function()\n  for p, c in utf8.codes({s}{lax}) do r[#r + 1] = p .. \":\" .. c end\nend)\nreturn ok, table.concat(r, \",\"), e"
            ),
            4 => format!(
                "return pcall(utf8.offset, {s}, b, {})",
                self.tc(&["1", "-1", "2", "b"])
            ),
            5 => format!("return ({s}):match(utf8.charpattern), #utf8.charpattern"),
            _ => format!("return utf8.len({s}, b), utf8.len(utf8.char(72, 228, 8364))"),
        }
    }

    /// Lexical forms: long brackets, escapes, numerals, call sugar, precedence.
    pub(super) fn t_lex(&mut self) -> String {
        let n = self.r.range(1, 3) as usize;
        let mut es = Vec::new();
        for _ in 0..n {
            es.push(self.tc(LEXEMES).to_string());
        }
        let sep = self.tc(&["", ";", ";;"]);
        match self.r.below(3) {
            0 => format!("return {}", es.join(", ")),
            1 => format!("local x = {}{sep}\nreturn x, a", es[0]),
            _ => format!(
                "::top::{sep}\nlocal t = {{{}}}\nreturn t[1], t[#t]",
                es.join("; ")
            ),
        }
    }

    /// Multiple assignment where the order of the stores shows.
    pub(super) fn t_assign(&mut self) -> String {
        self.helper_stmt("local function logged(log)\n  return setmetatable({}, {__newindex = function(t, k, v) log[#log + 1] = sk(k) .. \"=\" .. sk(v) rawset(t, k, v) end})\nend");
        self.tc(&[
            "local t, i = {}, 1\ni, t[i] = i + 1, 20\nreturn i, t[1], t[2]",
            "local log = {}\nlocal p = logged(log)\np.x, p.y, p[a] = b, c, a\nreturn table.concat(log, \" \")",
            "local x, y = a, b\nx, y = y, x\nreturn x, y",
            "local t = {}\nt[1], t[1] = 1, 2\nreturn t[1]",
            "local n = 0\nlocal function k() n = n + 1 return n end\nlocal t = {}\nt[k()], t[k()] = k(), k()\nreturn t[1], t[2], t[3], t[4]",
            "local a1, a1 = 1, 2\nreturn a1",
            "local t = {1, 2, 3}\nlocal i = 1\ni, t[i], t[i + 1] = 3, i, t[i]\nreturn i, t[1], t[2], t[3]",
            "local log = {}\nlocal p = logged(log)\np[1], p[2], p[1] = a, b, c\nreturn table.concat(log, \" \"), rawget(p, 1)",
            "local x, y, z = (function() return a, b, c end)()\nx, y, z = z, x\nreturn x, y, z",
        ])
        .to_string()
    }

    /// Table constructors: item kinds, keys that normalize, expansion at the end or not.
    pub(super) fn t_ctor(&mut self) -> String {
        self.helper_stmt("local function m3() return 1, 2, 3 end");
        let n = self.r.range(2, 7);
        let items: Vec<&str> = (0..n)
            .map(|_| {
                self.tc(&[
                    "a",
                    "nil",
                    "m3()",
                    "(m3())",
                    "[1] = \"one\"",
                    "[1.0] = \"onef\"",
                    "[2^53] = 1",
                    "[-0.0] = \"z\"",
                    "x = b",
                    "[\"x\"] = c",
                    "[a] = 1",
                    "[#\"ab\"] = 2",
                    "[3] = m3()",
                    "b",
                    "{}",
                ])
            })
            .collect();
        let sep = self.tc(&[", ", "; "]);
        let trail = self.tc(&["", ",", ";"]);
        format!(
            "local t = {{{}{trail}}}\nreturn t[1], t[2], t[3], t[4], t[0], t.x, t[2^53], select('#', table.unpack(t, 1, 5))",
            items.join(sep)
        )
    }

    /// Lua 5.5 vararg tables: named, written to, captured.
    pub(super) fn t_vatab(&mut self) -> String {
        for h in VA_HELPERS {
            self.helper_stmt(h);
        }
        let f = self.r.range(1, VA_HELPERS.len() as i64);
        let g = self.r.range(1, VA_HELPERS.len() as i64);
        self.tc(&[
            "return pcall(vaF, a, b, c)",
            "return pcall(vaF), pcall(vaG, a)",
            "return pcall(vaF, table.unpack({a, b, c}, 1, 3))",
            "return pcall(vaF, nil, nil, a, nil)",
            "return pcall(function(...) return vaF(...), vaG(...) end, a, b)",
            "local ok, x, y = pcall(vaF, vaG(a, b))\nreturn ok, x, y",
        ])
        .replace("vaF", &format!("va{f}"))
        .replace("vaG", &format!("va{g}"))
    }

    /// `_ENV` as a local, as a parameter, and global declarations.
    pub(super) fn t_env(&mut self) -> String {
        self.helper_stmt("local function inenv(_ENV) return x, y end");
        self.tc(&[
            "local _ENV = {x = a, y = b}\nreturn x, y, z",
            "return inenv({x = a, y = b}), inenv(setmetatable({}, {__index = function(t, k) return k end}))",
            "local f = load(\"global gx = ...\\nreturn gx\", \"=c\", \"t\", {gx = a})\nreturn pcall(f, b)",
            "local f = load(\"global gx = ...\\nreturn gx\", \"=c\", \"t\", {})\nreturn pcall(f, b)",
            "local f = load(\"global<const> *\\nY = 1\")\nreturn f == nil",
            "local f = load(\"global X\\nglobal<const> *\\nX = ...\\nreturn X, type(math)\", \"=c\", \"t\", {math = math})\nreturn f(a)",
            "local e = {}\nlocal f = load(\"x = ...\\nreturn x\", \"=c\", \"t\", e)\nreturn f(a), e.x",
            "local function g()\n  local _ENV <const> = {ga = 1}\n  return ga\nend\nreturn g(), ga",
            "do\n  global G1\n  G1 = a\nend\nreturn G1",
            "local _ENV = setmetatable({}, {__index = _ENV, __newindex = function(t, k, v) rawset(t, k, sk(v)) end})\nq = a\nreturn q, type(q)",
        ])
        .to_string()
    }

    /// `load`: chunks from text and from reader functions, varargs, environments, errors.
    pub(super) fn t_load(&mut self) -> String {
        self.helper_stmt("local function reader(parts)\n  local i = 0\n  return function() i = i + 1 return parts[i] end\nend");
        self.tc(&[
            "return load(\"return ...\", \"=c\")(a, b)",
            "return load(reader({\"return \", \"1 + \", \"...\", \"\"}))(b)",
            "return pcall(load(reader({\"return \", \"(...)\", nil, \"x\"})), a)",
            "local f, e = load(reader({\"return 1\", 2}))\nreturn f == nil, e ~= nil",
            "local f, e = load(\"return +\")\nreturn f == nil, type(e)",
            "local f = load(\"local x = ... return function() x = x + 1 return x end\")\nreturn pcall(f(b))",
            "return pcall(load(\"local v = ... return v + {}\", \"=chunk\"), a)",
            "return pcall(load(\"return x, y\", \"=c\", \"t\", {x = a, y = b}))",
            "return load(\"return select('#', ...)\")(a, nil, b, nil)",
            "local f = load(function() return nil end)\nreturn pcall(f)",
            "return pcall(load, \"return 1\", \"=c\", \"t\", a)",
            "local src = \"return \" .. string.rep(\"(\", 50) .. \"1\" .. string.rep(\")\", 50)\nreturn load(src)()",
            "local f = load(\"error('in chunk', 1)\", \"=named\")\nreturn pcall(f)",
            "return load(\"return function(...) return ... end\")()(a, b, c)",
        ])
        .to_string()
    }

    /// To-be-closed variables left by every way out of a scope.
    pub(super) fn t_tbc(&mut self) -> String {
        self.helper_stmt(CLOSER);
        self.helper_stmt("local function upto(s, c) if c < s then return c + 1 end end");
        self.tc(&[
            "do\n  local x <close> = closer(\"a\")\n  local y <close> = closer(\"b\")\n  if b == 1 then error(\"E\", 0) end\nend\nreturn 1",
            "for i = 1, 3 do\n  local x <close> = closer(\"L\" .. i)\n  if i == 2 then break end\nend\nreturn a",
            "do\n  local x <close> = closer(\"g\")\n  goto out\nend\n::out::\nreturn b",
            "local x <close> = closer(\"r\")\nlocal y <close> = nil\nreturn a, b",
            "local x <close> = setmetatable({}, {__close = function() error(\"inclose\", 0) end})\nlocal y <close> = closer(\"y\")\nreturn a",
            "local ok, e = pcall(function() local x <close> = a end)\nreturn ok, e",
            "local n = 0\nfor i in upto, 3, 0, closer(\"for\") do\n  n = n + i\n  if i == b then break end\nend\nreturn n",
            "local co = coroutine.create(function()\n  local x <close> = closer(\"co\")\n  coroutine.yield(1)\nend)\ncoroutine.resume(co)\nreturn coroutine.close(co), coroutine.status(co)",
            "local co = coroutine.create(function()\n  local x <close> = setmetatable({}, {__close = function() error(\"cl\", 0) end})\n  coroutine.yield(1)\nend)\ncoroutine.resume(co)\nreturn coroutine.close(co)",
            "local ok, e = pcall(function()\n  local x <close> = closer(\"p\")\n  local y <close> = closer(\"q\")\n  error({}, 0)\nend)\nreturn ok, type(e)",
            "local function f()\n  local x <close> = closer(\"t\")\n  return a, b\nend\nreturn f()",
            "local r = {}\nfor i = 1, 2 do\n  local x <close> = closer(\"i\" .. i)\n  if i == 1 then goto cont end\n  r[#r + 1] = i\n  ::cont::\nend\nreturn #r",
            "local x <close> = closer(\"outer\")\ndo\n  local y <close> = closer(\"inner\")\nend\nreturn (function() local z <close> = closer(\"fn\") return 3 end)()",
        ])
        .to_string()
    }

    /// Loops and exits built from `goto`.
    pub(super) fn t_goto(&mut self) -> String {
        self.tc(&[
            "local i, s = 1, 0\n::top::\nif i <= 10 then s = s + i i = i + 1 goto top end\nreturn s, i",
            "local s = 0\nfor i = 1, 6 do\n  if i % 3 == b then goto cont end\n  s = s + i\n  ::cont::\nend\nreturn s",
            "local fs = {}\nlocal i = 1\n::l::\nlocal j = i\nfs[i] = function() return j end\ni = i + 1\nif i <= 3 then goto l end\nreturn fs[1](), fs[2](), fs[3]()",
            "local hit\nfor i = 1, 3 do\n  for j = 1, 3 do\n    if i * j == b then hit = i .. \",\" .. j goto done end\n  end\nend\n::done::\nreturn hit",
            "local n = 0\nrepeat\n  local x = n\n  n = n + 1\n  if x < 2 then goto skip end\n  n = n + 10\n  ::skip::\nuntil x > 3\nreturn n",
            "do goto a end\n::b:: do return 'b' end\n::a:: goto b",
            "local n = 0\nwhile true do\n  n = n + 1\n  if n > 3 then goto out end\nend\n::out::\nreturn n",
            "local t = {}\nfor i = 1, 3 do\n  local v = i * 2\n  t[i] = function() return v end\n  if i == 2 then goto next end\n  v = v + 100\n  ::next::\nend\nreturn t[1](), t[2](), t[3]()",
        ])
        .to_string()
    }

    /// `xpcall` with handlers, error values that are no strings, error levels.
    pub(super) fn t_xpcall(&mut self) -> String {
        self.helper_stmt("local function lv(n, x) error(x, n) end");
        self.helper_stmt("local function lv2(n, x) lv(n, x) end");
        let f = self.tc(XFS);
        let h = self.tc(XHS);
        match self.r.below(5) {
            0 => format!("return xpcall({f}, {h}, a, b)"),
            1 => format!("return pcall(xpcall, {f}, {h}, a)"),
            2 => format!("return xpcall(xpcall, {h}, {f}, {h}, a)"),
            3 => format!(
                "return pcall(lv2, {}, {})",
                self.tc(&["0", "1", "2", "3", "4"]),
                self.tc(&["\"m\"", "a", "{}", "nil"])
            ),
            _ => format!("local ok, e = pcall({f}, a, b)\nreturn ok, type(e), ok or ({h})(e)"),
        }
    }

    /// Coroutines: generators, yields through calls and metamethods, `close`, status.
    pub(super) fn t_coro2(&mut self) -> String {
        self.tc(&[
            "local gen = coroutine.wrap(function(n) for i = 1, n do coroutine.yield(i, i * 2) end return \"end\" end)\nlocal r = {gen(3)}\nreturn r[1], r[2], gen(), gen(), gen()",
            "local co = coroutine.create(function(...) local x = coroutine.yield(...) return select('#', x), x end)\nlocal r1 = table.pack(coroutine.resume(co, a, nil, b))\nlocal r2 = table.pack(coroutine.resume(co))\nreturn r1.n, r1[2], r2.n, r2[2], coroutine.resume(co)",
            "local co\nco = coroutine.create(function() return coroutine.resume(co) end)\nreturn coroutine.resume(co)",
            "local inner\nlocal outer = coroutine.create(function() return coroutine.resume(inner) end)\ninner = coroutine.create(function() return coroutine.status(outer), coroutine.isyieldable() end)\nreturn coroutine.resume(outer)",
            "local co = coroutine.wrap(function() local ok, v = pcall(function() return coroutine.yield(1) + 1 end) return ok, v end)\nco()\nreturn co(a)",
            "local t = setmetatable({}, {__index = function(t, k) return coroutine.yield(k) end})\nlocal co = coroutine.wrap(function() return t.x, t.y end)\nreturn co(), co(1), co(2)",
            "local co = coroutine.wrap(function() table.sort({3, 2, 1}, function(x, y) coroutine.yield() return x < y end) return \"sorted\" end)\nlocal r\nfor i = 1, 20 do r = co() if r then break end end\nreturn r",
            "local co = coroutine.wrap(function() return (\"abc\"):gsub(\".\", function(c) return coroutine.yield(c) end) end)\nreturn co(), co(\"x\"), co(\"y\"), co(\"z\")",
            "local co = coroutine.create(function() error({a}) end)\nlocal ok, e = coroutine.resume(co)\nreturn ok, type(e), coroutine.status(co), coroutine.close(co)",
            "local co = coroutine.create(function() coroutine.yield() end)\ncoroutine.resume(co)\nreturn coroutine.close(co), coroutine.status(co), coroutine.resume(co)",
            "local w = coroutine.wrap(function() error(\"in wrap\") end)\nreturn pcall(w)",
            "return coroutine.isyieldable(), coroutine.running() ~= nil, select(2, coroutine.running())",
            "local co = coroutine.wrap(function(...) local n = select('#', ...) while true do n = n + select('#', coroutine.yield(n)) end end)\nreturn co(a, b), co(), co(nil, nil), co(a)",
        ])
        .to_string()
    }

    /// A class hierarchy through `__index` tables and functions, and metamethods, which are
    /// not inherited.
    pub(super) fn t_oop(&mut self) -> String {
        let depth = self.r.range(1, 4);
        let mut pre = String::from(
            "local K0 = {}\nK0.__index = K0\nfunction K0.new(x) return setmetatable({x = x}, K0) end\nfunction K0:get() return self.x end\nfunction K0:name() return \"K0\" end\nK0.__tostring = function(o) return \"obj:\" .. sk(o.x) end\nK0.__eq = function(p, q) return rawequal(p.x, q.x) end",
        );
        for k in 1..=depth {
            let p = k - 1;
            let link = if self.r.chance(0.3) {
                format!("setmetatable(K{k}, {{__index = function(t, n) return K{p}[n] end}})")
            } else {
                format!("setmetatable(K{k}, {{__index = K{p}}})")
            };
            pre.push_str(&format!(
                "\nlocal K{k} = {{}}\n{link}\nK{k}.__index = K{k}\nfunction K{k}.new(x) local o = K{p}.new(x) return setmetatable(o, K{k}) end\nfunction K{k}:name() return \"K{k}>\" .. K{p}.name(self) end"
            ));
            let extra = *self.r.pick(&[
                "",
                "K{k}.__add = function(p, q) return K{k}.new((p.x or 0) + (type(q) == \"table\" and q.x or q)) end",
                "K{k}.__len = function(o) return {k} end",
                "K{k}.__call = function(o, ...) return select('#', ...), o.x end",
                "K{k}.__lt = function(p, q) return p.x < q.x end",
                "K{k}.__metatable = \"locked\"",
                "K{k}.__concat = function(p, q) return \"cat\" end",
                "K{k}.__unm = function(o) return -o.x end",
                "K{k}.__tostring = K{p}.__tostring",
                "function K{k}:get() return \"sub\", K{p}.get(self) end",
            ]);
            pre.push('\n');
            pre.push_str(
                &extra
                    .replace("{k}", &k.to_string())
                    .replace("{p}", &p.to_string()),
            );
        }
        self.helper_stmt(&pre);
        let c = format!("K{}", self.r.range(0, depth));
        self.tc(&[
            "local o = C.new(a)\nreturn o:get(), o:name()",
            "local o = C.new(a)\nreturn tostring(o) == tostring(C.new(a)), getmetatable(o) == C",
            "local o = C.new(a)\nreturn pcall(function() return o + b end)",
            "local o = C.new(a)\nreturn pcall(function() return #o, o(b, c) end)",
            "local o = C.new(a)\nreturn o == C.new(a), o == C.new(b), o ~= o",
            "local o, p = C.new(a), C.new(b)\nreturn pcall(function() return o < p, o .. p, -o end)",
            "local o = C.new(a)\nreturn pcall(setmetatable, o, {}), getmetatable(o) == C",
            "local o = C.new(a)\nreturn rawget(o, \"name\"), rawget(C, \"name\") ~= nil, o.missing",
            "return pcall(function() return C.new(a):name():upper() end)",
        ])
        .replace('C', &c)
    }

    /// The table library at its edges.
    pub(super) fn t_tablib(&mut self) -> String {
        self.tc(&[
            "local t = {5, 3, 9, 1, 7, -2, 8}\ntable.sort(t, function(x, y) return x > y end)\nreturn table.concat(t, \",\")",
            "local t = {\"b\", \"a\", \"B\", \"\", \"a\\0\", \"ab\"}\ntable.sort(t)\nreturn table.concat(t, \"|\")",
            "local t = {}\nfor i = 1, 40 do t[i] = (i * 37) % 41 end\ntable.sort(t)\nreturn t[1], t[20], t[40]",
            "local t = {{k = 3}, {k = 1}, {k = 2}}\ntable.sort(t, function(x, y) return x.k < y.k end)\nreturn t[1].k, t[2].k, t[3].k",
            "local t = {1, 2, 3, 4, 5}\ntable.move(t, 1, 3, 2)\nreturn table.concat(t, \",\")",
            "local t = {1, 2, 3, 4, 5}\ntable.move(t, 2, 5, 1)\nreturn table.concat(t, \",\")",
            "return table.concat(table.move({1, 2, 3}, 1, 3, 3, {9, 9}), \",\")",
            "return pcall(table.move, {1, 2, 3}, 1, 3, b)",
            "return pcall(table.concat, {1, \"x\", 3}, \"-\", 1, b)",
            "return pcall(table.concat, {1, {}, 3})",
            "return table.concat({}, \"x\"), table.concat({\"a\"}, \"x\", 2, 1)",
            "local t = {1, 2, 3}\nreturn pcall(table.insert, t, b, \"x\"), table.concat(t, \",\")",
            "local t = {1, 2, 3}\nreturn pcall(table.insert, t, 5, \"x\")",
            "local t = {1, 2, 3}\nreturn table.remove(t, 1), table.remove(t), table.remove(t), table.remove(t), table.remove(t), #t",
            "return pcall(table.remove, {1, 2}, b)",
            "local t = {}\nreturn table.remove(t, 0), table.remove(t, #t + 1), #t",
            "return table.unpack({1, 2, 3}, -1, 1)",
            "return select('#', table.unpack({}, 1, 0)), pcall(table.unpack, {}, 1, 1 << 40)",
            "local p = table.pack(a, nil, b, nil)\nreturn p.n, p[1], p[3], select('#', table.unpack(p, 1, p.n))",
            "return pcall(table.insert, {}, 1, 2, 3)",
        ])
        .to_string()
    }

    /// Numbers at their edges: overflow, conversions, signs of `//` and `%`, coercions.
    pub(super) fn t_numedge(&mut self) -> String {
        let n = self.r.range(1, 2);
        let es: Vec<&str> = (0..n).map(|_| self.tc(NUMEDGE)).collect();
        match self.r.below(3) {
            0 => format!("return {}", es.join(", ")),
            1 => format!(
                "return pcall(function(a, b) return {} end, a, b)",
                es.join(", ")
            ),
            _ => format!("local x = {}\nreturn x, math.type(x)", es[0]),
        }
    }
}
