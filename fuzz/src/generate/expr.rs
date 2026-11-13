//! Expressions by the kind of value they produce.

use super::{Gen, Kind, Role, pool};
use crate::ast::{Expr, Item, bin, call, name, raw, str, un};

const INTS: &[i64] = &[
    0,
    1,
    -1,
    2,
    3,
    7,
    8,
    16,
    63,
    64,
    65,
    100,
    255,
    256,
    -5,
    -64,
    2147483647,
    2147483648,
    -2147483648,
    -2147483649,
    4294967296,
    1 << 40,
    1 << 53,
    1 << 62,
    i64::MAX,
    i64::MIN,
];
const SMALL: &[i64] = &[0, 1, 2, 3, 4, 5, 7, 8, 10, -1, -2];
const FLOATS: &[f64] = &[
    0.0,
    -0.0,
    0.5,
    1.0,
    -1.0,
    1.5,
    -2.5,
    2.0,
    3.0,
    0.1,
    100.25,
    1e15,
    9007199254740992.0,
    9223372036854775808.0,
    -9223372036854775808.0,
    1e300,
    1e308,
    f64::INFINITY,
    f64::NEG_INFINITY,
    f64::NAN,
    2147483648.0,
    4294967296.0,
];
const STRS: &[&str] = &[
    "",
    "a",
    "abc",
    "hello world",
    "x\0y",
    "10",
    "0x10",
    " 3 ",
    "1e2",
    "-7",
    "2.5",
    "10a",
    "B",
    "%d",
    "nil",
    "x",
];
const ARITH: &[&str] = &[
    "+", "-", "*", "/", "//", "%", "^", "&", "|", "~", "<<", ">>",
];
const INT_OPS: &[&str] = &["+", "-", "*", "//", "%", "&", "|", "~", "<<", ">>"];
const FLOAT_OPS: &[&str] = &["+", "-", "*", "/", "^", "//", "%"];
const CMP: &[&str] = &["==", "~=", "<", "<=", ">", ">="];
const FIELDS: &[&str] = &["x", "y", "z", "w"];
const TYPES: &[&str] = &["nil", "number", "string", "table", "function", "boolean"];

impl Gen {
    pub(super) fn field(&mut self) -> String {
        self.r.pick(FIELDS).to_string()
    }

    pub(super) fn small_int(&mut self) -> Expr {
        Expr::Int(*self.r.pick(SMALL))
    }

    fn int_lit(&mut self) -> Expr {
        if self.r.chance(0.5) {
            self.small_int()
        } else {
            Expr::Int(*self.r.pick(INTS))
        }
    }

    fn str_lit(&mut self) -> Expr {
        str(self.r.pick(STRS))
    }

    fn var_of(&mut self, k: Kind) -> Option<Expr> {
        self.var(|v| v.kind == k).map(|(n, _)| name(&n))
    }

    /// An argument of `f`, or a literal where none is in sight.
    fn arg(&mut self) -> Expr {
        match self.var_of(Kind::Arg) {
            Some(e) => e,
            None => self.leaf(),
        }
    }

    fn leaf(&mut self) -> Expr {
        match self.r.below(12) {
            0..=2 => self.int_lit(),
            3 | 4 => Expr::Float(*self.r.pick(FLOATS)),
            5 | 6 => self.str_lit(),
            7 => self.r.pick(&[Expr::True, Expr::False, Expr::Nil]).clone(),
            _ => self.small_int(),
        }
    }

    /// An argument `f` is warmed up with values of `role` for; without one, a value of that
    /// sort. Now and then any argument, whatever it is warmed up with.
    fn arg_for(&mut self, role: Role) -> Expr {
        if self.risky() {
            return self.arg();
        }
        let fit: Vec<&str> = (["a", "b", "c"].into_iter().zip(self.roles))
            .filter(|&(p, r)| (r == role || role == Role::Num && r == Role::Int) && self.sees(p))
            .map(|(p, _)| p)
            .collect();
        if !fit.is_empty() {
            return name(self.r.pick(&fit));
        }
        match role {
            Role::Num => {
                match self.var(|v| matches!(v.kind, Kind::Int | Kind::Float | Kind::Num)) {
                    Some((n, _)) => name(&n),
                    None => self.int_lit(),
                }
            }
            Role::Int => self.var_of(Kind::Int).unwrap_or_else(|| self.int_lit()),
            Role::Str => self.var_of(Kind::Str).unwrap_or_else(|| self.str_lit()),
            Role::Tab => match self.var(|v| matches!(v.kind, Kind::Rec)) {
                Some((n, _)) => name(&n),
                None => raw(format!("V.tb[{}]", self.r.range(1, pool::size("tb")))),
            },
            Role::Func | Role::Mixed => raw(format!("V.fn[{}]", self.r.range(1, pool::size("fn")))),
        }
    }

    /// An operand of arithmetic: most often a number.
    pub(super) fn num(&mut self, d: u32) -> Expr {
        match self.r.below(if d == 0 { 11 } else { 14 }) {
            0..=3 => self.arg_for(Role::Num),
            4 | 5 => {
                let found = self.var(|v| matches!(v.kind, Kind::Int | Kind::Float | Kind::Num));
                found
                    .map(|(n, _)| name(&n))
                    .unwrap_or_else(|| self.int_lit())
            }
            6 | 10 if !self.risky() => self.arg_for(Role::Num),
            6 => match self.var(|v| v.kind == Kind::Any) {
                Some((n, _)) => name(&n),
                None => self.arg_for(Role::Num),
            },
            7 | 8 => self.int_lit(),
            9 => Expr::Float(*self.r.pick(FLOATS)),
            10 => self.any(0),
            _ => {
                let k = *self.r.pick(&[Kind::Int, Kind::Float, Kind::Num]);
                self.e(k, d - 1)
            }
        }
    }

    /// An operand of a string operation: most often a string.
    pub(super) fn text(&mut self, d: u32) -> Expr {
        match self.r.below(if d == 0 { 8 } else { 10 }) {
            0..=2 => self.arg_for(Role::Str),
            3 | 4 => self.var_of(Kind::Str).unwrap_or_else(|| self.str_lit()),
            5 | 6 => self.str_lit(),
            7 if self.risky() => self.any(0),
            7 => self.arg_for(Role::Str),
            _ => self.e(Kind::Str, d - 1),
        }
    }

    /// A string of bounded length: a literal, an argument, or a number's text.
    pub(super) fn fresh(&mut self, d: u32) -> Expr {
        match self.r.below(5) {
            0 | 1 => self.str_lit(),
            // An argument is of the pool: bounded, whatever came before.
            2 => match ["a", "b", "c"]
                .into_iter()
                .zip(self.roles)
                .find(|&(p, r)| r == Role::Str && self.sees(p))
            {
                Some((p, _)) if !self.risky() => name(p),
                Some(_) => self.arg(),
                None => self.str_lit(),
            },
            3 => self.int_lit(),
            _ => call("tostring", vec![self.e(Kind::Int, d.min(1))]),
        }
    }

    /// An operand a string may be built from: in a strict unit one whose text is the same on
    /// tcvm and reference Lua.
    fn text_operand(&mut self, d: u32) -> Expr {
        if self.strict {
            match self.r.below(4) {
                0 => self.e(Kind::Str, d),
                1 => self.e(Kind::Int, d),
                _ => self.arg_for(Role::Str),
            }
        } else {
            self.text(d)
        }
    }

    fn concat(&mut self, d: u32) -> Expr {
        let mut e = self.text_operand(d);
        for _ in 0..self.r.range(1, 2) {
            let f = self.fresh(d);
            e = if self.r.chance(0.8) {
                bin("..", e, f)
            } else {
                bin("..", f, e)
            };
        }
        e
    }

    /// A value to index a sequence with.
    fn seq_index(&mut self, s: &Expr, d: u32) -> Expr {
        match self.r.below(4) {
            0 => un("#", s.clone()),
            1 if d > 0 => self.e(Kind::Int, d - 1),
            _ => Expr::Int(self.r.range(1, 4)),
        }
    }

    fn seq_of(&mut self, k: Kind, d: u32) -> Expr {
        let n = self.r.range(0, 4);
        let d = d.saturating_sub(1);
        let items = (0..n).map(|_| Item::Pos(self.element(k, d))).collect();
        Expr::Table(items)
    }

    /// An expression of kind `k`, at most `d` operations deep.
    pub(super) fn e(&mut self, k: Kind, d: u32) -> Expr {
        let leaf = d == 0;
        let d1 = d.saturating_sub(1);
        match k {
            Kind::Arg => self.arg(),
            Kind::Any => self.any(d),
            Kind::Bool => match self.r.below(if leaf { 2 } else { 10 }) {
                0 => self.r.pick(&[Expr::True, Expr::False]).clone(),
                1 => self.var_of(Kind::Bool).unwrap_or(Expr::True),
                2 | 3 => {
                    let k =
                        *self
                            .r
                            .pick(&[Kind::Int, Kind::Int, Kind::Num, Kind::Float, Kind::Str]);
                    let (a, b) = (self.e(k, d1), self.e(k, d1));
                    bin(self.r.pick(CMP), a, b)
                }
                4 => {
                    let (a, b) = if self.r.chance(0.7) {
                        (self.num(d1), self.num(d1))
                    } else {
                        (self.text(d1), self.text(d1))
                    };
                    bin(self.r.pick(CMP), a, b)
                }
                5 => un("not", self.any(d1)),
                6 => {
                    let t = str(self.r.pick(TYPES));
                    bin("==", call("type", vec![self.any(d1)]), t)
                }
                7 => {
                    let t = str(self.r.pick(&["integer", "float"]));
                    bin("==", call("math.type", vec![self.any(d1)]), t)
                }
                8 => call("rawequal", vec![self.any(d1), self.any(d1)]),
                // Values of any two kinds: equality never raises.
                _ => bin(self.r.pick(&["==", "~="]), self.any(d1), self.any(d1)),
            },
            Kind::Int => match self.r.below(if leaf { 3 } else { 12 }) {
                0 | 1 => self.int_lit(),
                2 => self.var_of(Kind::Int).unwrap_or_else(|| self.small_int()),
                3..=6 => {
                    let op = *self.r.pick(INT_OPS);
                    let a = self.e(Kind::Int, d1);
                    // Most divisors are not zero.
                    let b = if matches!(op, "//" | "%") && self.r.chance(0.8) {
                        Expr::Int(*self.r.pick(&[1, 2, 3, 7, -1, -3, 16]))
                    } else {
                        self.e(Kind::Int, d1)
                    };
                    bin(op, a, b)
                }
                7 => un(self.r.pick(&["-", "~"]), self.e(Kind::Int, d1)),
                8 => {
                    let k = *self
                        .r
                        .pick(&[Kind::Str, Kind::SeqInt, Kind::SeqStr, Kind::SeqAny]);
                    un("#", self.e(k, d1))
                }
                9 => {
                    let f = *self.r.pick(&["math.max", "math.min", "math.abs"]);
                    let n = if f == "math.abs" {
                        1
                    } else {
                        self.r.range(1, 3)
                    };
                    call(f, (0..n).map(|_| self.e(Kind::Int, d1)).collect())
                }
                10 => {
                    let n = self.r.range(0, 3);
                    let mut args = vec![str("#")];
                    args.extend((0..n).map(|_| self.any(d1)));
                    call("select", args)
                }
                _ => call("string.len", vec![self.e(Kind::Str, d1)]),
            },
            Kind::Float => match self.r.below(if leaf { 2 } else { 9 }) {
                0 => Expr::Float(*self.r.pick(FLOATS)),
                1 => self
                    .var_of(Kind::Float)
                    .unwrap_or(Expr::Float(*self.r.pick(FLOATS))),
                2..=4 => {
                    let (ka, kb) = *self.r.pick(&[
                        (Kind::Float, Kind::Float),
                        (Kind::Int, Kind::Float),
                        (Kind::Float, Kind::Int),
                    ]);
                    let (a, b) = (self.e(ka, d1), self.e(kb, d1));
                    bin(self.r.pick(FLOAT_OPS), a, b)
                }
                5 => {
                    let (a, b) = (self.e(Kind::Int, d1), self.e(Kind::Int, d1));
                    bin(self.r.pick(&["/", "^"]), a, b)
                }
                6 => un("-", self.e(Kind::Float, d1)),
                7 => {
                    let f = *self.r.pick(&[
                        "math.sqrt",
                        "math.sin",
                        "math.cos",
                        "math.exp",
                        "math.log",
                    ]);
                    call(f, vec![self.e(Kind::Num, d1)])
                }
                _ => bin("+", self.e(Kind::Int, d1), Expr::Float(0.0)),
            },
            Kind::Num => match self.r.below(if leaf { 2 } else { 7 }) {
                0 => self.e(Kind::Int, d),
                1 => self.e(Kind::Float, d),
                2 => self.var_of(Kind::Num).unwrap_or_else(|| self.int_lit()),
                3 => {
                    let f = *self.r.pick(&["math.floor", "math.ceil", "math.abs"]);
                    call(f, vec![self.e(Kind::Float, d1)])
                }
                4 => {
                    let f = *self.r.pick(&["math.max", "math.min", "math.fmod"]);
                    call(f, vec![self.e(Kind::Num, d1), self.e(Kind::Num, d1)])
                }
                5 => {
                    let (a, b) = (self.e(Kind::Num, d1), self.e(Kind::Num, d1));
                    bin(self.r.pick(FLOAT_OPS), a, b)
                }
                _ => {
                    let s = str(self.r.pick(&["10", "0x10", "1e2", "-7", "2.5", " 3 "]));
                    bin(self.r.pick(&["+", "*", "-"]), s, self.e(Kind::Int, d1))
                }
            },
            Kind::Str => match self.r.below(if leaf { 3 } else { 12 }) {
                0 | 1 => self.str_lit(),
                2 => self.var_of(Kind::Str).unwrap_or_else(|| self.str_lit()),
                3 | 4 => self.concat(d1),
                5 => {
                    let s = self.e(Kind::Str, d1);
                    let (i, j) = (self.e(Kind::Int, d1.min(1)), self.e(Kind::Int, d1.min(1)));
                    if self.r.chance(0.5) {
                        call("string.sub", vec![s, i, j])
                    } else {
                        Expr::Method(Box::new(s), "sub".into(), vec![i, j])
                    }
                }
                6 => {
                    let m = *self.r.pick(&["upper", "lower", "reverse"]);
                    Expr::Method(Box::new(self.e(Kind::Str, d1)), m.into(), vec![])
                }
                7 => {
                    let n = Expr::Int(self.r.range(0, 3));
                    let mut args = vec![self.fresh(d1), n];
                    if self.r.chance(0.3) {
                        args.push(str(","));
                    }
                    call("string.rep", args)
                }
                8 => {
                    let k = *self.r.pick(&[Kind::Int, Kind::Bool, Kind::Str, Kind::Int]);
                    call("tostring", vec![self.e(k, d1)])
                }
                9 => call("type", vec![self.any(d1)]),
                10 => self.format(d1),
                _ => {
                    let k = *self.r.pick(&[Kind::SeqInt, Kind::SeqStr]);
                    let mut args = vec![self.e(k, d1)];
                    if self.r.chance(0.5) {
                        args.push(str(self.r.pick(&[",", "", "--"])));
                    }
                    call("table.concat", args)
                }
            },
            Kind::SeqInt | Kind::SeqStr | Kind::SeqAny => match self.var_of(k) {
                Some(v) if self.r.chance(0.6) => v,
                _ => self.seq_of(k, d),
            },
            Kind::Rec => match self.var_of(k) {
                Some(v) if self.r.chance(0.6) => v,
                _ => {
                    let mut fields: Vec<&str> = FIELDS.to_vec();
                    let n = self.r.range(0, 4) as usize;
                    let mut items = Vec::new();
                    for _ in 0..n {
                        let f = fields.swap_remove(self.r.below(fields.len()));
                        items.push(Item::Named(f.into(), self.any(d1)));
                    }
                    if self.r.chance(0.2) {
                        items.push(Item::Pos(self.any(d1)));
                    }
                    Expr::Table(items)
                }
            },
            Kind::Func => match self.var_of(k) {
                Some(v) if self.r.chance(0.5) => v,
                _ => match self.helper() {
                    Some((h, _)) => name(&h),
                    None => raw(format!("V.fn[{}]", self.r.range(1, pool::size("fn")))),
                },
            },
        }
    }

    fn format(&mut self, d: u32) -> Expr {
        let mut fmt = String::new();
        let mut args = Vec::new();
        let mut built = false;
        for _ in 0..self.r.range(1, 3) {
            let (spec, k) = *self.r.pick(&[
                ("%d", Kind::Int),
                ("%5d", Kind::Int),
                ("%x", Kind::Int),
                ("%s", Kind::Str),
                ("%s", Kind::Int),
                ("%q", Kind::Str),
                ("%q", Kind::Int),
                ("%g", Kind::Num),
                ("%.3f", Kind::Num),
                ("%e", Kind::Float),
                ("%a", Kind::Float),
                ("%5.1f", Kind::Num),
                ("%i", Kind::Int),
                ("%%", Kind::Any),
            ]);
            fmt.push_str(spec);
            match (k, spec) {
                (Kind::Any, _) => {}
                // `%q` escapes: a string fed its own quoted text doubles each time.
                (Kind::Str, "%q") => args.push(self.fresh(d)),
                // One operand may have been built before, the rest are bounded.
                (Kind::Str, _) if built => args.push(self.fresh(d)),
                (Kind::Str, _) => {
                    built = true;
                    args.push(self.e(Kind::Str, d))
                }
                _ => args.push(self.e(k, d)),
            }
            fmt.push_str(self.r.pick(&["", " ", ",", "-"]));
        }
        let mut all = vec![str(&fmt)];
        all.extend(args);
        call("string.format", all)
    }

    /// A table to read from or store into, and what it is.
    pub(super) fn container(&mut self, d: u32) -> (Expr, Kind) {
        if self.r.chance(0.35) {
            let t = self.arg_for(Role::Tab);
            let k = if matches!(&t, Expr::Name(n) if !["a", "b", "c"].contains(&n.as_str())) {
                Kind::Rec
            } else {
                Kind::Arg
            };
            return (t, k);
        }
        let any = self.risky();
        let found = self.var(|v| v.kind == Kind::Rec || v.kind.seq() || any && v.kind == Kind::Any);
        match found {
            Some((n, k)) if self.r.chance(0.85) => (name(&n), k),
            _ => {
                let k = *self.r.pick(&[Kind::Rec, Kind::SeqInt, Kind::SeqAny]);
                (self.e(k, d), k)
            }
        }
    }

    pub(super) fn key(&mut self, d: u32) -> Expr {
        match self.r.below(6) {
            0 => str(self.r.pick(FIELDS)),
            1 => Expr::Int(self.r.range(-1, 4)),
            2 => Expr::Float(*self.r.pick(&[1.0, 2.0, 2.5])),
            3 => self.arg(),
            _ => self.any(d),
        }
    }

    fn index(&mut self, d: u32) -> Expr {
        let (t, k) = self.container(d);
        match k {
            Kind::Rec => Expr::Field(Box::new(t), self.field()),
            k if k.seq() => {
                let i = self.seq_index(&t, d);
                Expr::Index(Box::new(t), Box::new(i))
            }
            _ if self.r.chance(0.35) => Expr::Field(
                Box::new(t),
                self.r.pick(&["x", "y", "z", "n", "zz"]).to_string(),
            ),
            _ => {
                let key = self.key(d);
                Expr::Index(Box::new(t), Box::new(key))
            }
        }
    }

    pub(super) fn args(&mut self, n: usize, d: u32) -> Vec<Expr> {
        (0..n).map(|_| self.any(d)).collect()
    }

    /// A call whose results are anything: of a function of the unit, of the pool, of a
    /// variable, or of a builtin that takes any value.
    pub(super) fn call_any(&mut self, d: u32) -> Expr {
        match self.r.below(10) {
            0..=2 => {
                let Some((f, n)) = self.helper() else {
                    return self.call_any_builtin(d);
                };
                let n = if self.r.chance(0.8) {
                    n
                } else {
                    self.r.below(4)
                };
                call(&f, self.args(n, d))
            }
            3 => {
                let f = format!("V.fn[{}]", self.r.range(1, pool::size("fn")));
                let n = self.r.below(3);
                call(&f, self.args(n, d))
            }
            // Never a variable that may come to hold a function defined later: no recursion.
            4 => {
                let f = match self.var_of(Kind::Func) {
                    Some(f) if self.r.chance(0.5) => f,
                    _ => self.arg_for(Role::Func),
                };
                let n = self.r.below(3);
                Expr::Call(Box::new(f), self.args(n, d))
            }
            5 => {
                let o = self.text(d);
                let (m, n) = match self.risky() {
                    true => ("m", 1),
                    false => *self.r.pick(&[
                        ("len", 0),
                        ("upper", 0),
                        ("sub", 2),
                        ("byte", 1),
                        ("sub", 1),
                    ]),
                };
                let args = (0..n).map(|_| self.num(d)).collect();
                Expr::Method(Box::new(o), m.into(), args)
            }
            _ => self.call_any_builtin(d),
        }
    }

    fn call_any_builtin(&mut self, d: u32) -> Expr {
        // N a number, S a string, T a table, A anything.
        let (f, sig) = *self.r.pick(&[
            ("math.max", "NNn"),
            ("math.min", "NNn"),
            ("math.abs", "N"),
            ("math.floor", "N"),
            ("math.ceil", "N"),
            ("math.fmod", "NN"),
            ("math.tointeger", "A"),
            ("math.ult", "NN"),
            ("math.type", "A"),
            ("math.sqrt", "N"),
            ("tonumber", "A"),
            ("tonumber", "SB"),
            ("rawget", "TA"),
            ("rawequal", "AA"),
            ("type", "A"),
            ("string.byte", "Snn"),
            ("string.sub", "SNn"),
            ("string.len", "S"),
            ("getmetatable", "A"),
            ("select", "IAaa"),
            ("ipairs", "T"),
            ("assert", "Aaa"),
        ]);
        let mut args = Vec::new();
        for c in sig.chars() {
            // A lowercase parameter is optional.
            if c.is_ascii_lowercase() && self.r.chance(0.5) {
                break;
            }
            // Now and then a value of no particular type in its place.
            let c = if !matches!(c, 'I' | 'B') && self.risky() {
                'A'
            } else {
                c.to_ascii_uppercase()
            };
            args.push(match c {
                'N' => self.num(d),
                'S' => self.text(d),
                'T' => self.container(d).0,
                'I' => Expr::Int(*self.r.pick(&[1, 2, -1, 3, -2])),
                'B' => Expr::Int(*self.r.pick(&[2, 10, 16, 36])),
                _ => self.any(d),
            });
        }
        call(f, args)
    }

    /// An expression of any kind, as a value: an operand goes through `num` or `text`.
    pub(super) fn any(&mut self, d: u32) -> Expr {
        if d == 0 {
            return match self.r.below(10) {
                0..=3 => self.arg(),
                4..=6 => match self.var(|v| !v.kind.seq()) {
                    Some((n, _)) => name(&n),
                    None => self.leaf(),
                },
                7 => raw(format!("G{}", self.r.range(1, 3))),
                _ => self.leaf(),
            };
        }
        let d1 = d - 1;
        match self.r.below(26) {
            0..=5 => {
                let (a, b) = (self.num(d1), self.num(d1));
                bin(self.r.pick(&ARITH[..7]), a, b)
            }
            // Bitwise operations take integers.
            6 => {
                let int = |g: &mut Gen| match g.r.below(3) {
                    0 => g.e(Kind::Int, d1),
                    _ if g.risky() => g.num(d1),
                    _ => g.arg_for(Role::Int),
                };
                let (a, b) = (int(self), int(self));
                bin(self.r.pick(&ARITH[7..]), a, b)
            }
            7..=9 => self.any(0),
            10..=12 => {
                let k = *self.r.pick(&[
                    Kind::Int,
                    Kind::Float,
                    Kind::Num,
                    Kind::Str,
                    Kind::Bool,
                    Kind::Int,
                ]);
                self.e(k, d)
            }
            13..=15 => self.index(d1),
            16 | 17 => self.call_any(d1),
            18 => {
                let (a, b) = (self.any(d1), self.any(d1));
                bin(self.r.pick(&["and", "or"]), a, b)
            }
            19 => {
                let (c, a, b) = (self.e(Kind::Bool, d1), self.any(d1), self.any(d1));
                bin("or", bin("and", c, a), b)
            }
            20 => match self.r.below(3) {
                0 => un("not", self.any(d1)),
                1 => un("~", self.arg_for(Role::Int)),
                _ => un("-", self.num(d1)),
            },
            21 => self.concat(d1),
            22 if self.va.is_some() => {
                let t = name(self.va.as_deref().expect("the name"));
                match self.r.below(3) {
                    0 => Expr::Field(Box::new(t), "n".into()),
                    1 => Expr::Index(Box::new(t), Box::new(Expr::Int(self.r.range(1, 3)))),
                    _ => {
                        let i = self.e(Kind::Int, d1.min(1));
                        Expr::Index(Box::new(t), Box::new(i))
                    }
                }
            }
            22 if self.vararg => match self.r.below(3) {
                0 => Expr::Paren(Box::new(Expr::Vararg)),
                1 => call("select", vec![Expr::Int(self.r.range(1, 3)), Expr::Vararg]),
                _ => call("select", vec![str("#"), Expr::Vararg]),
            },
            23 if !self.strict => un(
                "#",
                if self.r.chance(0.5) {
                    self.text(d1)
                } else {
                    self.container(d1).0
                },
            ),
            24 => {
                // A sequence only as a new table: one held in a variable must keep its shape.
                let k = *self.r.pick(&[
                    Kind::Rec,
                    Kind::SeqInt,
                    Kind::SeqStr,
                    Kind::SeqAny,
                    Kind::Func,
                ]);
                if k.seq() {
                    self.seq_of(k, d1)
                } else {
                    self.e(k, d1)
                }
            }
            _ => {
                let (a, b) = (self.num(d1), self.num(d1));
                bin(self.r.pick(CMP), a, b)
            }
        }
    }
}
