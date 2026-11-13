//! The Lua programs the generator builds, and their text.

#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    Nil,
    True,
    False,
    Int(i64),
    Float(f64),
    Str(Vec<u8>),
    Name(String),
    Vararg,
    Index(Box<Expr>, Box<Expr>),
    Field(Box<Expr>, String),
    Call(Box<Expr>, Vec<Expr>),
    Method(Box<Expr>, String, Vec<Expr>),
    Bin(&'static str, Box<Expr>, Box<Expr>),
    Un(&'static str, Box<Expr>),
    Func(Box<Func>),
    Table(Vec<Item>),
    /// `(e)`: one value of a call or `...`.
    Paren(Box<Expr>),
    /// Source text taken as is.
    Raw(String),
}

#[derive(Clone, Debug, PartialEq)]
pub enum Item {
    Pos(Expr),
    Named(String, Expr),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Func {
    pub params: Vec<String>,
    /// `...`, or `...name`.
    pub vararg: Option<Option<String>>,
    pub body: Block,
}

pub type Block = Vec<Stmt>;

#[derive(Clone, Debug, PartialEq)]
pub enum Stmt {
    /// Names with their attribute (`const`, `close`).
    Local(Vec<(String, Option<&'static str>)>, Vec<Expr>),
    Assign(Vec<Expr>, Vec<Expr>),
    Call(Expr),
    Do(Block),
    While(Expr, Block),
    Repeat(Block, Expr),
    If(Vec<(Expr, Block)>, Option<Block>),
    NumFor(String, Expr, Expr, Option<Expr>, Block),
    GenFor(Vec<String>, Vec<Expr>, Block),
    LocalFunc(String, Func),
    /// `function t.name` or `function t:name`.
    FuncDef(String, String, bool, Func),
    Return(Vec<Expr>),
    Break,
    Goto(String),
    Label(String),
    /// Source lines taken as is.
    Raw(String),
}

pub fn name(s: &str) -> Expr {
    Expr::Name(s.to_string())
}

pub fn raw(s: impl Into<String>) -> Expr {
    Expr::Raw(s.into())
}

pub fn str(s: &str) -> Expr {
    Expr::Str(s.as_bytes().to_vec())
}

pub fn call(f: &str, args: Vec<Expr>) -> Expr {
    Expr::Call(Box::new(raw(f)), args)
}

pub fn bin(op: &'static str, a: Expr, b: Expr) -> Expr {
    Expr::Bin(op, Box::new(a), Box::new(b))
}

pub fn un(op: &'static str, a: Expr) -> Expr {
    Expr::Un(op, Box::new(a))
}

pub fn local(n: &str, e: Expr) -> Stmt {
    Stmt::Local(vec![(n.to_string(), None)], vec![e])
}

pub fn assign(target: Expr, e: Expr) -> Stmt {
    Stmt::Assign(vec![target], vec![e])
}

pub struct Printer {
    pub out: String,
    indent: usize,
}

impl Printer {
    pub fn new(indent: usize) -> Printer {
        Printer {
            out: String::new(),
            indent,
        }
    }

    fn line(&mut self, s: &str) {
        for _ in 0..self.indent {
            self.out.push_str("  ");
        }
        self.out.push_str(s);
        self.out.push('\n');
    }

    pub fn block(&mut self, b: &Block) {
        self.indent += 1;
        for s in b {
            self.stmt(s);
        }
        self.indent -= 1;
    }

    fn func(&mut self, head: &str, f: &Func) {
        let mut ps = f.params.clone();
        match &f.vararg {
            Some(Some(n)) => ps.push(format!("...{n}")),
            Some(None) => ps.push("...".into()),
            None => {}
        }
        self.line(&format!("{head}({})", ps.join(", ")));
        self.block(&f.body);
        self.line("end");
    }

    pub fn stmt(&mut self, s: &Stmt) {
        match s {
            Stmt::Local(names, values) => {
                let ns: Vec<String> = (names.iter())
                    .map(|(n, a)| match a {
                        Some(a) => format!("{n} <{a}>"),
                        None => n.clone(),
                    })
                    .collect();
                if values.is_empty() {
                    self.line(&format!("local {}", ns.join(", ")));
                } else {
                    self.line(&format!("local {} = {}", ns.join(", "), exprs(values)));
                }
            }
            Stmt::Assign(targets, values) => {
                self.line(&format!("{} = {}", exprs(targets), exprs(values)))
            }
            Stmt::Call(e) => self.line(&expr(e)),
            Stmt::Do(b) => {
                self.line("do");
                self.block(b);
                self.line("end");
            }
            Stmt::While(c, b) => {
                self.line(&format!("while {} do", expr(c)));
                self.block(b);
                self.line("end");
            }
            Stmt::Repeat(b, c) => {
                self.line("repeat");
                self.block(b);
                self.line(&format!("until {}", expr(c)));
            }
            Stmt::If(arms, els) => {
                for (i, (c, b)) in arms.iter().enumerate() {
                    let kw = if i == 0 { "if" } else { "elseif" };
                    self.line(&format!("{kw} {} then", expr(c)));
                    self.block(b);
                }
                if let Some(b) = els {
                    self.line("else");
                    self.block(b);
                }
                self.line("end");
            }
            Stmt::NumFor(v, a, b, step, body) => {
                let step = step.as_ref().map(|s| format!(", {}", expr(s)));
                self.line(&format!(
                    "for {v} = {}, {}{} do",
                    expr(a),
                    expr(b),
                    step.unwrap_or_default()
                ));
                self.block(body);
                self.line("end");
            }
            Stmt::GenFor(vs, es, body) => {
                self.line(&format!("for {} in {} do", vs.join(", "), exprs(es)));
                self.block(body);
                self.line("end");
            }
            Stmt::LocalFunc(n, f) => self.func(&format!("local function {n}"), f),
            Stmt::FuncDef(t, n, method, f) => {
                let sep = if *method { ':' } else { '.' };
                self.func(&format!("function {t}{sep}{n}"), f)
            }
            Stmt::Return(es) => self.line(&format!("return {}", exprs(es))),
            Stmt::Break => self.line("break"),
            Stmt::Goto(l) => self.line(&format!("goto {l}")),
            Stmt::Label(l) => self.line(&format!("::{l}::")),
            Stmt::Raw(text) => {
                for l in text.lines() {
                    self.line(l);
                }
            }
        }
    }
}

fn exprs(es: &[Expr]) -> String {
    es.iter().map(expr).collect::<Vec<_>>().join(", ")
}

/// A prefix expression: something a call, index or method may follow.
fn prefix(e: &Expr) -> String {
    match e {
        Expr::Name(_) | Expr::Index(..) | Expr::Field(..) | Expr::Call(..) | Expr::Method(..) => {
            expr(e)
        }
        Expr::Paren(_) | Expr::Bin(..) | Expr::Un(..) => expr(e),
        Expr::Raw(s)
            if s.chars()
                .all(|c| c.is_alphanumeric() || c == '_' || c == '.') =>
        {
            s.clone()
        }
        _ => format!("({})", expr(e)),
    }
}

pub fn expr(e: &Expr) -> String {
    match e {
        Expr::Nil => "nil".into(),
        Expr::True => "true".into(),
        Expr::False => "false".into(),
        Expr::Int(i64::MIN) => "math.mininteger".into(),
        Expr::Int(n) if *n < 0 => format!("({n})"),
        Expr::Int(n) => n.to_string(),
        Expr::Float(f) if f.is_nan() => "(0/0)".into(),
        Expr::Float(f) if f.is_infinite() => (if *f > 0.0 { "(1/0)" } else { "(-1/0)" }).into(),
        Expr::Float(f) => {
            let s = format!("{f:?}");
            if f.is_sign_negative() {
                format!("({s})")
            } else {
                s
            }
        }
        Expr::Str(b) => {
            let mut s = String::from("\"");
            for &c in b {
                match c {
                    b'"' => s.push_str("\\\""),
                    b'\\' => s.push_str("\\\\"),
                    b'\n' => s.push_str("\\n"),
                    0x20..=0x7e => s.push(c as char),
                    _ => s.push_str(&format!("\\{c:03}")),
                }
            }
            s.push('"');
            s
        }
        Expr::Name(n) => n.clone(),
        Expr::Vararg => "...".into(),
        Expr::Index(t, k) => format!("{}[{}]", prefix(t), expr(k)),
        Expr::Field(t, n) => format!("{}.{n}", prefix(t)),
        Expr::Call(f, args) => format!("{}({})", prefix(f), exprs(args)),
        Expr::Method(o, n, args) => format!("{}:{n}({})", prefix(o), exprs(args)),
        // Every operation in its own parentheses: no precedence to get wrong.
        Expr::Bin(op, a, b) => format!("({} {op} {})", expr(a), expr(b)),
        Expr::Un(op, a) => {
            let sep = if *op == "not" { " " } else { "" };
            format!("({op}{sep}{})", expr(a))
        }
        Expr::Func(f) => {
            let mut p = Printer::new(0);
            p.func("function", f);
            // Inline: a function expression's lines join the enclosing statement's.
            p.out.trim_end().replace('\n', " ")
        }
        Expr::Table(items) => {
            let parts: Vec<String> = (items.iter())
                .map(|i| match i {
                    Item::Pos(e) => expr(e),
                    Item::Named(n, e) => format!("{n} = {}", expr(e)),
                })
                .collect();
            format!("{{{}}}", parts.join(", "))
        }
        Expr::Paren(e) => format!("({})", expr(e)),
        Expr::Raw(s) => s.clone(),
    }
}
