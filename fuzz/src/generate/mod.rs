//! The program generator. A program is the prelude and a few units; a unit is one function `f`
//! with the driver that warms it up on one family of values and then calls it with others.
//!
//! What a generated program prints must depend on nothing but its text, on any correct Lua:
//!
//! - It ends. A loop that its text does not bound spends fuel (`tick`), and a function calls
//!   itself or one defined later only with a counter that bounds the depth (`recur`).
//! - No string grows by more than a bounded amount per operation: an operation that builds a
//!   string takes at most one operand that may itself have been built (`fresh`), and one that
//!   multiplies its operand's size (`rep`, `%q`) takes none, so sizes stay linear in the fuel
//!   spent.
//!   A string or sequence that outlives a call is cut or capped where it is assigned, and a
//!   loop inside a loop spends fuel whatever bounds it: the cost of a call is bounded too.
//! - `pairs` only feeds sums that do not depend on its order.
//! - No address reaches the output: `tostring` and `%s` take values known to be no table,
//!   function or thread.
//!
//! A *strict* unit also stays where tcvm and reference Lua are meant to agree, so its output
//! is compared with `lua`'s: it writes no float as text but one of its arguments' (tcvm writes
//! `%.14g`), and takes lengths only of strings and of tables it built without holes.

mod driver;
mod expr;
mod pool;
mod scenario;
mod stmt;
mod templates;

use std::sync::LazyLock;

use crate::ast::{self, Block, Expr, Func, Item, Printer, Stmt};
use crate::rng::Rng;

/// The head of every program: the output functions, and `reset` with the pool.
static PRELUDE: LazyLock<String> =
    LazyLock::new(|| include_str!("../prelude.lua").replace("--@POOL\n", &pool::lua()));

/// What the generator knows of an expression's value.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    /// Nothing.
    Any,
    /// One of `f`'s arguments, unchanged: a value of the pool.
    Arg,
    Bool,
    Int,
    Float,
    /// An integer or a float.
    Num,
    Str,
    /// A table built as a sequence of integers, of strings, or of non-nil values.
    SeqInt,
    SeqStr,
    SeqAny,
    /// A table with the fields `x`, `y`, `z`, `w` and no metatable.
    Rec,
    Func,
}

impl Kind {
    fn seq(self) -> bool {
        matches!(self, Kind::SeqInt | Kind::SeqStr | Kind::SeqAny)
    }
}

/// What `f` is warmed up with for one of its parameters, and so what its body mostly takes
/// the parameter for; the calls after the warm-up pass anything.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Role {
    Num,
    /// Numbers that are all integers: what bitwise operations take.
    Int,
    Str,
    Tab,
    Func,
    Mixed,
}

struct Var {
    name: String,
    kind: Kind,
    /// Never assigned after its declaration.
    fixed: bool,
    /// A function of the unit, and how many parameters it names.
    helper: Option<usize>,
    /// Declared outside `f`: it outlives a call, so what it holds must not grow with the
    /// number of calls.
    outlives: bool,
}

pub struct Unit {
    pub tag: String,
    /// What the unit is, for its header line: a template's name, or a description.
    pub kind: String,
    pub strict: bool,
    /// Many times the usual size.
    pub big: bool,
    pub code: Block,
}

pub struct Program {
    pub units: Vec<Unit>,
}

impl Program {
    pub fn text(&self) -> String {
        let mut out = PRELUDE.clone();
        for u in &self.units {
            let strict = if u.strict { " strict" } else { "" };
            out.push_str(&format!("do -- {} {}{strict}\n  reset()\n", u.tag, u.kind));
            let mut p = Printer::new(0);
            p.block(&u.code);
            out.push_str(&p.out);
            out.push_str("end\n");
        }
        out.push_str("emit(\"END\")\n");
        out
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Form {
    Local,
    Assign,
    Store,
    Call,
    Probe,
    If,
    For,
    ForDyn,
    While,
    Repeat,
    Ipairs,
    Pairs,
    Iter,
    Closure,
    Method,
    Pcall,
    Error,
    Close,
    Coroutine,
    Global,
    Multi,
    Return,
    Vararg,
    Goto,
    Break,
    Junk,
    Pattern,
    Xpcall,
    Ctor,
    Tablib,
}

/// The statement forms and how often a unit takes each; a unit drops some of them altogether
/// (swarm testing: a form that is always there hides what its absence would show).
const FORMS: &[(Form, u32)] = &[
    (Form::Local, 10),
    (Form::Assign, 10),
    (Form::Store, 6),
    (Form::Call, 4),
    (Form::Probe, 4),
    (Form::If, 7),
    (Form::For, 6),
    (Form::ForDyn, 3),
    (Form::While, 3),
    (Form::Repeat, 2),
    (Form::Ipairs, 3),
    (Form::Pairs, 2),
    (Form::Iter, 1),
    (Form::Closure, 4),
    (Form::Method, 2),
    (Form::Pcall, 3),
    (Form::Error, 2),
    (Form::Close, 1),
    (Form::Coroutine, 2),
    (Form::Global, 3),
    (Form::Multi, 2),
    (Form::Return, 2),
    (Form::Vararg, 2),
    (Form::Goto, 2),
    (Form::Break, 2),
    (Form::Junk, 2),
    (Form::Pattern, 3),
    (Form::Xpcall, 2),
    (Form::Ctor, 2),
    (Form::Tablib, 2),
];

/// Builds Lua text: a template's body of `f`, a scenario's part.
type Build = fn(&mut Gen) -> String;

/// Whether `gen`'s KIND names a kind of unit.
pub fn is_kind(name: &str) -> bool {
    matches!(name, "free" | "limits" | "scenario")
        || templates::TEMPLATES.iter().any(|t| t.name == name)
}

pub struct Gen {
    pub r: Rng,
    strict: bool,
    scopes: Vec<Vec<Var>>,
    /// Unit-level statements ahead of `f`.
    pre: Block,
    names: u32,
    /// Statements left to the unit.
    budget: i32,
    loops: u32,
    /// A loop was built: `f` is slow to call.
    heavy: bool,
    /// A unit many times the usual size: its calls get more fuel.
    big: bool,
    /// The innermost loop takes `goto` to its end.
    conts: Vec<Option<String>>,
    /// Directly in a coroutine's body: `coroutine.yield` is allowed.
    yields: bool,
    /// The function being built takes `...`.
    vararg: bool,
    /// The function being built names its extra arguments' table.
    va: Option<String>,
    weights: Vec<u32>,
    roles: [Role; 3],
    /// How often an operand is a value of no particular type where the operation wants one:
    /// most units keep to the types, so that most of their code runs.
    risk: f64,
}

impl Gen {
    pub fn new(seed: u64) -> Gen {
        Gen::with(Rng::new(seed))
    }

    /// At the start of a unit.
    fn with(r: Rng) -> Gen {
        Gen {
            r,
            strict: false,
            scopes: vec![Vec::new()],
            pre: Vec::new(),
            names: 0,
            budget: 0,
            loops: 0,
            heavy: false,
            big: false,
            conts: Vec::new(),
            yields: false,
            vararg: false,
            va: None,
            weights: Vec::new(),
            roles: [Role::Mixed; 3],
            risk: 0.0,
        }
    }

    pub fn program(&mut self, only: Option<&str>) -> Program {
        let n = self.r.range(2, 5) as usize;
        let units = (1..=n).map(|i| self.unit(i, only)).collect();
        Program { units }
    }

    fn fresh_name(&mut self, prefix: &str) -> String {
        self.names += 1;
        format!("{prefix}{}", self.names)
    }

    fn declare(&mut self, name: &str, kind: Kind, fixed: bool) -> &mut Var {
        let scope = self.scopes.last_mut().expect("a scope");
        scope.push(Var {
            name: name.to_string(),
            kind,
            fixed,
            helper: None,
            outlives: false,
        });
        scope.last_mut().expect("the variable")
    }

    /// A visible variable `ok` accepts: (name, kind).
    fn var(&mut self, ok: impl Fn(&Var) -> bool) -> Option<(String, Kind)> {
        let found: Vec<(String, Kind)> = (self.scopes.iter().flatten())
            .filter(|v| ok(v))
            .map(|v| (v.name.clone(), v.kind))
            .collect();
        if found.is_empty() {
            return None;
        }
        Some(self.r.pick(&found).clone())
    }

    /// A function of the unit in sight: (name, parameters).
    fn helper(&mut self) -> Option<(String, usize)> {
        let found: Vec<(String, usize)> = (self.scopes.iter().flatten())
            .filter_map(|v| Some((v.name.clone(), v.helper?)))
            .collect();
        (!found.is_empty()).then(|| self.r.pick(&found).clone())
    }

    fn risky(&mut self) -> bool {
        self.r.chance(self.risk)
    }

    fn outlives(&self, name: &str) -> bool {
        self.scopes
            .iter()
            .flatten()
            .any(|v| v.name == name && v.outlives)
    }

    fn sees(&self, name: &str) -> bool {
        self.scopes.iter().flatten().any(|v| v.name == name)
    }

    fn unit(&mut self, idx: usize, only: Option<&str>) -> Unit {
        let tag = format!("U{idx}");
        *self = Gen::with(self.r.clone());
        self.risk = *self.r.pick(&[0.0, 0.0, 0.0, 0.02, 0.02, 0.08, 0.25]);
        self.strict = self.r.chance(0.6);
        self.weights = (FORMS.iter())
            .map(|&(form, w)| {
                let core = matches!(form, Form::Local | Form::Assign | Form::If);
                if !core && self.r.chance(0.25) || form == Form::Error && self.risk == 0.0 {
                    0
                } else {
                    w * self.r.range(1, 3) as u32
                }
            })
            .collect();
        self.budget = self.r.range(4, 28) as i32;
        // A few hundred lines of parts that are programs of their own, wired together.
        if only == Some("scenario") || only.is_none() && self.r.chance(0.08) {
            let (kind, code) = self.scenario_unit(&tag);
            return Unit {
                tag,
                kind,
                strict: self.strict,
                big: false,
                code,
            };
        }
        // The hand-written shapes: small functions that each aim at one thing the JIT
        // speculates on. Not strict: they take lengths and write floats freely.
        let pick = self.r.f64();
        let template = match only {
            Some("free") => false,
            Some(_) => true,
            None => pick < 0.35,
        };
        if template {
            self.strict = false;
            self.risk = 0.0;
            if only == Some("limits") || only.is_none() && pick < 0.03 {
                let code = self.limits_unit(&tag);
                return Unit {
                    tag,
                    kind: "limits".into(),
                    strict: false,
                    big: false,
                    code,
                };
            }
            let (t, mut body, wf) = self.template(only);
            if self.r.chance(0.25) {
                body = format!("{}\n{body}", self.junk());
            }
            self.strict = t.strict;
            let wf = wf.map(|f| self.fam(f));
            let mut code = std::mem::take(&mut self.pre);
            code.push(Stmt::Raw(format!(
                "local function f(a, b, c)\n  {}\nend",
                body.replace('\n', "\n  ")
            )));
            code.extend(self.driver(&tag, t.heavy, Some(wf), &body));
            return Unit {
                tag,
                kind: t.name.into(),
                strict: self.strict,
                big: false,
                code,
            };
        }
        self.roles = std::array::from_fn(|_| {
            *self.r.pick(&[
                Role::Num,
                Role::Num,
                Role::Num,
                Role::Int,
                Role::Int,
                Role::Str,
                Role::Str,
                Role::Tab,
                Role::Tab,
                Role::Func,
                Role::Mixed,
            ])
        });
        // Unit-level state `f` reaches as upvalues: literals, which cannot raise out here.
        for _ in 0..self.r.range(0, 2) {
            let (k, e) = match self.r.below(5) {
                0 => (Kind::Int, Expr::Int(self.r.range(-3, 100))),
                1 => (
                    Kind::Float,
                    Expr::Float(*self.r.pick(&[0.0, 0.5, 1.0, 2.5, -1.5])),
                ),
                2 => (Kind::Str, ast::str(self.r.pick(&["", "a", "up", "10"]))),
                3 => (Kind::Rec, Expr::Table(vec![])),
                _ => (
                    Kind::SeqInt,
                    Expr::Table(
                        (0..self.r.range(0, 3))
                            .map(|i| Item::Pos(Expr::Int(i)))
                            .collect(),
                    ),
                ),
            };
            let n = self.fresh_name("u");
            self.declare(&n, k, false).outlives = true;
            self.pre.push(ast::local(&n, e));
        }
        // Lua 5.5 global declarations: a declared global with its definition check, and under
        // `global *` every other name as before.
        if self.r.chance(0.2) {
            let g = format!("g{tag}v");
            self.pre.push(Stmt::Raw(format!(
                "global *\nglobal {g} = {}",
                self.r.range(-3, 100)
            )));
            self.declare(&g, Kind::Int, false);
            if self.r.chance(0.5) {
                let h = format!("g{tag}f");
                self.pre.push(Stmt::Raw(format!(
                    "global function {h}(x)\n  return x, {g}\nend"
                )));
                self.declare(&h, Kind::Func, true).helper = Some(1);
            }
        }
        self.scopes.push(Vec::new());
        for p in ["a", "b", "c"] {
            self.declare(p, Kind::Arg, true);
        }
        // Now and then a unit many times the usual size, nested deeper.
        // Kept to the types, so that more of it runs.
        let big = self.r.chance(0.05);
        let (stmts, depth) = if big {
            self.budget = self.r.range(60, 150) as i32;
            self.heavy = true;
            self.big = true;
            self.risk = 0.0;
            let e = FORMS.iter().position(|f| f.0 == Form::Error);
            self.weights[e.expect("a form")] = 0;
            (self.r.range(15, 40) as usize, 4)
        } else {
            (self.r.range(2, 9) as usize, 3)
        };
        let mut body = self.block_in_scope(stmts, depth);
        body.push(self.final_return());
        self.scopes.pop();
        let mut code = std::mem::take(&mut self.pre);
        code.push(Stmt::LocalFunc(
            "f".into(),
            Func {
                params: vec!["a".into(), "b".into(), "c".into()],
                vararg: None,
                body,
            },
        ));
        code.extend(self.driver(&tag, self.heavy, None, ""));
        Unit {
            tag,
            kind: format!(
                "free{} risk={} roles={:?}",
                if big { " big" } else { "" },
                self.risk,
                self.roles
            ),
            strict: self.strict,
            big,
            code,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_program_is_a_function_of_its_seed() {
        for seed in 0..50 {
            let text = Gen::new(seed).program(None).text();
            assert_eq!(text, Gen::new(seed).program(None).text());
            assert!(text.ends_with("emit(\"END\")\n"));
        }
    }
}
