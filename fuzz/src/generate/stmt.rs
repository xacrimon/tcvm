//! Statements. Each form returns the statements it adds to the block being built and
//! declares what they bring into scope.

use super::{FORMS, Form, Gen, Kind};
use crate::ast::{
    self, Block, Expr, Func, Item, Stmt, assign, bin, call, local, name, raw, str, un,
};

/// What a function body must not inherit from the code around it.
struct Outer {
    loops: u32,
    conts: Vec<Option<String>>,
    yields: bool,
    vararg: bool,
    va: Option<String>,
}

impl Gen {
    fn enter_function(&mut self) -> Outer {
        Outer {
            loops: std::mem::take(&mut self.loops),
            conts: std::mem::take(&mut self.conts),
            yields: std::mem::take(&mut self.yields),
            vararg: std::mem::take(&mut self.vararg),
            va: self.va.take(),
        }
    }

    fn leave_function(&mut self, o: Outer) {
        self.loops = o.loops;
        self.conts = o.conts;
        self.yields = o.yields;
        self.vararg = o.vararg;
        self.va = o.va;
    }

    /// A function's body in a scope of its own: what `setup` declares and starts it with, `count`
    /// (a range) statements at most, and a return.
    fn func_body(
        &mut self,
        count: (i64, i64),
        d: u32,
        setup: impl FnOnce(&mut Gen) -> Block,
    ) -> Block {
        let outer = self.enter_function();
        self.scopes.push(Vec::new());
        let mut body = setup(self);
        let n = self.r.range(count.0, count.1) as usize;
        body.extend(self.block_in_scope(n, d));
        body.push(self.final_return());
        self.scopes.pop();
        self.leave_function(outer);
        body
    }

    /// A method's function: `self` a record, and the parameter `p`.
    fn method_func(&mut self, p: String, d: u32) -> Func {
        let body = self.func_body((0, 2), d, |g| {
            g.declare("self", Kind::Rec, true);
            g.declare(&p, Kind::Any, false);
            Vec::new()
        });
        Func {
            params: vec![p],
            vararg: None,
            body,
        }
    }

    /// `function() … end` of a few statements, most often with a return.
    fn thunk(&mut self, d: u32) -> Expr {
        let outer = self.enter_function();
        let n = self.r.range(1, 3) as usize;
        let mut body = self.block(n, d);
        if self.r.chance(0.6) {
            body.push(Stmt::Return(self.args(1, d)));
        }
        self.leave_function(outer);
        Expr::Func(Box::new(Func {
            params: vec![],
            vararg: None,
            body,
        }))
    }

    /// Up to `n` statements in a scope of their own.
    pub(super) fn block(&mut self, n: usize, d: u32) -> Block {
        self.scopes.push(Vec::new());
        let b = self.block_in_scope(n, d);
        self.scopes.pop();
        b
    }

    pub(super) fn block_in_scope(&mut self, n: usize, d: u32) -> Block {
        let mut b = Vec::new();
        for _ in 0..n {
            if self.budget <= 0 {
                break;
            }
            self.budget -= 1;
            b.extend(self.stmt(d));
        }
        b
    }

    fn small_block(&mut self, d: u32) -> Block {
        let n = self.r.range(1, 3) as usize;
        self.block(n, d)
    }

    /// Spend fuel: the first statement of a loop body its text does not bound.
    fn tick(&mut self) -> Stmt {
        if self.r.chance(0.5) {
            Stmt::Raw("tick()".into())
        } else {
            Stmt::Raw("FUEL = FUEL - 1 if FUEL < 0 then error(\"fuel\", 0) end".into())
        }
    }

    /// A loop its text bounds spends fuel all the same inside another loop: trips multiply.
    fn nested_tick(&mut self) -> Block {
        if self.loops > 0 {
            vec![self.tick()]
        } else {
            Vec::new()
        }
    }

    fn cond(&mut self, d: u32) -> Expr {
        if self.r.chance(0.7) {
            self.e(Kind::Bool, d)
        } else {
            self.any(d)
        }
    }

    fn pick_kind(&mut self) -> Kind {
        const KINDS: &[(Kind, u32)] = &[
            (Kind::Any, 8),
            (Kind::Int, 7),
            (Kind::Float, 3),
            (Kind::Num, 2),
            (Kind::Str, 5),
            (Kind::Bool, 2),
            (Kind::SeqInt, 2),
            (Kind::SeqStr, 1),
            (Kind::SeqAny, 2),
            (Kind::Rec, 4),
        ];
        self.r.pick_by(KINDS, |k| k.1).0
    }

    pub(super) fn local_stmt(&mut self, d: u32) -> Block {
        let k = self.pick_kind();
        let e = self.e(k, d);
        let n = self.fresh_name("v");
        let fixed = self.r.chance(0.15);
        self.declare(&n, k, fixed);
        let attr = (fixed && self.r.chance(0.5)).then_some("const");
        vec![Stmt::Local(vec![(n, attr)], vec![e])]
    }

    /// An element a sequence of kind `k` may grow by: never nil.
    pub(super) fn element(&mut self, k: Kind, d: u32) -> Expr {
        match k {
            Kind::SeqInt => self.e(Kind::Int, d),
            Kind::SeqStr => match self.r.below(3) {
                0 => call("tostring", vec![self.e(Kind::Int, d.min(1))]),
                _ => str(self.r.pick(&["", "a", "bc", "10", "x"])),
            },
            _ => {
                let k = *self
                    .r
                    .pick(&[Kind::Int, Kind::Str, Kind::Bool, Kind::Float, Kind::Rec]);
                self.e(k, d)
            }
        }
    }

    fn assign_stmt(&mut self, d: u32) -> Block {
        let Some((n, k)) = self.var(|v| !v.fixed && v.kind != Kind::Func) else {
            return self.local_stmt(d);
        };
        let v = name(&n);
        let e = match k {
            Kind::Int if self.r.chance(0.6) => {
                let op = *self
                    .r
                    .pick(&["+", "-", "*", "//", "%", "~", "&", "|", "<<", ">>"]);
                bin(op, v.clone(), self.e(Kind::Int, d))
            }
            Kind::Float | Kind::Num if self.r.chance(0.6) => {
                let op = *self.r.pick(&["+", "-", "*", "/"]);
                bin(op, v.clone(), self.e(k, d))
            }
            Kind::Str if self.r.chance(0.5) => bin("..", v.clone(), self.fresh(d)),
            Kind::Any if self.risky() => {
                let op = *self
                    .r
                    .pick(&["+", "-", "*", "/", "//", "%", "..", "&", "|", "~"]);
                if op == ".." {
                    if self.strict {
                        self.any(d)
                    } else {
                        bin("..", v.clone(), self.fresh(d))
                    }
                } else {
                    bin(op, v.clone(), self.any(d))
                }
            }
            Kind::Bool if self.r.chance(0.3) => un("not", v.clone()),
            _ => self.e(k, d),
        };
        let e = match k {
            Kind::Str if self.outlives(&n) => {
                call("string.sub", vec![e, Expr::Int(1), Expr::Int(48)])
            }
            _ => e,
        };
        vec![assign(v, e)]
    }

    fn store_stmt(&mut self, d: u32) -> Block {
        let (t, k) = self.container(d);
        if !matches!(t, Expr::Name(_)) {
            return self.local_stmt(d);
        }
        let s = match k {
            Kind::Rec => assign(Expr::Field(Box::new(t), self.field()), self.any(d)),
            k if k.seq() => {
                let e = self.element(k, d);
                let capped = matches!(&t, Expr::Name(n) if self.outlives(n))
                    .then(|| bin("<", un("#", t.clone()), Expr::Int(16)));
                let grow = if self.r.chance(0.5) {
                    let at = bin("+", un("#", t.clone()), Expr::Int(1));
                    assign(Expr::Index(Box::new(t), Box::new(at)), e)
                } else {
                    Stmt::Call(call("table.insert", vec![t, e]))
                };
                match capped {
                    Some(c) => Stmt::If(vec![(c, vec![grow])], None),
                    None => grow,
                }
            }
            _ => {
                let v = if self.r.chance(0.15) {
                    Expr::Nil
                } else {
                    self.any(d)
                };
                match self.r.below(5) {
                    0 => assign(
                        Expr::Field(Box::new(t), self.r.pick(&["x", "y", "q"]).to_string()),
                        v,
                    ),
                    1 => Stmt::Call(call("rawset", vec![t, self.key(d), v])),
                    _ => {
                        let key = self.key(d);
                        assign(Expr::Index(Box::new(t), Box::new(key)), v)
                    }
                }
            }
        };
        vec![s]
    }

    /// Locals that go out of scope at once: the slots they held keep values other than nil
    /// for what is put there next.
    pub(super) fn junk(&mut self) -> String {
        let n = self.r.range(3, 10) as usize;
        let names: Vec<String> = (1..=n).map(|i| format!("j{i}")).collect();
        let vals: Vec<&str> = (0..n)
            .map(|_| {
                *self
                    .r
                    .pick(&["1", "2.5", "\"s\"", "true", "MT", "-7", "{}", "V"])
            })
            .collect();
        format!("do local {} = {} end", names.join(", "), vals.join(", "))
    }

    /// A body that runs with the loop state `conts` adds; the `goto` label it used, last.
    fn loop_body(&mut self, cont: bool, head: Block, n: usize, d: u32) -> Block {
        self.loops += 1;
        self.heavy = true;
        let label = cont.then(|| self.fresh_name("c"));
        self.conts.push(label.clone());
        let mut b = head;
        b.extend(self.block_in_scope(n, d));
        self.conts.pop();
        self.loops -= 1;
        if let Some(l) = label
            && contains_goto(&b, &l)
        {
            b.push(Stmt::Label(l));
        }
        b
    }

    /// `loop_body` in a scope of its own that holds the loop's variables.
    fn loop_scope(&mut self, vars: &[(&str, Kind)], head: Block, n: usize, d: u32) -> Block {
        self.scopes.push(Vec::new());
        for &(v, kind) in vars {
            self.declare(v, kind, true);
        }
        let body = self.loop_body(true, head, n, d);
        self.scopes.pop();
        body
    }

    fn for_stmt(&mut self, d: u32) -> Block {
        let v = self.fresh_name("i");
        let (a, b, step, kind) = if self.r.chance(0.15) {
            let a = *self.r.pick(&[0.0, 1.0, 0.5, -1.0]);
            let s = *self.r.pick(&[0.5, 1.0, 0.25, 1.5]);
            let n = self.r.range(0, 8) as f64;
            (
                Expr::Float(a),
                Expr::Float(a + s * n),
                Some(Expr::Float(s)),
                Kind::Float,
            )
        } else {
            let a = self.r.range(-2, 3);
            let s = *self.r.pick(&[1, 1, 1, 2, 3, -1, -2]);
            let n = self.r.range(0, 12);
            let step = (s != 1 || self.r.chance(0.2)).then_some(Expr::Int(s));
            (
                Expr::Int(a),
                Expr::Int(a + s * n - s.signum()),
                step,
                Kind::Int,
            )
        };
        let n = self.r.range(1, 4) as usize;
        let head = self.nested_tick();
        let body = self.loop_scope(&[(&v, kind)], head, n, d);
        vec![Stmt::NumFor(v, a, b, step, body)]
    }

    fn fordyn_stmt(&mut self, d: u32) -> Block {
        let v = self.fresh_name("i");
        // Most run a few times; the rest go wherever their operands take them, on fuel.
        let tame = self.r.chance(0.85);
        let bound = |g: &mut Gen| match g.r.below(4) {
            0 => g.e(Kind::Int, 1),
            1 => g.e(Kind::Num, 1),
            _ => g.num(0),
        };
        let (a, b, step) = if tame {
            let hi = match self.r.below(3) {
                0 => bin("%", self.num(0), Expr::Int(self.r.range(2, 12))),
                1 => {
                    let k = *self.r.pick(&[Kind::SeqInt, Kind::SeqAny, Kind::Str]);
                    un("#", self.e(k, 1))
                }
                _ => bin(
                    "&",
                    self.e(Kind::Int, 1),
                    Expr::Int(*self.r.pick(&[3, 7, 15])),
                ),
            };
            (
                Expr::Int(self.r.range(0, 1)),
                hi,
                self.r.chance(0.2).then_some(Expr::Int(2)),
            )
        } else {
            let (a, b) = (bound(self), bound(self));
            (a, b, self.r.chance(0.6).then(|| bound(self)))
        };
        let head = vec![self.tick()];
        let n = self.r.range(1, 4) as usize;
        let body = self.loop_scope(&[(&v, Kind::Num)], head, n, d);
        vec![Stmt::NumFor(v, a, b, step, body)]
    }

    fn while_stmt(&mut self, d: u32) -> Block {
        let n = self.r.range(1, 4) as usize;
        if self.r.chance(0.5) {
            // Bounded by its counter, which nothing else assigns.
            let c = self.fresh_name("n");
            let lim = Expr::Int(self.r.range(0, 15));
            let mut head = vec![assign(name(&c), bin("+", name(&c), Expr::Int(1)))];
            head.extend(self.nested_tick());
            let body = self.loop_scope(&[(&c, Kind::Int)], head, n, d);
            let op = *self.r.pick(&["<", "<=", "~="]);
            return vec![
                local(&c, Expr::Int(0)),
                Stmt::While(bin(op, name(&c), lim), body),
            ];
        }
        // A condition of the program's own, most often under a bound on the trips.
        let c = self.cond(d);
        let trips = self.fresh_name("n");
        let bounded = self.r.chance(0.9);
        let mut head = vec![self.tick()];
        if bounded {
            head.push(assign(name(&trips), bin("+", name(&trips), Expr::Int(1))));
        }
        let body = self.loop_scope(&[], head, n, d);
        if !bounded {
            return vec![Stmt::While(c, body)];
        }
        let c = bin(
            "and",
            bin("<", name(&trips), Expr::Int(self.r.range(1, 24))),
            c,
        );
        vec![local(&trips, Expr::Int(0)), Stmt::While(c, body)]
    }

    fn repeat_stmt(&mut self, d: u32) -> Block {
        self.scopes.push(Vec::new());
        let head = vec![self.tick()];
        let n = self.r.range(1, 4) as usize;
        // No `goto` to the end: `until` reads the body's locals.
        let body = self.loop_body(false, head, n, d);
        let c = self.cond(d);
        self.scopes.pop();
        vec![Stmt::Repeat(body, c)]
    }

    fn ipairs_stmt(&mut self, d: u32) -> Block {
        let (t, k) = self.container(d);
        let (i, v) = (self.fresh_name("i"), self.fresh_name("e"));
        let vk = match k {
            Kind::SeqInt => Kind::Int,
            Kind::SeqStr => Kind::Str,
            _ => Kind::Any,
        };
        let head = vec![self.tick()];
        let n = self.r.range(1, 3) as usize;
        let body = self.loop_scope(&[(&i, Kind::Int), (&v, vk)], head, n, d);
        vec![Stmt::GenFor(
            vec![i, v],
            vec![call("ipairs", vec![t])],
            body,
        )]
    }

    /// `pairs` into a count and a sum, which no order changes.
    fn pairs_stmt(&mut self, d: u32) -> Block {
        let (t, _) = self.container(d);
        let (n, s) = (self.fresh_name("n"), self.fresh_name("s"));
        let cond = *self.r.pick(&[
            "true",
            "type(v) == \"number\"",
            "hk(k) % 2 == 0",
            "v == k",
            "type(k) == \"string\"",
        ]);
        let body =
            format!("tick()\nif {cond} then {n} = {n} + 1 end\n{s} = {s} + hk(k) * 3 + hk(v)");
        let out = vec![
            Stmt::Local(
                vec![(n.clone(), None), (s.clone(), None)],
                vec![Expr::Int(0), Expr::Int(0)],
            ),
            Stmt::GenFor(
                vec!["k".into(), "v".into()],
                vec![call("pairs", vec![t])],
                vec![Stmt::Raw(body)],
            ),
        ];
        self.declare(&n, Kind::Int, false);
        self.declare(&s, Kind::Int, false);
        self.heavy = true;
        out
    }

    fn iter_stmt(&mut self, d: u32) -> Block {
        let it = self.fresh_name("it");
        let step = self.r.range(1, 3);
        let def = format!(
            "local function {it}(s, i)\n  i = i + {step}\n  if i <= s then return i, i * {} end\nend",
            self.r.range(1, 3)
        );
        let (i, j) = (self.fresh_name("i"), self.fresh_name("j"));
        let n = self.r.range(1, 3) as usize;
        let head = self.nested_tick();
        let body = self.loop_scope(&[(&i, Kind::Int), (&j, Kind::Int)], head, n, d);
        let lim = Expr::Int(self.r.range(0, 12));
        vec![
            Stmt::Raw(def),
            Stmt::GenFor(vec![i, j], vec![name(&it), lim, Expr::Int(0)], body),
        ]
    }

    /// A function of the unit, callable from what follows it: never from itself.
    fn closure_stmt(&mut self, d: u32, vararg: bool) -> Block {
        let h = self.fresh_name("h");
        let params: Vec<String> = (0..self.r.range(0, 3))
            .map(|_| self.fresh_name("p"))
            .collect();
        let mut va = None;
        let body = self.func_body((1, 4), d, |g| {
            for p in &params {
                g.declare(p, Kind::Any, false);
            }
            if vararg && g.r.chance(0.3) {
                let t = g.fresh_name("va");
                g.declare(&t, Kind::Any, true);
                g.va = Some(t.clone());
                va = Some(Some(t));
            } else if vararg {
                g.vararg = true;
                va = Some(None);
            }
            Vec::new()
        });
        self.declare(&h, Kind::Func, true).helper = Some(params.len());
        let f = Func {
            params,
            vararg: va,
            body,
        };
        vec![Stmt::LocalFunc(h, f)]
    }

    /// A method found through a metatable's `__index`: `local c = {} c.__index = c
    /// function c:m(p) … end local o = setmetatable({…}, c) o:m(…)`.
    fn class_stmt(&mut self, d: u32) -> Block {
        let (c, o, m, p) = (
            self.fresh_name("k"),
            self.fresh_name("o"),
            self.fresh_name("m"),
            self.fresh_name("p"),
        );
        let f = self.method_func(p, d);
        let fields = self.e(Kind::Rec, d);
        let fields = if matches!(fields, Expr::Table(_)) {
            fields
        } else {
            Expr::Table(vec![])
        };
        let mut out = vec![
            Stmt::Raw(format!("local {c} = {{}}\n{c}.__index = {c}")),
            Stmt::FuncDef(c.clone(), m.clone(), true, f),
            local(&o, call("setmetatable", vec![fields, name(&c)])),
        ];
        for _ in 0..self.r.range(1, 3) {
            let v = self.fresh_name("v");
            out.push(local(
                &v,
                Expr::Method(Box::new(name(&o)), m.clone(), self.args(1, d)),
            ));
            self.declare(&v, Kind::Any, false);
        }
        out
    }

    fn method_stmt(&mut self, d: u32) -> Block {
        if self.r.chance(0.4) {
            return self.class_stmt(d);
        }
        let mut out = Vec::new();
        let r = match self.var(|v| v.kind == Kind::Rec) {
            Some((r, _)) => r,
            None => {
                let r = self.fresh_name("v");
                out.push(local(&r, self.e(Kind::Rec, d)));
                self.declare(&r, Kind::Rec, true);
                r
            }
        };
        let m = self.fresh_name("m");
        let p = self.fresh_name("p");
        let f = self.method_func(p, d);
        out.push(Stmt::FuncDef(r.clone(), m.clone(), true, f));
        for _ in 0..self.r.range(1, 2) {
            let v = self.fresh_name("v");
            let c = Expr::Method(Box::new(name(&r)), m.clone(), self.args(1, d));
            out.push(local(&v, c));
            self.declare(&v, Kind::Any, false);
        }
        out
    }

    /// A pattern function on a string: what it finds is anything.
    fn pattern_stmt(&mut self, d: u32) -> Block {
        let s = self.text(d);
        let p = str(self.r.pick(&[
            "%a+",
            "(%w+)",
            "%d+",
            "^%s*(.-)%s*$",
            "%b()",
            "%f[%a]%a",
            "()",
            "(.)%1",
            "[^%s]+",
            ".-",
            "a*",
            "(a)(b?)",
        ]));
        let v = self.fresh_name("v");
        let e = match self.r.below(4) {
            0 => call("string.find", vec![s, p, self.e(Kind::Int, 0)]),
            1 => call("string.match", vec![s, p]),
            2 => {
                let r = match self.r.below(3) {
                    0 => str(self.r.pick(&["%1", "<%0>", "", "%%"])),
                    1 => raw("{a = \"A\", b = false}"),
                    _ => raw("function(m) if #m > 1 then return m:sub(2) end end"),
                };
                // At most one replacement per character: the length stays bounded.
                call("string.gsub", vec![s, p, r, Expr::Int(self.r.range(1, 4))])
            }
            _ => {
                let n = self.fresh_name("n");
                self.declare(&n, Kind::Int, false);
                return vec![
                    local(&n, Expr::Int(0)),
                    Stmt::Raw(format!(
                        "for _ in string.gmatch({}, {}) do {n} = {n} + 1 if {n} > 20 then break end end",
                        ast::expr(&s),
                        ast::expr(&p)
                    )),
                ];
            }
        };
        self.declare(&v, Kind::Any, false);
        vec![local(&v, e)]
    }

    /// `xpcall` of a function of the unit or a new one, with a handler.
    fn xpcall_stmt(&mut self, d: u32) -> Block {
        let (ok, e) = (self.fresh_name("ok"), self.fresh_name("r"));
        let h = raw(*self.r.pick(&[
            "function(e) return type(e) end",
            "function(e) return e end",
            "type",
            "function(e) return select('#', e) end",
        ]));
        let f = match self.helper().filter(|_| self.r.chance(0.4)) {
            Some((f, _)) => name(&f),
            None => self.thunk(d),
        };
        let mut args = vec![f, h];
        let n = self.r.below(3);
        args.extend(self.args(n, d.min(1)));
        self.declare(&ok, Kind::Bool, false);
        self.declare(&e, Kind::Any, false);
        vec![Stmt::Local(
            vec![(ok, None), (e, None)],
            vec![call("xpcall", args)],
        )]
    }

    /// A constructor that mixes item kinds and expands a call or `...` at its end or not.
    fn ctor_stmt(&mut self, d: u32) -> Block {
        let mut items = Vec::new();
        for _ in 0..self.r.range(1, 5) {
            items.push(match self.r.below(5) {
                0 => Item::Named(self.field(), self.any(d)),
                1 => Item::Pos(self.call_any(d.min(1))),
                2 if self.vararg => Item::Pos(Expr::Vararg),
                _ => Item::Pos(self.any(d)),
            });
        }
        let v = self.fresh_name("v");
        self.declare(&v, Kind::Any, false);
        vec![local(&v, Expr::Table(items))]
    }

    /// `sort`, `concat`, `move` on a sequence the unit built.
    fn tablib_stmt(&mut self, d: u32) -> Block {
        let found = self.var(|v| matches!(v.kind, Kind::SeqInt | Kind::SeqStr));
        let Some((t, k)) = found else {
            return self.local_stmt(d);
        };
        let v = self.fresh_name("v");
        let (e, vk) = match self.r.below(4) {
            0 => {
                let cmp = self.r.pick(&["", ", function(x, y) return x > y end"]);
                return vec![Stmt::Raw(format!("table.sort({t}{cmp})"))];
            }
            1 => (
                call(
                    "table.concat",
                    vec![name(&t), str(self.r.pick(&[",", "", "--"]))],
                ),
                Kind::Str,
            ),
            // A copy: the sequence keeps no holes.
            2 => (raw(format!("table.move({t}, 1, #{t}, 1, {{}})")), k),
            _ => (
                raw(format!("select('#', table.unpack({t}, 1, #{t}))")),
                Kind::Int,
            ),
        };
        self.declare(&v, vk, false);
        vec![local(&v, e)]
    }

    fn pcall_stmt(&mut self, d: u32) -> Block {
        let (ok, e) = (self.fresh_name("ok"), self.fresh_name("r"));
        let h = self.helper();
        let c = if let Some((f, n)) = h.filter(|_| self.r.chance(0.4)) {
            let mut args = vec![name(&f)];
            args.extend(self.args(n, d));
            call("pcall", args)
        } else {
            call("pcall", vec![self.thunk(d)])
        };
        self.declare(&ok, Kind::Bool, false);
        self.declare(&e, Kind::Any, false);
        vec![Stmt::Local(vec![(ok, None), (e, None)], vec![c])]
    }

    fn error_stmt(&mut self, d: u32) -> Block {
        let v = match self.r.below(5) {
            0 => self.e(Kind::Int, d),
            1 => self.e(Kind::Rec, d),
            2 => self.arg_or_nil(),
            _ => self.fresh(d),
        };
        let mut args = vec![v];
        if self.r.chance(0.4) {
            args.push(Expr::Int(self.r.range(0, 2)));
        }
        vec![Stmt::If(
            vec![(self.cond(d), vec![Stmt::Call(call("error", args))])],
            None,
        )]
    }

    fn arg_or_nil(&mut self) -> Expr {
        if self.r.chance(0.3) {
            Expr::Nil
        } else {
            self.e(Kind::Arg, 0)
        }
    }

    fn close_stmt(&mut self, d: u32) -> Block {
        let c = self.fresh_name("cl");
        let v = raw(*self.r.pick(&["CL", "CL", "nil", "false"]));
        let mut b = vec![Stmt::Local(vec![(c, Some("close"))], vec![v])];
        let n = self.r.range(1, 3) as usize;
        b.extend(self.block(n, d));
        vec![Stmt::Do(b)]
    }

    fn coroutine_stmt(&mut self, d: u32) -> Block {
        if self.yields {
            let y = self.fresh_name("y");
            let e = call("coroutine.yield", self.args(1, d));
            self.declare(&y, Kind::Any, false);
            return vec![local(&y, e)];
        }
        let co = self.fresh_name("co");
        let p = self.fresh_name("p");
        let body = self.func_body((1, 4), d, |g| {
            g.yields = true;
            g.declare(&p, Kind::Any, false);
            g.budget += 3;
            let y = g.fresh_name("y");
            vec![local(&y, call("coroutine.yield", g.args(1, 1)))]
        });
        let f = Func {
            params: vec![p],
            vararg: None,
            body,
        };
        let wrap = self.r.chance(0.6);
        let make = if wrap {
            "coroutine.wrap"
        } else {
            "coroutine.create"
        };
        let mut out = vec![local(&co, call(make, vec![Expr::Func(Box::new(f))]))];
        // Two calls end a body that yields once; a third finds it dead.
        let calls = if self.risky() { 3 } else { self.r.range(1, 2) };
        for _ in 0..calls {
            let a = self.args(1, d);
            if wrap {
                let v = self.fresh_name("v");
                out.push(local(&v, call(&co, a)));
                self.declare(&v, Kind::Any, false);
            } else {
                let (ok, v) = (self.fresh_name("ok"), self.fresh_name("v"));
                let mut args = vec![name(&co)];
                args.extend(a);
                out.push(Stmt::Local(
                    vec![(ok.clone(), None), (v.clone(), None)],
                    vec![call("coroutine.resume", args)],
                ));
                self.declare(&ok, Kind::Bool, false);
                self.declare(&v, Kind::Any, false);
            }
        }
        if !wrap {
            let s = self.fresh_name("v");
            out.push(local(&s, call("coroutine.status", vec![name(&co)])));
            self.declare(&s, Kind::Str, false);
        }
        out
    }

    fn multi_stmt(&mut self, d: u32) -> Block {
        match self.r.below(3) {
            0 => {
                let Some((x, k)) = self.var(|v| !v.fixed && v.kind != Kind::Func) else {
                    return self.local_stmt(d);
                };
                let Some((y, _)) = self.var(|v| !v.fixed && v.kind == k && v.name != x) else {
                    return self.local_stmt(d);
                };
                vec![Stmt::Assign(
                    vec![name(&x), name(&y)],
                    vec![name(&y), name(&x)],
                )]
            }
            k => {
                let names: Vec<String> = (0..self.r.range(2, 3))
                    .map(|_| self.fresh_name("v"))
                    .collect();
                let c = if k == 1 {
                    self.call_any(d)
                } else {
                    let (t, _) = self.container(d);
                    let hi = Expr::Int(self.r.range(0, 4));
                    call("table.unpack", vec![t, Expr::Int(1), hi])
                };
                for n in &names {
                    self.declare(n, Kind::Any, false);
                }
                vec![Stmt::Local(
                    names.into_iter().map(|n| (n, None)).collect(),
                    vec![c],
                )]
            }
        }
    }

    pub(super) fn final_return(&mut self) -> Stmt {
        let n = self.r.range(1, 3) as usize;
        let mut es = self.args(n, 2);
        if self.r.chance(0.25) {
            es.push(self.call_any(1));
        }
        Stmt::Return(es)
    }

    fn stmt(&mut self, d: u32) -> Block {
        let nested = d > 0;
        let w: Vec<u32> = (FORMS.iter().zip(&self.weights))
            .map(|(&(form, _), &w)| {
                let ok = match form {
                    Form::Local
                    | Form::Assign
                    | Form::Store
                    | Form::Call
                    | Form::Probe
                    | Form::Global
                    | Form::Multi
                    | Form::Junk
                    | Form::Pattern
                    | Form::Ctor
                    | Form::Tablib => true,
                    Form::Goto => matches!(self.conts.last(), Some(Some(_))),
                    Form::Break => self.loops > 0,
                    Form::Coroutine => nested || self.yields,
                    _ => nested,
                };
                if ok { w } else { 0 }
            })
            .collect();
        if w.iter().all(|&x| x == 0) {
            return self.local_stmt(d);
        }
        let d1 = d.saturating_sub(1);
        match FORMS[self.r.weighted(&w)].0 {
            Form::Local => self.local_stmt(2),
            Form::Assign => self.assign_stmt(2),
            Form::Store => self.store_stmt(2),
            Form::Call => {
                let c = self.call_any(2);
                // A statement that starts with `(` would continue the one before it.
                if ast::expr(&c).starts_with('(') {
                    vec![local("_", c)]
                } else {
                    vec![Stmt::Call(c)]
                }
            }
            Form::Probe => vec![Stmt::Call(call("probe", self.args(1, 2)))],
            Form::If => {
                let mut arms = vec![(self.cond(2), self.small_block(d1))];
                if self.r.chance(0.25) {
                    arms.push((self.cond(1), self.small_block(d1)));
                }
                let els = self.r.chance(0.4).then(|| self.small_block(d1));
                vec![Stmt::If(arms, els)]
            }
            Form::For => self.for_stmt(d1),
            Form::ForDyn => self.fordyn_stmt(d1),
            Form::While => self.while_stmt(d1),
            Form::Repeat => self.repeat_stmt(d1),
            Form::Ipairs => self.ipairs_stmt(d1),
            Form::Pairs => self.pairs_stmt(1),
            Form::Iter => self.iter_stmt(d1),
            Form::Closure => self.closure_stmt(d1, false),
            Form::Vararg => self.closure_stmt(d1, true),
            Form::Method => self.method_stmt(d1),
            Form::Pcall => self.pcall_stmt(d1),
            Form::Error => self.error_stmt(1),
            Form::Close => self.close_stmt(d1),
            Form::Coroutine => self.coroutine_stmt(d1),
            Form::Global => {
                let g = raw(format!("G{}", self.r.range(1, 3)));
                vec![assign(g, self.any(2))]
            }
            Form::Multi => self.multi_stmt(1),
            Form::Junk => vec![Stmt::Raw(self.junk())],
            Form::Pattern => self.pattern_stmt(2),
            Form::Xpcall => self.xpcall_stmt(d1),
            Form::Ctor => self.ctor_stmt(2),
            Form::Tablib => self.tablib_stmt(2),
            Form::Return => vec![Stmt::If(
                vec![(self.cond(2), vec![Stmt::Return(self.args(1, 2))])],
                None,
            )],
            Form::Goto => {
                let l = self.conts.last().cloned().flatten().expect("a label");
                vec![Stmt::If(vec![(self.cond(2), vec![Stmt::Goto(l)])], None)]
            }
            Form::Break => vec![Stmt::If(vec![(self.cond(2), vec![Stmt::Break])], None)],
        }
    }
}

fn contains_goto(b: &Block, label: &str) -> bool {
    b.iter().any(|s| match s {
        Stmt::Goto(l) => l == label,
        Stmt::Do(b)
        | Stmt::While(_, b)
        | Stmt::Repeat(b, _)
        | Stmt::NumFor(.., b)
        | Stmt::GenFor(.., b) => contains_goto(b, label),
        Stmt::If(arms, els) => {
            arms.iter().any(|(_, b)| contains_goto(b, label))
                || els.as_ref().is_some_and(|b| contains_goto(b, label))
        }
        _ => false,
    })
}
