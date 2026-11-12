//! A unit's driver: `f` is called many times with values of one family each (so it compiles
//! speculating on them), then with values of other families, then with the first again. A
//! warm-up's results are counted per distinct result, the calls after it are written out.

use super::{Gen, Role};
use crate::ast::{Block, Stmt};

const FAMS: &[&str] = &[
    "si", "bi", "fl", "fs", "ns", "st", "ot", "tb", "fn", "nil1", "sp", "sn", "fp", "mn", "ib",
    "tt", "tm", "mx", "ms",
];
const NUMF: &[&str] = &["si", "sp", "fl", "fp", "mn", "ib", "bi"];
/// The pool's families and their sizes, by how often a single call takes a value of each.
const POOL: &[(&str, i64, u32)] = &[
    ("si", 10, 3),
    ("bi", 10, 2),
    ("fl", 10, 2),
    ("fs", 10, 2),
    ("ns", 8, 1),
    ("st", 7, 1),
    ("ot", 2, 1),
    ("tb", 7, 1),
    ("fn", 3, 1),
    ("co", 1, 1),
    ("mx", 8, 1),
    ("ms", 6, 1),
];
const EVENTS: &[&str] = &[
    "MTI.__index.x = 'changed'",
    "MT.__add = nil; MT.__index = MTI.__index",
    "MTI.__index = function(t, k) return 'fn' end",
    "MT.__eq = nil; MT.__lt = nil",
    "MT.__len = nil",
    "MT.__call = nil",
    "MTI.__index.x = nil",
    "V.tb[4].x = nil; V.tb[4][1] = nil",
    "setmetatable(V.tb[5], MT)",
    "setmetatable(V.tb[1], MTI)",
    "setmetatable(V.tb[3], MT)",
    "V.tb[3].x = 'str'; V.tb[3].q = 1; V.tb[2][4] = 'four'",
    "V.tb[3].x = nil",
    "G1 = 1; G2 = 'g'",
];

/// The ways `f` is entered, and how often a phase takes each. All pass on `f`'s results and
/// raise its errors unchanged, so what a unit prints does not depend on the choice.
const ADAPTERS: &[(&str, u32)] = &[
    ("f", 40),
    ("via_tail", 12),
    ("via_tail2", 6),
    ("via_va", 12),
    ("via_pcall", 12),
    ("via_cs", 12),
    ("via_co", 6),
];
/// The definitions of the adapters; `call` is the one `run` enters `f` by.
const ADAPTER_DEFS: &str = "local callee = f
local function via_tail(a, b, c) return f(a, b, c) end
local function via_tail2(a, b, c) return via_tail(a, b, c) end
local function via_va(...) return f(...) end
local function via_cs(a, b, c) return callee(a, b, c) end
local function via_pcall(...)
  local r = table.pack(pcall(f, ...))
  if not r[1] then error(r[2], 0) end
  return table.unpack(r, 2, r.n)
end
local function via_co(...)
  local r = table.pack(coroutine.resume(coroutine.create(f), ...))
  if not r[1] then error(r[2], 0) end
  return table.unpack(r, 2, r.n)
end
local call = f
";
/// What the call site of `via_cs` is made to call instead of `f`.
const CALLEES: &[&str] = &[
    "math.max",
    "math.abs",
    "rawequal",
    "function() end",
    "function(a, b, c) return c, b, a end",
    "function(...) return ... end",
    "V.fn[1]",
    "V.fn[2]",
];

impl Gen {
    /// A strict unit takes no float whose text differs between tcvm and reference Lua.
    pub(super) fn fam(&self, f: &'static str) -> &'static str {
        if self.strict && f == "fs" { "fc" } else { f }
    }

    /// The family each parameter is warmed up with: one its role calls for.
    fn warm_fams(&mut self) -> [&'static str; 3] {
        let roles = self.roles;
        roles.map(|role| {
            let f = match role {
                Role::Num => *self.r.pick(NUMF),
                Role::Int => *self.r.pick(&["si", "sp", "sn", "ib", "bi"]),
                Role::Str => *self.r.pick(&["st", "st", "ns"]),
                Role::Tab => *self.r.pick(&["tt", "tm", "tb", "tt"]),
                Role::Func => "fn",
                Role::Mixed => *self.r.pick(FAMS),
            };
            self.fam(f)
        })
    }

    fn adapter(&mut self) -> &'static str {
        let w: Vec<u32> = ADAPTERS.iter().map(|a| a.1).collect();
        ADAPTERS[self.r.weighted(&w)].0
    }

    /// A family other than `f`'s, often a numeric one for a numeric one: int and float feedback.
    fn other_fam(&mut self, f: &'static str) -> &'static str {
        let l = if NUMF.contains(&f) && self.r.chance(0.5) {
            NUMF
        } else {
            FAMS
        };
        let g = *self.r.pick(l);
        self.fam(g)
    }

    fn pool_value(&mut self) -> String {
        if self.r.chance(0.05) {
            return "nil".into();
        }
        let w: Vec<u32> = POOL.iter().map(|p| p.2).collect();
        let (fam, n, _) = POOL[self.r.weighted(&w)];
        let n = if self.strict && fam == "fs" { 8 } else { n };
        format!("V.{}[{}]", self.fam(fam), self.r.range(1, n))
    }

    /// `wf` overrides the families the roles call for; `body` is `f`'s text where it is a
    /// template's.
    pub(super) fn driver(
        &mut self,
        tag: &str,
        heavy: bool,
        wf: Option<[&'static str; 3]>,
        body: &str,
    ) -> Block {
        let wf = wf.unwrap_or_else(|| self.warm_fams());
        // A call that runs out of fuel costs all of it: the more fuel, the fewer calls.
        let (fuel, nwarm) = if heavy {
            *self.r.pick(&[(200, 150), (600, 60), (2000, 20)])
        } else {
            (300, *self.r.pick(&[300, 600, 1000, 2000]))
        };
        let again = nwarm.min(200);
        let each = format!(
            "pos = i\n      local a, b, c = LA[(i - 1) % LA.n + 1], LB[i % LB.n + 1], LC[(i + 1) % LC.n + 1]\n      FUEL = {fuel}\n      local s = sers(call(a, b, c)) .. probes()\n      h[s] = (h[s] or 0) + 1"
        );
        let run = if self.r.chance(0.12) {
            format!(
                "local co = coroutine.wrap(function()\n    for i = pos + 1, n do\n      {each}\n      if i % 7 == 0 then coroutine.yield() end\n    end\n    return \"done\"\n  end)\n  while co() ~= \"done\" do end"
            )
        } else {
            format!("for i = pos + 1, n do\n      {each}\n    end")
        };
        let mut out = vec![Stmt::Raw(format!(
            "{ADAPTER_DEFS}local function run(LA, LB, LC, n, h)\n  {run}\nend\nlocal function drive(LA, LB, LC, n, tag)\n  local h, errs = {{}}, 0\n  pos = 0\n  while pos < n do\n    local ok, e = pcall(run, LA, LB, LC, n, h)\n    if ok then break end\n    local k = errn(e) .. probes()\n    h[k] = (h[k] or 0) + 1\n    errs = errs + 1\n    if errs > 300 then break end\n  end\n  hist(tag, h)\nend"
        ))];
        let line = |s: String| Stmt::Raw(s);
        // `f` is entered by an adapter: the call site of `run` and the callee's own entry see
        // a route they were not warmed by.
        let warm_call = self.adapter();
        if warm_call != "f" {
            out.push(line(format!("call = {warm_call}")));
        }
        out.push(line(format!(
            "drive(V.{}, V.{}, V.{}, {nwarm}, \"{tag}w\")",
            wf[0], wf[1], wf[2]
        )));
        // Another family, long enough for the code to be compiled again on what it saw there.
        if self.r.chance(0.35) {
            let mut wf2 = wf;
            for s in 0..3 {
                if self.r.chance(0.6) {
                    wf2[s] = self.other_fam(wf[s]);
                }
            }
            let n2 = (*self.r.pick(&[50, 300])).min(nwarm);
            if self.r.chance(0.5) {
                let c = self.adapter();
                out.push(line(format!("call = {c}")));
            }
            out.push(line(format!(
                "drive(V.{}, V.{}, V.{}, {n2}, \"{tag}w2\")",
                wf2[0], wf2[1], wf2[2]
            )));
        }
        // Between the warm-up and the rest, something the compiled code assumed changes.
        let mut restore = None;
        let x = self.r.f64();
        if x < 0.06 && body.contains("math.floor") {
            out.push(line(
                "local sv = math.floor; math.floor = function(...) return 'FL', ... end".into(),
            ));
            restore = Some("math.floor = sv");
        } else if x < 0.06 && body.contains("math.") {
            out.push(line("local sv = math.max; math.max = math.min".into()));
            restore = Some("math.max = sv");
        } else if x < 0.25 {
            out.push(line(self.r.pick(EVENTS).to_string()));
        }
        let mut swapped = false;
        for j in 0..self.r.range(6, 16) {
            if self.r.chance(0.4) {
                let c = self.adapter();
                // The call site of `via_cs` sees another function than it was warmed with.
                if c == "via_cs" && self.r.chance(0.5) {
                    out.push(line(format!("callee = {}", self.r.pick(CALLEES))));
                    swapped = true;
                }
                out.push(line(format!("call = {c}")));
            }
            let mut vs = [self.pool_value(), self.pool_value(), self.pool_value()];
            // Often two slots stay in the warm family and one changes.
            if self.r.chance(0.5) {
                let keep = self.r.below(3);
                for s in 0..3 {
                    if s != keep {
                        vs[s] = if wf[s] == "nil1" {
                            "nil".into()
                        } else {
                            format!("V.{}[{}]", wf[s], self.r.range(1, 3))
                        };
                    }
                }
            }
            let [a, b, c] = vs;
            out.push(line(if self.r.chance(0.6) {
                format!("drive({{n = 1, {a}}}, {{n = 1, {b}}}, {{n = 1, {c}}}, 2, \"{tag}p{j}\")")
            } else {
                // Not always three arguments: fewer are nil, more are dropped.
                let args = match self.r.below(12) {
                    0 => format!("call, {a}"),
                    1 => format!("call, {a}, {b}"),
                    2 => format!("call, {a}, {b}, {c}, {a}"),
                    3 => "call".to_string(),
                    _ => format!("call, {a}, {b}, {c}"),
                };
                format!("FUEL = {fuel}\nrec(\"{tag}q{j}\", pcall({args}))")
            }));
        }
        if swapped {
            out.push(line("callee = f".into()));
        }
        out.push(line(format!("call = {warm_call}")));
        out.push(line(format!(
            "drive(V.{}, V.{}, V.{}, {again}, \"{tag}z\")",
            wf[0], wf[1], wf[2]
        )));
        out.extend(restore.map(|r| line(r.into())));
        out
    }
}
