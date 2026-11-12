//! Templates past the sizes and shapes the other forms reach: chains longer than a fixpoint's
//! round cap, recursion and tail-call cycles, values kept across a call made inside an inlined
//! body, and operations that may raise behind a guard in a loop.

use crate::generate::Gen;

/// The templates of this file that are strict: no float or table reaches a string, no length of
/// a table with holes, no `pairs` order.
pub(super) const STRICT: &[&str] = &["chain", "recur", "hoist"];

/// The templates whose call is slow: the driver calls them fewer times.
pub(super) const HEAVY: &[&str] = &["chain", "recur"];

/// What the end of a chain computes from the value that left its front.
const INJECT: &[&str] = &[
    "X + 0.5", "X * 2", "X // 1", "-X", "X + b", "X / 2", "X - 1", "X % 7",
];

/// A guard and what it keeps from running: something that raises, or a loop whose trip count
/// divides by its step, when the guard is false.
const HAZARDS: &[(&str, &str)] = &[
    ("b ~= 0", "s = s + a // b"),
    ("b ~= 0", "s = s + a % b"),
    ("math.type(b) == \"integer\" and b ~= 0", "s = s + a // b"),
    ("c", "s = s + a // b"),
    ("c", "s = s + a % b"),
    (
        "c ~= 0",
        "for k = a, b, c do n = n + 1 if n > 50 then break end end",
    ),
    (
        "c",
        "for k = a, b, c do n = n + 1 if n > 50 then break end end",
    ),
    (
        "c and c ~= 0",
        "for k = a, 1 << 40, c do n = n + 1 if n > 50 then break end end",
    ),
    ("type(a) == \"table\"", "n = n + (a.x and 1 or 0)"),
    ("c", "n = n + (a[1] and 1 or 0)"),
    ("c", "s = s + (1 << 62) // b"),
    ("i > 1000", "s = s + a // b"),
    ("b > 0", "s = s + a // (b - b)"),
];

impl Gen {
    /// A loop that moves values along a chain of locals: a type that enters at one end needs
    /// one fixpoint round per local to reach the other.
    pub(super) fn t_chain(&mut self) -> String {
        let n = *self.r.pick(&[6, 20, 40, 52, 60, 90, 120]);
        let form = self.r.below(4);
        // Calls and branches per hop make each round costlier: shorter chains.
        let n = if form >= 2 { n.min(60) } else { n };
        let vs: Vec<String> = (1..=n).map(|i| format!("v{i}")).collect();
        let init: Vec<&str> = (0..n)
            .map(|_| *self.r.pick(&["a", "a", "a", "b"]))
            .collect();
        let inject = *self.r.pick(INJECT);
        let mut out = format!("local {} = {}\n", vs.join(", "), init.join(", "));
        if form == 2 {
            self.helper_stmt("local function cid(x) return x end");
        }
        let trips = n as i64 + *self.r.pick(&[-3, 2, 5]);
        out.push_str(&format!("for i = 1, {trips} do\n"));
        match form {
            0 => out.push_str(&format!(
                "  {} = {}, {}\n",
                vs.join(", "),
                vs[1..].join(", "),
                inject.replace('X', "v1")
            )),
            _ => {
                out.push_str("  local t = v1\n");
                for k in 0..n - 1 {
                    let (to, from) = (&vs[k], &vs[k + 1]);
                    out.push_str(&match form {
                        1 => format!("  {to} = {from}\n"),
                        2 => format!("  {to} = cid({from})\n"),
                        _ => format!("  if i > {} then {to} = {from} end\n", k % 3),
                    });
                }
                out.push_str(&format!("  {} = {}\n", vs[n - 1], inject.replace('X', "t")));
            }
        }
        out.push_str("end\n");
        let mid = &vs[n / 2];
        out.push_str(&format!("return v1, math.type(v1), {mid}, {}", vs[n - 1]));
        out
    }

    /// Recursion bounded by a counter: tail-call cycles through a vararg function (deep on
    /// every 97th call), mutual recursion, recursion through `__index` and `pcall`.
    pub(super) fn t_recur(&mut self) -> String {
        self.helper_stmt("local RC = 0");
        match self.r.below(4) {
            0 => {
                let extra =
                    *self
                        .r
                        .pick(&["", ", 1", ", 1, 2, 3", ", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10"]);
                let x = *self
                    .r
                    .pick(&["x", "x", "select(\"#\", ...)", "(...)", "x + 1"]);
                let locals = *self.r.pick(&["", "local p, q, r ", "local p = ... "]);
                self.helper_stmt(&format!(
                    "local rcF\nlocal function rcg(k, x)\n  if k <= 0 then return x end\n  return rcF(k - 1, x{extra})\nend\nrcF = function(k, x, ...) {locals}return rcg(k, {x}) end"
                ));
                let deep = *self.r.pick(&[3000, 100000]);
                let shallow = *self.r.pick(&[0, 1, 3, 8]);
                format!(
                    "RC = RC + 1\nlocal d = RC % 97 == 0 and {deep} or {shallow}\nreturn rcF(d, a)"
                )
            }
            1 => {
                let op = *self.r.pick(&["+", "-", "*", "//"]);
                self.helper_stmt(&format!(
                    "local rcod\nlocal function rcev(n, x)\n  if n <= 0 then return x, 0 end\n  local r, c = rcod(n - 1, x {op} 1)\n  return r, c + 1\nend\nrcod = function(n, x)\n  if n <= 0 then return x, 1 end\n  local r, c = rcev(n - 1, x)\n  return r, c + 2\nend"
                ));
                let tail = *self
                    .r
                    .pick(&["return rcev(n, a)", "local r, k = rcod(n, a)\nreturn k, r"]);
                format!("local n = math.type(b) == \"integer\" and b % 50 or 7\n{tail}")
            }
            2 => {
                self.helper_stmt(
                    "local rcmt = {}\nrcmt.__index = function(t, k)\n  if math.type(k) ~= \"integer\" or k <= 0 then return 0 end\n  return t[k - 1] + 1\nend",
                );
                "local t = setmetatable({}, rcmt)\nlocal n = math.type(b) == \"integer\" and b % 40 or 5\nreturn t[n], t[n // 2]"
                    .to_string()
            }
            _ => {
                self.helper_stmt(
                    "local function rcp(n, x)\n  if n <= 0 then error(x, 0) end\n  local ok, e = pcall(rcp, n - 1, x)\n  return n, ok, e\nend",
                );
                "local n = math.type(b) == \"integer\" and b % 30 or 4\nreturn rcp(n, a)"
                    .to_string()
            }
        }
    }

    /// A value copied, then kept across a call made inside an inlined body, then assigned back
    /// and read after an exit: the collector clears the slots the call does not keep.
    pub(super) fn t_callslot(&mut self) -> String {
        // Much allocation once `b` is no number: a collection inside the call that exits.
        self.helper_stmt(
            "local function csh(k, ...)\n  local x = {}\n  for j = 1, type(k) == \"number\" and 2 or 3000 do x[j % 8] = {j} end\nend",
        );
        self.helper_stmt("local function csmk(...) return {v = 1} end");
        if self.r.chance(0.5) {
            let (mk, rd) = *self.r.pick(&[
                ("csmk()", "x.v"),
                ("{v = a}", "x.v"),
                ("csmk(a)", "x.v"),
                ("a", "x"),
            ]);
            let calls = *self.r.pick(&["csh(k)", "csh(k) csh(k)", "csmk() csh(k)"]);
            self.helper_stmt(&format!("local function csg(t, k) {calls} return t end"));
            let ret = self
                .r
                .pick(&[
                    format!("return {rd} + b, {rd}"),
                    format!("return {rd} + b"),
                    format!("local r = {rd} + b\nreturn r, x"),
                ])
                .clone();
            // In a vararg function of its own: `{...}` is what the issue's case took.
            let tab = *self.r.pick(&["{...}", "{...}", "{b}"]);
            let shape =
                format!("local x = {mk}\nlocal t = {tab}\nlocal y = x\ncsg(t, b)\nx = y\n{ret}");
            if self.r.chance(0.3) {
                return shape.replace("{...}", "{c}");
            }
            let body = shape.replace('\n', "\n  ");
            self.helper_stmt(&format!("local function csrun(a, b, ...)\n  {body}\nend"));
            self.tc(&[
                "return csrun(a, b, c)",
                "return csrun(a, b)",
                "local r, v = csrun(a, b, c, a)\nreturn v, r",
            ])
            .to_string()
        } else {
            self.helper_stmt("local function cs3(...) return {v = 1}, 2, 3 end");
            let first = *self
                .r
                .pick(&["local p, q, r = cs3()", "local q, p = 1, cs3()"]);
            self.helper_stmt(&format!(
                "local function csg2(k)\n  {first}\n  local y = p\n  csh(k)\n  p = y\n  return p.v + k, p.v\nend"
            ));
            let tail = *self.r.pick(&[
                "local t = {c}\nlocal x, y = csg2(b)\nreturn x + y, b",
                "return csg2(b)",
                "local x, y = csg2(b)\nreturn y, x",
            ]);
            tail.to_string()
        }
    }

    /// An operation that raises (or a loop whose trip count divides by its step) behind a
    /// guard, inside a loop: what a hoist must not move above the guard.
    pub(super) fn t_hoist(&mut self) -> String {
        let (guard, haz) = *self.r.pick(HAZARDS);
        let trips = *self.r.pick(&[3, 20, 60]);
        let outer = if self.r.chance(0.3) {
            format!("for o = 1, 2 do\nfor i = 1, {trips} do")
        } else {
            format!("do\nfor i = 1, {trips} do")
        };
        format!(
            "local s, n = 0, 0\n{outer}\n  if {guard} then\n    {haz}\n  end\nend\nend\nreturn s, n"
        )
    }
}
