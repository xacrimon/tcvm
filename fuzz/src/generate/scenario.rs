//! Scenarios: units of a few hundred lines made of parts that each are a small program of their
//! own (a class hierarchy, a tokenizer with a parser and an evaluator, a coroutine scheduler, a
//! stack VM, a serializer, a vector type, memoized recursion, an LRU cache, an event bus), each
//! with its own variations, wired into `f` so that what one part returns feeds the next.
//!
//! A part's entry is `P_run(n, s)` for an integer `n` in `0..37` and a string of at most 32
//! bytes, and it returns a number and a string. Parts keep to what reference Lua agrees on: no
//! float reaches a string, sorts order totally, `pairs` only feeds sorted keys.

use super::{Build, Gen, Role};
use crate::ast::{Block, Stmt};

/// The parts by the prefix of their names, and what builds each.
const PARTS: &[(&str, Build)] = &[
    ("cls", Gen::sc_cls),
    ("lex", Gen::sc_lex),
    ("sched", Gen::sc_sched),
    ("vm", Gen::sc_vm),
    ("ser", Gen::sc_ser),
    ("vec", Gen::sc_vec),
    ("memo", Gen::sc_memo),
    ("lru", Gen::sc_lru),
    ("events", Gen::sc_events),
];

/// What every part reads its inputs through.
const WIRING: &str = "local function sc_norm(x)
  if math.type(x) == \"integer\" then return x % 37 end
  if math.type(x) == \"float\" then
    local f = math.floor(x)
    if math.type(f) ~= \"integer\" then return 11 end
    return f % 37
  end
  if type(x) == \"string\" then return #x % 37 end
  return 5
end
local function sc_str(x)
  if type(x) == \"string\" then return x:sub(1, 32) end
  return \"s\" .. sc_norm(x)
end";

impl Gen {
    /// A scenario unit's code: its parts, `f` wiring them, and the driver.
    pub(super) fn scenario_unit(&mut self, tag: &str) -> (String, Block) {
        let k = self.r.range(3, 5) as usize;
        let mut parts = PARTS.to_vec();
        self.r.shuffle(&mut parts);
        parts.truncate(k);
        let mut code: Block = vec![Stmt::Raw(WIRING.to_string())];
        for (_, build) in &parts {
            code.push(Stmt::Raw(build(self)));
        }
        let parts: Vec<&str> = parts.iter().map(|p| p.0).collect();
        let body = self.sc_wire(&parts);
        code.push(Stmt::Raw(format!(
            "local function f(a, b, c)\n  {}\nend",
            body.replace('\n', "\n  ")
        )));
        self.strict = true;
        self.roles =
            std::array::from_fn(|_| *self.r.pick(&[Role::Num, Role::Int, Role::Str, Role::Mixed]));
        code.extend(self.driver(tag, true, None, &body));
        (format!("scenario parts={parts:?}"), code)
    }

    /// `f`: the parts in a row, each fed what the one before it returned. Results go to `rs`
    /// by a counter: a nil result must leave no hole a length would see.
    fn sc_wire(&mut self, parts: &[&str]) -> String {
        let mut out = String::from("local n, s = sc_norm(a), sc_str(b)\nlocal rs, nr = {}, 0\n");
        let style = self.r.below(4);
        if style == 1 {
            let fs: Vec<String> = parts.iter().map(|p| format!("{p}_run")).collect();
            out.push_str(&format!(
                "local fs = {{{}}}\nfor i = 1, #fs do\n  local r, t = fs[i](n, s)\n  rs[nr + 1], rs[nr + 2], nr = r, t, nr + 2\n  n, s = sc_norm(r), sc_str(t)\nend\n",
                fs.join(", ")
            ));
        } else {
            for (i, p) in parts.iter().enumerate() {
                let call = match style {
                    0 => format!("{p}_run(n, s)"),
                    // The status is dropped: an error's message feeds the next part.
                    2 => format!("select(2, pcall({p}_run, n, s))"),
                    _ => format!("coroutine.wrap({p}_run)(n, s)"),
                };
                out.push_str(&format!(
                    "local r{i}, t{i} = {call}\nrs[nr + 1], rs[nr + 2], nr = r{i}, t{i}, nr + 2\nn, s = sc_norm(r{i}), sc_str(t{i})\n"
                ));
                if self.r.chance(0.3) {
                    out.push_str("n = (n + sc_norm(c)) % 37\n");
                }
            }
        }
        out.push_str("return n, s, select('#', table.unpack(rs, 1, nr)), rs[1], rs[2]");
        out
    }

    fn sc_cls(&mut self) -> String {
        let link = *self.r.pick(&[
            "setmetatable(Sq, {__index = Shape})",
            "setmetatable(Sq, {__index = function(t, k) return Shape[k] end})",
        ]);
        let tri = *self.r.pick(&[
            "local Tri = setmetatable({}, {__index = Shape})",
            "local Tri = setmetatable({}, {__index = Sq})",
        ]);
        let call = *self.r.pick(&[
            "",
            "setmetatable(Shape, {__call = function(c, ...) return c.new(...) end})",
        ]);
        let mk = if call.is_empty() {
            "Shape.new"
        } else {
            "Shape"
        };
        format!(
            "local Shape = {{}}
Shape.__index = Shape
function Shape.new(kind, w, h) return setmetatable({{kind = kind, w = w, h = h}}, Shape) end
function Shape:area() return self.w * self.h end
function Shape:label() return self.kind .. \":\" .. self:area() end
local function shape_key(p) return p:area() * 1000 + #p.kind * 100 + p.w end
Shape.__lt = function(p, q) return shape_key(p) < shape_key(q) end
Shape.__le = function(p, q) return shape_key(p) <= shape_key(q) end
Shape.__eq = function(p, q) return p:area() == q:area() end
Shape.__tostring = function(o) return o:label() end
{call}
local Sq = {{}}
{link}
Sq.__index = Sq
Sq.__lt, Sq.__le, Sq.__eq, Sq.__tostring = Shape.__lt, Shape.__le, Shape.__eq, Shape.__tostring
function Sq.new(x) return setmetatable(Shape.new(\"sq\", x, x), Sq) end
function Sq:label() return \"[\" .. Shape.label(self) .. \"]\" end
{tri}
Tri.__index = Tri
Tri.__lt, Tri.__le, Tri.__eq, Tri.__tostring = Shape.__lt, Shape.__le, Shape.__eq, Shape.__tostring
function Tri.new(w, h) return setmetatable({{kind = \"tri\", w = w, h = h}}, Tri) end
function Tri:area() return self.w * self.h // 2 end
local function cls_run(n, s)
  local list = {{}}
  for i = 1, 3 + n % 5 do
    local k = (i * 7 + n + #s) % 3
    list[i] = k == 0 and {mk}(\"r\", i, n % 4 + 1) or k == 1 and Sq.new(i % 4 + 1) or Tri.new(i, n % 3 + 2)
  end
  table.sort(list, function(p, q) return p < q end)
  local parts = {{}}
  for i, o in ipairs(list) do parts[i] = tostring(o) end
  local same = 0
  for i = 2, #list do if list[i] == list[i - 1] then same = same + 1 end end
  return list[#list]:area() - list[1]:area() + same, table.concat(parts, \",\"):sub(1, 40)
end"
        )
    }

    /// An expression over `x`, `y`, numbers and integer operators, as text.
    fn lex_src(&mut self, d: u32) -> String {
        if d == 0 || self.r.chance(0.25) {
            return match self.r.below(4) {
                0 => "x".into(),
                1 => "y".into(),
                _ => self.r.range(0, 20).to_string(),
            };
        }
        match self.r.below(6) {
            0 => format!("({})", self.lex_src(d - 1)),
            1 => format!("-{}", self.lex_src(d - 1)),
            _ => {
                let op = *self.r.pick(&["+", "-", "*", "//", "%", "+", "*"]);
                let sp = *self.r.pick(&[" ", "", "  "]);
                format!("{}{sp}{op}{sp}{}", self.lex_src(d - 1), self.lex_src(d - 1))
            }
        }
    }

    fn sc_lex(&mut self) -> String {
        let depth = self.r.range(2, 4) as u32;
        let mut src = self.lex_src(depth);
        if self.r.chance(0.15) {
            src.push_str(self.r.pick(&[")", "(", " $", " 3", "//"]));
        }
        let tokens = *self.r.pick(&[
            "local function lex_tokens(src)
  local toks, i = {}, 1
  while i <= #src do
    local c = src:sub(i, i)
    if c:match(\"%s\") then
      i = i + 1
    elseif c:match(\"%d\") then
      local j = src:find(\"[^%d]\", i) or #src + 1
      toks[#toks + 1] = {t = \"n\", v = math.tointeger(src:sub(i, j - 1))}
      i = j
    elseif c:match(\"%a\") then
      local j = src:find(\"[^%w]\", i) or #src + 1
      toks[#toks + 1] = {t = \"id\", v = src:sub(i, j - 1)}
      i = j
    elseif src:sub(i, i + 1) == \"//\" then
      toks[#toks + 1] = {t = \"op\", v = \"//\"}
      i = i + 2
    else
      toks[#toks + 1] = {t = \"op\", v = c}
      i = i + 1
    end
  end
  return toks
end",
            "local function lex_tokens(src)
  local toks = {}
  for sp, word in src:gmatch(\"(%s*)(%S+)\") do
    local i = 1
    while i <= #word do
      local num = word:match(\"^%d+\", i)
      local id = word:match(\"^%a%w*\", i)
      if num then
        toks[#toks + 1] = {t = \"n\", v = math.tointeger(num)}
        i = i + #num
      elseif id then
        toks[#toks + 1] = {t = \"id\", v = id}
        i = i + #id
      elseif word:find(\"^//\", i) then
        toks[#toks + 1] = {t = \"op\", v = \"//\"}
        i = i + 2
      else
        toks[#toks + 1] = {t = \"op\", v = word:sub(i, i)}
        i = i + 1
      end
    end
  end
  return toks
end",
        ]);
        let eval = *self.r.pick(&[
            "local lex_ops = {
  [\"+\"] = function(x, y) return x + y end, [\"-\"] = function(x, y) return x - y end,
  [\"*\"] = function(x, y) return x * y end, [\"//\"] = function(x, y) return x // y end,
  [\"%\"] = function(x, y) return x % y end,
}
local function lex_eval(e, env)
  if e.k == \"n\" then return e.v end
  if e.k == \"id\" then return env[e.v] or 0 end
  if e.k == \"neg\" then return -lex_eval(e.a, env) end
  return lex_ops[e.op](lex_eval(e.a, env), lex_eval(e.b, env))
end",
            "local function lex_compile(e)
  if e.k == \"n\" then local v = e.v return function() return v end end
  if e.k == \"id\" then local k = e.v return function(env) return env[k] or 0 end end
  if e.k == \"neg\" then local a = lex_compile(e.a) return function(env) return -a(env) end end
  local a, b, op = lex_compile(e.a), lex_compile(e.b), e.op
  if op == \"+\" then return function(env) return a(env) + b(env) end end
  if op == \"-\" then return function(env) return a(env) - b(env) end end
  if op == \"*\" then return function(env) return a(env) * b(env) end end
  if op == \"//\" then return function(env) return a(env) // b(env) end end
  return function(env) return a(env) % b(env) end
end
local function lex_eval(e, env) return lex_compile(e)(env) end",
        ]);
        format!(
            "{tokens}
local function lex_parse(toks)
  local pos = 1
  local expr
  local prec = {{[\"+\"] = 1, [\"-\"] = 1, [\"*\"] = 2, [\"//\"] = 2, [\"%\"] = 2}}
  local function take() pos = pos + 1 return toks[pos - 1] end
  local function atom()
    local t = take()
    if not t then error({{at = pos, msg = \"eof\"}}) end
    if t.t == \"n\" then return {{k = \"n\", v = t.v}} end
    if t.t == \"id\" then return {{k = \"id\", v = t.v}} end
    if t.v == \"(\" then
      local e = expr(1)
      local c = take()
      if not c or c.v ~= \")\" then error({{at = pos, msg = \"paren\"}}) end
      return e
    end
    if t.v == \"-\" then return {{k = \"neg\", a = atom()}} end
    error({{at = pos, msg = \"unexpected \" .. t.v}})
  end
  expr = function(min)
    local l = atom()
    while true do
      local t = toks[pos]
      local p = t and t.t == \"op\" and prec[t.v]
      if not p or p < min then return l end
      take()
      l = {{k = \"bin\", op = t.v, a = l, b = expr(p + 1)}}
    end
  end
  local e = expr(1)
  if pos <= #toks then error({{at = pos, msg = \"trailing\"}}) end
  return e
end
{eval}
local function lex_run(n, s)
  local src = {src:?}
  if s:match(\"^%d+$\") then src = src .. \" + \" .. s end
  local ok, r = pcall(function() return lex_eval(lex_parse(lex_tokens(src)), {{x = n, y = #s}}) end)
  if not ok then
    if type(r) == \"table\" then return r.at, r.msg end
    return -1, (r:gsub(\"^[^:]*:%d+: \", \"\"))
  end
  return r, \"v\" .. r
end"
        )
    }

    fn sc_sched(&mut self) -> String {
        let fail = *self.r.pick(&[
            "",
            "if i == k % 4 + 1 then error(\"boom\" .. i, 0) end",
            "if i == 2 then error({i}) end",
        ]);
        let cons = *self.r.pick(&[
            "while true do
      if #chan > 0 then sum = sum + table.remove(chan, 1)
      elseif done then return sum end
      coroutine.yield(\"c\")
    end",
            "repeat
      while #chan == 0 and not done do coroutine.yield() end
      if #chan > 0 then sum = sum + table.remove(chan) end
    until done and #chan == 0
    return sum",
        ]);
        let res = *self.r.pick(&[
            "",
            "local r <close> = setmetatable({}, {__close = function() log[#log + 1] = \"closed\" end})",
        ]);
        let deep = *self.r.pick(&[
            "coroutine.yield(\"p\")",
            "local function hop(d) if d == 0 then return coroutine.yield(\"p\") end return hop(d - 1) end\n      hop(2)",
            "pcall(coroutine.yield, \"p\")",
        ]);
        format!(
            "local function sched_run(n, s)
  local chan, log, done = {{}}, {{}}, false
  local tasks = {{}}
  local function spawn(name, f, ...) tasks[#tasks + 1] = {{name = name, co = coroutine.create(f), args = table.pack(...)}} end
  spawn(\"prod\", function(k)
    {res}
    for i = 1, k do
      chan[#chan + 1] = i * 3
      {fail}
      {deep}
    end
    done = true
    return \"pdone\"
  end, n % 6 + 1)
  spawn(\"cons\", function()
    local sum = 0
    {cons}
  end)
  spawn(\"gen\", function(m)
    local g = coroutine.wrap(function() for i = 1, m do coroutine.yield(i) end end)
    local t = 0
    for v in g do t = t + v coroutine.yield(coroutine.isyieldable()) end
    return t
  end, #s % 5)
  local steps = 0
  while #tasks > 0 and steps < 200 do
    steps = steps + 1
    local t = table.remove(tasks, 1)
    local ok, r = coroutine.resume(t.co, table.unpack(t.args, 1, t.args.n))
    t.args = {{n = 0}}
    if not ok then
      done = true
      log[#log + 1] = t.name .. \"!\" .. (type(r) == \"string\" and r or type(r))
    elseif coroutine.status(t.co) == \"dead\" then
      log[#log + 1] = t.name .. \"=\" .. tostring(r)
    else
      tasks[#tasks + 1] = t
    end
  end
  for _, t in ipairs(tasks) do coroutine.close(t.co) end
  return steps, table.concat(log, \";\")
end"
        )
    }

    fn sc_vm(&mut self) -> String {
        let n = self.r.range(6, 20);
        let mut code = Vec::new();
        for _ in 0..n {
            let ins = match self.r.below(14) {
                0..=2 => format!("{{\"push\", {}}}", self.r.range(-3, 9)),
                3 | 4 => format!("{{\"load\", {}}}", self.r.range(1, 4)),
                5 => format!("{{\"store\", {}}}", self.r.range(1, 4)),
                6 => "{\"add\"}".into(),
                7 => "{\"sub\"}".into(),
                8 => "{\"mul\"}".into(),
                9 => "{\"mod\"}".into(),
                10 => "{\"dup\"}".into(),
                11 => format!("{{\"jz\", {}}}", self.r.range(-4, 3)),
                12 => format!("{{\"jmp\", {}}}", self.r.range(-5, 3)),
                _ => "{\"swap\"}".into(),
            };
            code.push(ins);
        }
        let prog = format!("local vm_code = {{{}}}", code.join(", "));
        let dispatch = *self.r.pick(&["if", "table", "goto"]);
        let lp = match dispatch {
            "table" => "local H = {
    push = function(x) sp = sp + 1 st[sp] = x end,
    load = function(x) sp = sp + 1 st[sp] = regs[x] end,
    store = function(x) regs[x] = st[sp] st[sp] = nil sp = sp - 1 end,
    add = function() st[sp - 1] = st[sp - 1] + st[sp] st[sp] = nil sp = sp - 1 end,
    sub = function() st[sp - 1] = st[sp - 1] - st[sp] st[sp] = nil sp = sp - 1 end,
    mul = function() st[sp - 1] = st[sp - 1] * st[sp] st[sp] = nil sp = sp - 1 end,
    mod = function() st[sp - 1] = st[sp - 1] % st[sp] st[sp] = nil sp = sp - 1 end,
    dup = function() st[sp + 1] = st[sp] sp = sp + 1 end,
    swap = function() st[sp], st[sp - 1] = st[sp - 1], st[sp] end,
    jz = function(x) local v = st[sp] st[sp] = nil sp = sp - 1 if v == 0 then pc = pc + x end end,
    jmp = function(x) pc = pc + x end,
  }
  while steps < 300 do
    steps = steps + 1
    local ins = vm_code[pc]
    if not ins then break end
    pc = pc + 1
    H[ins[1]](ins[2])
  end",
            "goto" => "::next::
  steps = steps + 1
  local ins = vm_code[pc]
  if steps > 300 or not ins then goto out end
  pc = pc + 1
  do
    local op, x = ins[1], ins[2]
    if op == \"push\" then sp = sp + 1 st[sp] = x goto next end
    if op == \"load\" then sp = sp + 1 st[sp] = regs[x] goto next end
    if op == \"store\" then regs[x] = st[sp] st[sp] = nil sp = sp - 1 goto next end
    if op == \"dup\" then st[sp + 1] = st[sp] sp = sp + 1 goto next end
    if op == \"swap\" then st[sp], st[sp - 1] = st[sp - 1], st[sp] goto next end
    if op == \"jz\" then local v = st[sp] st[sp] = nil sp = sp - 1 if v == 0 then pc = pc + x end goto next end
    if op == \"jmp\" then pc = pc + x goto next end
    local r = st[sp]
    st[sp] = nil
    sp = sp - 1
    if op == \"add\" then st[sp] = st[sp] + r elseif op == \"sub\" then st[sp] = st[sp] - r
    elseif op == \"mul\" then st[sp] = st[sp] * r else st[sp] = st[sp] % r end
  end
  goto next
  ::out::",
            _ => "while steps < 300 do
    steps = steps + 1
    local ins = vm_code[pc]
    if not ins then break end
    local op, x = ins[1], ins[2]
    pc = pc + 1
    if op == \"push\" then sp = sp + 1 st[sp] = x
    elseif op == \"load\" then sp = sp + 1 st[sp] = regs[x]
    elseif op == \"store\" then regs[x] = st[sp] st[sp] = nil sp = sp - 1
    elseif op == \"dup\" then st[sp + 1] = st[sp] sp = sp + 1
    elseif op == \"swap\" then st[sp], st[sp - 1] = st[sp - 1], st[sp]
    elseif op == \"jz\" then local v = st[sp] st[sp] = nil sp = sp - 1 if v == 0 then pc = pc + x end
    elseif op == \"jmp\" then pc = pc + x
    else
      local r = st[sp]
      st[sp] = nil
      sp = sp - 1
      if op == \"add\" then st[sp] = st[sp] + r elseif op == \"sub\" then st[sp] = st[sp] - r
      elseif op == \"mul\" then st[sp] = st[sp] * r else st[sp] = st[sp] % r end
    end
  end",
        };
        format!(
            "{prog}
local function vm_run(n, s)
  local st, sp, regs, pc, steps = {{}}, 0, {{n, #s, 0, 1}}, 1, 0
  local ok, e = pcall(function()
  {lp}
  end)
  local err = ok and \"ok\" or type(e) == \"string\" and e:gsub(\"^[^:]*:%d+: \", \"\") or type(e)
  return (regs[3] or 0) + (regs[1] or 0), err .. \":\" .. steps .. \":\" .. sp
end"
        )
    }

    fn sc_ser(&mut self) -> String {
        let data = *self.r.pick(&[
            "{n, s, {n * 2, x = s:upper(), [n % 3 + 1.5] = \"f\"}, {{}, {{n}}}}",
            "{[s] = n, list = {1, 2, nil, 4}, nested = {a = {b = {c = s .. \"\\0\"}}}, [-1] = true}",
            "{s:byte(1, -1)}",
            "{n, n + 0.5, -0.0, 1 / 0, math.mininteger, s:rep(2, \"\\n\")}",
            "setmetatable({x = n}, {__index = function() return \"no\" end})",
        ]);
        let back = *self.r.pick(&[
            "load(\"return \" .. enc)()",
            "load(\"local t = \" .. enc .. \" return t\", \"=ser\", \"t\", {})()",
        ]);
        format!(
            "local function ser_lt(x, y)
  local tx, ty = type(x), type(y)
  if tx ~= ty then return tx < ty end
  return x < y
end
local function ser_enc(v, depth)
  local t = type(v)
  if t == \"table\" then
    if depth > 6 then return \"...\" end
    local keys = {{}}
    for k in pairs(v) do keys[#keys + 1] = k end
    table.sort(keys, ser_lt)
    local parts = {{}}
    for _, k in ipairs(keys) do parts[#parts + 1] = \"[\" .. ser_enc(k, depth + 1) .. \"]=\" .. ser_enc(v[k], depth + 1) end
    return \"{{\" .. table.concat(parts, \",\") .. \"}}\"
  elseif t == \"string\" or t == \"number\" then
    return string.format(\"%q\", v)
  end
  return tostring(v)
end
local function ser_run(n, s)
  local enc = ser_enc({data}, 0)
  local ok, again = pcall(function() return ser_enc({back}, 0) end)
  return #enc, enc:sub(1, 40) .. (ok and again == enc and \"=\" or \"!\")
end"
        )
    }

    fn sc_vec(&mut self) -> String {
        let idx = *self.r.pick(&[
            "V3.__index = function(v, k) if k == \"len2\" then return v.x * v.x + v.y * v.y + v.z * v.z end return V3[k] end",
            "V3.__index = setmetatable({len2 = 0}, {__index = function(t, k) return rawget(V3, k) end})",
        ]);
        format!(
            "local V3 = {{}}
{idx}
local function vec(x, y, z) return setmetatable({{x = x, y = y, z = z}}, V3) end
V3.__add = function(p, q) return vec(p.x + q.x, p.y + q.y, p.z + q.z) end
V3.__sub = function(p, q) return vec(p.x - q.x, p.y - q.y, p.z - q.z) end
V3.__mul = function(p, q)
  if type(p) == \"number\" then return vec(p * q.x, p * q.y, p * q.z) end
  if type(q) == \"number\" then return vec(q * p.x, q * p.y, q * p.z) end
  return p.x * q.x + p.y * q.y + p.z * q.z
end
V3.__unm = function(p) return vec(-p.x, -p.y, -p.z) end
V3.__eq = function(p, q) return p.x == q.x and p.y == q.y and p.z == q.z end
V3.__len = function() return 3 end
V3.__concat = function(p, q)
  local function t(v) return type(v) == \"table\" and (v.x .. \",\" .. v.y .. \",\" .. v.z) or v end
  return t(p) .. \"|\" .. t(q)
end
V3.__newindex = function(v, k, val) error(\"frozen \" .. tostring(k), 0) end
function V3.cross(p, q) return vec(p.y * q.z - p.z * q.y, p.z * q.x - p.x * q.z, p.x * q.y - p.y * q.x) end
local function vec_run(n, s)
  local a, b = vec(n, 1, -2), vec(#s, n % 5, 3)
  local c = (a + b) * 2 - -a
  local ok, e = pcall(function() c.q = 1 end)
  local d = a:cross(b)
  return c * d + #c, (c .. d) .. (a == vec(n, 1, -2) and \"eq\" or \"ne\") .. tostring(ok) .. e
end"
        )
    }

    fn sc_memo(&mut self) -> String {
        let memo = *self.r.pick(&[
            "local function memoize(f)
  local cache = {}
  return function(...)
    local k = table.concat({...}, \",\")
    local v = cache[k]
    if v == nil then v = f(...) cache[k] = v end
    return v
  end
end",
            "local function memoize(f)
  local cache = setmetatable({}, {__index = function(t, n)
    local row = {}
    rawset(t, n, row)
    return row
  end})
  return function(n, k)
    k = k or 0
    local v = cache[n][k]
    if v == nil then v = f(n, k) cache[n][k] = v end
    return v
  end
end",
        ]);
        format!(
            "{memo}
local binom
binom = memoize(function(n, k) if k == 0 or k == n then return 1 end return binom(n - 1, k - 1) + binom(n - 1, k) end)
local fib
fib = memoize(function(n) if n < 2 then return n end return fib(n - 1) + fib(n - 2) end)
local parts
parts = memoize(function(n, k) if n == 0 then return 1 end if n < 0 or k == 0 then return 0 end return parts(n - k, k) + parts(n, k - 1) end)
local function memo_run(n, s)
  local m = n % 30
  return fib(m) + binom(m, m // 2) + parts(m % 20, m % 20), \"m\" .. m .. \":\" .. parts(#s % 12, #s % 12)
end"
        )
    }

    fn sc_lru(&mut self) -> String {
        let n = self.r.range(6, 20);
        let ops: Vec<String> = (0..n)
            .map(|_| {
                if self.r.chance(0.5) {
                    format!("{{\"p\", {}, {}}}", self.r.range(0, 6), self.r.range(0, 99))
                } else {
                    format!("{{\"g\", {}}}", self.r.range(0, 6))
                }
            })
            .collect();
        let style = *self.r.pick(&["closures", "methods"]);
        let new = if style == "closures" {
            "local function lru_new(cap)
  local head = {}
  head.next, head.prev = head, head
  local map, size = {}, 0
  local function unlink(e) e.prev.next, e.next.prev = e.next, e.prev end
  local function front(e) e.next, e.prev = head.next, head head.next.prev = e head.next = e end
  local c = {}
  function c.get(k) local e = map[k] if not e then return nil end unlink(e) front(e) return e.v end
  function c.put(k, v)
    local e = map[k]
    if e then e.v = v unlink(e) front(e) return end
    if size >= cap then local last = head.prev unlink(last) map[last.k] = nil size = size - 1 end
    e = {k = k, v = v}
    map[k] = e
    front(e)
    size = size + 1
  end
  function c.keys() local r = {} local e = head.next while e ~= head do r[#r + 1] = e.k e = e.next end return table.concat(r, \" \") end
  return c
end"
        } else {
            "local Lru = {}
Lru.__index = Lru
local function lru_new(cap)
  local head = {}
  head.next, head.prev = head, head
  local c = setmetatable({head = head, map = {}, size = 0, cap = cap}, Lru)
  c.get = function(k) return c:find(k) end
  c.put = function(k, v) return c:store(k, v) end
  c.keys = function() return c:list() end
  return c
end
function Lru:unlink(e) e.prev.next, e.next.prev = e.next, e.prev end
function Lru:front(e) local h = self.head e.next, e.prev = h.next, h h.next.prev = e h.next = e end
function Lru:find(k) local e = self.map[k] if not e then return nil end self:unlink(e) self:front(e) return e.v end
function Lru:store(k, v)
  local e = self.map[k]
  if e then e.v = v self:unlink(e) self:front(e) return end
  if self.size >= self.cap then local last = self.head.prev self:unlink(last) self.map[last.k] = nil self.size = self.size - 1 end
  e = {k = k, v = v}
  self.map[k] = e
  self:front(e)
  self.size = self.size + 1
end
function Lru:list() local r = {} local e = self.head.next while e ~= self.head do r[#r + 1] = e.k e = e.next end return table.concat(r, \" \") end"
        };
        format!(
            "{new}
local lru_ops = {{{}}}
local function lru_run(n, s)
  local c = lru_new(n % 4 + 2)
  local hits = 0
  for _, op in ipairs(lru_ops) do
    if op[1] == \"p\" then c.put(op[2] + n % 3, op[3] + #s) elseif c.get(op[2]) then hits = hits + 1 end
  end
  return hits, c.keys()
end",
            ops.join(", ")
        )
    }

    fn sc_events(&mut self) -> String {
        let handler = *self.r.pick(&[
            "function(m) return \"H\" end",
            "function(m) return type(m) end",
        ]);
        format!(
            "local function ev_on(bus, name, f, once)
  local l = bus[name] or {{}}
  bus[name] = l
  l[#l + 1] = {{f = f, once = once}}
end
local function ev_emit(bus, name, ...)
  local l = bus[name]
  if not l then return 0 end
  local n, i = 0, 1
  while i <= #l do
    local e = l[i]
    local ok, r = xpcall(e.f, {handler}, ...)
    if ok then n = n + (tonumber(r) or 0) else n = n + 100 end
    if e.once then table.remove(l, i) else i = i + 1 end
  end
  return n
end
local function events_run(n, s)
  local bus, count = {{}}, 0
  ev_on(bus, \"tick\", function(k) count = count + k return 1 end)
  ev_on(bus, \"tick\", function(k) if k == 3 then error(\"bad\") end return 2 end, true)
  ev_on(bus, \"tick\", function(...) return select(\"#\", ...) end)
  ev_on(bus, \"x\", function() ev_on(bus, \"tick\", function() return 7 end, true) return 0 end)
  local total = 0
  for k = 1, n % 6 + 2 do total = total + ev_emit(bus, k % 2 == 0 and \"tick\" or \"x\", k, s) end
  return total, \"c\" .. count .. \":\" .. #(bus.tick or {{}})
end"
        )
    }
}
