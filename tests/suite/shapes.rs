//! The shape tree behind string-keyed fields. Expected results are lua
//! 5.5.1's, running the same chunks with `collectgarbage()` in place of the
//! host's collections.

use tcvm::Lua;

use crate::common::run_chunks;

fn run(src: &str) -> String {
    let mut lua = Lua::new();
    lua.load_all();
    run_chunks(&mut lua, src)
}

/// Defines `keys(t)`, the sorted `k=v` pairs `pairs` visits.
const KEYS: &str = "local function keys(t) local ks = {} \
    for k, v in pairs(t) do ks[#ks + 1] = k .. '=' .. v end \
    table.sort(ks) return table.concat(ks, ',') end ";

/// Shapes along a chain share one key array: a branch off it, a metatable
/// sibling, and a shape whose longer sibling died each add a key correctly.
#[test]
fn shared_keys() {
    let src = r#"
a = {}; a.x = 1; a.y = 2; a.z = 3
b = {}; b.x = 1; b.y = 2; b.w = 4
c = {}; c.x = 1; c.y = 2; setmetatable(c, {}); c.v = 5
e = {}; e.x = 1; e.y = 2; e.z = 3; setmetatable(e, {}); e.v = 6
a.u = 7
keep = {}; keep.p = 1; keep.q = 2
s = {}; s.p = 1; s.q = 2; s['dead' .. 1] = 3
s = nil
--gc
keep.r = 3
"#;
    let check = format!(
        "{KEYS}return table.concat({{keys(a), keys(b), keys(c), keys(e), keys(keep), \
         tostring(keep['dead' .. 1])}}, ' ')"
    );
    assert_eq!(
        run(&format!("{src}{check}")),
        "u=7,x=1,y=2,z=3 w=4,x=1,y=2 v=5,x=1,y=2 v=6,x=1,y=2,z=3 p=1,q=2,r=3 nil"
    );
}

/// A metatable whose tables are all dead is freed: the set-metatable edge on
/// their shape doesn't keep it, or the dead shape it led to, alive.
#[test]
fn dead_metatables_are_freed() {
    const N: usize = 2000;
    let live = |n: usize| {
        let mut lua = Lua::new();
        lua.load_all();
        run_chunks(
            &mut lua,
            &format!("for i = 1, {n} do setmetatable({{}}, {{}}) end\n--gc\n"),
        );
        lua.live_bytes()
    };
    let (none, many) = (live(0), live(N));
    assert!(many < none + 64 * N, "none={none} many={many}");
}

/// One shape with children along keys and metatables, some of them dropped
/// and added again.
#[test]
fn many_transitions() {
    let setup = r#"
mt1, mt2 = {}, {}
function make(k, mt)
  local t = {}; t.p = 1
  if k then t[k] = 2 end
  if mt then setmetatable(t, mt) end
  return t
end
function make2(k1, k2) local t = {}; t[k1] = 1; t[k2] = 2; return t end
first, lone = make('a'), make2('q', 'r')
ts = {make('b'), make(nil, mt1), make(nil, mt2), make('a', mt1), make('c')}
setmetatable(ts[2], nil)
--gc
first, lone = nil, nil
--gc
"#;
    let check = r#"
ts[6] = make('a')
ts[7] = make('d', mt2)
ts[8] = make(nil, mt1)
ts[9] = make2('q', 's')
local out = {}
for i, t in ipairs(ts) do
  local m = getmetatable(t)
  out[i] = keys(t) .. (m == mt1 and '/1' or m == mt2 and '/2' or '')
end
return table.concat(out, ' ')
"#;
    assert_eq!(
        run(&format!("{setup}{KEYS}{check}")),
        "b=2,p=1 p=1 p=1/2 a=2,p=1/1 c=2,p=1 a=2,p=1 d=2,p=1/2 p=1/1 q=1,s=2"
    );
}

/// Lookups past the length a shape scans go through its key array's index,
/// which a shorter shape sharing the array, a branch off it, and a shape whose
/// longer sibling died all read correctly.
#[test]
fn indexed_lookups() {
    let setup = r#"
function fill(t, prefix, n) for i = 1, n do t[prefix .. i] = i end return t end
a = fill({}, 'k', 40)
b = fill({}, 'k', 30)
early = tostring(rawget(b, 'k35')) .. ' ' .. b.k30
b.x = 1
keep = fill({}, 'd', 20)
gone = fill({}, 'd', 40)
probe = gone.d40 + keep.d20
gone = nil
--gc
"#;
    let check = r#"
local out = {early, probe, a.k1 + a.k40, tostring(a['k' .. 41]), tostring(b.k35), b.x}
for i = 21, 40 do
  if keep['d' .. i] ~= nil then out[#out + 1] = 'stale d' .. i end
end
keep.e = 1
keep['d' .. 21] = 21
out[#out + 1] = keys(keep)
return table.concat(out, ' ')
"#;
    assert_eq!(
        run(&format!("{setup}{KEYS}{check}")),
        "nil 30 60 41 nil nil 1 d10=10,d11=11,d12=12,d13=13,d14=14,d15=15,d16=16,d17=17,\
         d18=18,d19=19,d1=1,d20=20,d21=21,d2=2,d3=3,d4=4,d5=5,d6=6,d7=7,d8=8,d9=9,e=1"
    );
}

/// Constant-key stores keep a table in shape mode past 64 keys, where lookups
/// and `next` go through the key array's index, and move it to dict mode past
/// `MAX_PROPERTIES_FAST`; `t[k] = v` still does past 64.
#[test]
fn long_tables() {
    let mut src = String::from("t, big, keyed = {}, {}, {}\n");
    for i in 1..=100 {
        src += &format!("t.k{i} = {i}\n");
    }
    for i in 1..=520 {
        src += &format!("big.k{i} = {i}\n");
    }
    src += r#"
for i = 1, 70 do keyed['k' .. i] = i end
local n, sum = 0, 0
for k, v in pairs(t) do n = n + 1; sum = sum + v end
for k, v in pairs(t) do if v % 2 == 0 then t[k] = nil end end
local function count(t) local c = 0 for _ in pairs(t) do c = c + 1 end return c end
return table.concat({n, sum, count(t), t['k' .. 99], tostring(t['k' .. 100]),
  rawget(t, 'k1'), big.k1 + big.k520, big['k' .. 300], count(big),
  keyed.k70, count(keyed)}, ' ')
"#;
    assert_eq!(run(&src), "100 5050 50 99 nil 1 521 300 520 70 70");
}

/// A constructor's table starts in the shape of its constant fields, so a
/// field stored nil is a nil slot: invisible to `pairs` and `next`, as an
/// absent key is. Metamethod and `__mode` fields still take effect, and a
/// constructor past the shape-mode cap still holds all its fields.
#[test]
fn constructor_templates() {
    let weak = "weak = setmetatable({}, {__mode = 'k'})\nweak[{}] = 1\n--gc\n";
    let src = r#"
local none
local a = {x = 1, y = none, z = 3}
local b = {x = 1, x = 2}
local c = {p = none}
c.q = 1
local base = {get = function() return 'base' end}
local o = setmetatable({}, {__index = base})
local n = 0
for _ in pairs(weak) do n = n + 1 end
local d = {x = 1, y = 2}
d.y = nil
d.w = 4
local steps = {}
local k = next(a)
while k do steps[#steps + 1] = k; k = next(a, k) end
table.sort(steps)
return table.concat({keys(a), keys(b), keys(c), o.get(), n, keys(d),
  table.concat(steps, ','), tostring(rawget(a, 'y'))}, ' ')
"#;
    assert_eq!(
        run(&format!("{weak}{KEYS}{src}")),
        "x=1,z=3 x=2 q=1 base 0 w=4,x=1 x,z nil"
    );

    let fields: Vec<String> = (1..=600).map(|i| format!("k{i} = {i}")).collect();
    let src = format!(
        "local t = {{{}}} local n, sum = 0, 0 \
         for _, v in pairs(t) do n = n + 1; sum = sum + v end \
         return n .. ' ' .. sum .. ' ' .. t.k1 .. ' ' .. t.k600",
        fields.join(", ")
    );
    assert_eq!(run(&src), "600 180300 1 600");
}
