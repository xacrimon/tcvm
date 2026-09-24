//! TAILCALL shapes: argument/parameter mismatches, varargs on either side,
//! MULTRET arguments, captured locals under the overwritten arguments, a
//! callee window larger than the caller's, deep tail recursion, `__call`,
//! natives, and metamethod frames. The snapshot is `lua` 5.5.1's output.

use tcvm::env::LuaString;
use tcvm::{Executor, LoadError, Lua};

const CHUNK: &str = r##"local out = {}
local function p(...) local t = table.pack(...) for i = 1, t.n do t[i] = tostring(t[i]) end out[#out + 1] = table.concat(t, ",") end
local function id(...) return ... end
local function two(a, b) return a, b end
local function va(a, ...) return a, select("#", ...), ... end
-- fewer / more args than params, varargs on either side, MULTRET args
p((function() return two(1) end)())
p((function() return two(1, 2, 3) end)())
p((function(...) return va(...) end)(1, 2, 3))
p((function(...) return two(...) end)(7, 8, 9))
p((function(x, ...) return va(x, ...) end)(1))
p((function(...) local a, b = ... return id(b, a, ...) end)(4, 5, 6))
p((function() return id(id(1, 2), id(3, 4)) end)())
-- a closure captured a local that the tail call's args overwrite
local g
local function cap(a, b) g = function() return a, b end return two(b, a) end
p(cap(10, 20)); p(g())
local function cap2(a) local c = a * 2 g = function() return c end return va(a + 1, c) end
p(cap2(3)); p(g())
-- callee with a much larger window than the caller
local function big(x, ...) local a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31, a32, a33, a34, a35, a36, a37, a38, a39, a40, a41, a42, a43, a44, a45, a46, a47, a48, a49, a50, a51, a52, a53, a54, a55, a56, a57, a58, a59, a60, a61, a62, a63, a64, a65, a66, a67, a68, a69, a70, a71, a72, a73, a74, a75, a76, a77, a78, a79, a80, a81, a82, a83, a84, a85, a86, a87, a88, a89, a90, a91, a92, a93, a94, a95, a96, a97, a98, a99, a100, a101, a102, a103, a104, a105, a106, a107, a108, a109, a110, a111, a112, a113, a114, a115, a116, a117, a118, a119, a120, a121, a122, a123, a124, a125, a126, a127, a128, a129, a130, a131, a132, a133, a134, a135, a136, a137, a138, a139, a140, a141, a142, a143, a144, a145, a146, a147, a148, a149, a150, a151, a152, a153, a154, a155, a156, a157, a158, a159, a160, a161, a162, a163, a164, a165, a166, a167, a168, a169, a170, a171, a172, a173, a174, a175, a176, a177, a178, a179, a180 = ... return x, a1, a180, select('#', ...) end
p((function(x) return big(x, 2, 3) end)(1))
-- deep tail recursion runs in constant stack
local function loop(n, acc) if n == 0 then return acc end return loop(n - 1, acc + n) end
p(loop(1000000, 0))
local function vloop(n, ...) if n == 0 then return select("#", ...), ... end return vloop(n - 1, ...) end
p(vloop(100000, "x", "y"))
-- __call chain and natives in tail position
local callable = setmetatable({}, {__call = function(self, a, b) return "called", a, b end})
p((function() return callable(1, 2) end)())
p((function(x) return math.floor(x) end)(3.7))
p((function(...) return select(2, ...) end)(1, 2, 3))
p((function(t) return table.unpack(t) end)({1, 2, 3}))
p(pcall(function() return (nil)() end))
-- results adjusted by the original caller
local r1 = (function() return two(1, 2) end)()
local r2, r3, r4 = (function() return two(1, 2) end)()
p(r1, r2, r3, r4)
-- tail call from a metamethod frame
local mt = {__index = function(t, k) return id(k .. "!") end, __add = function(a, b) return two(3, 4) end}
local o = setmetatable({}, mt)
p(o.key, o + 1)
-- mutual recursion
local even, odd
function even(n) if n == 0 then return true end return odd(n - 1) end
function odd(n) if n == 0 then return false end return even(n - 1) end
p(even(100001))
-- natives in tail position: fast entries (hit and miss) and plain natives,
-- from vararg frames, frames with open upvalues, metamethod frames and
-- pcall'd frames, through __call, and with more results than RETURN encodes
p((function(...) return math.floor(...) end)(2.5), (function(...) return select(2, ...) end)(1, 2, 3))
local function upf(x) g = function() return x end return math.floor(x) end
p(upf(4.5)); p(g())
local function upn(x) g = function() return x end return select(2, x, x + 1) end
p(upn(7)); p(g())
p((function(x) return math.floor(x) end)("3.7"), (function(x) return math.floor(x) end)(2^40 + 0.5), (function(x) return math.abs(x) end)(math.mininteger))
local o2 = setmetatable({}, {__index = function(t, k) return math.floor(k) end, __lt = function() return math.abs(-1) end, __len = function(t) return select("#", t, t) end})
p(o2[5.5], o2 < o2, #o2)
p(pcall(function() return math.floor(1.5) end))
p(pcall(function() return select(2, "a", "b") end))
local tc = setmetatable({}, {__call = type})
p((function() return tc(1) end)(), (function(...) return tc(...) end)(1, 2))
local many = {} for i = 1, 300 do many[i] = i end
p(select("#", (function() return table.unpack(many) end)()), (select(300, (function() return table.unpack(many) end)())))
local s1, s2, s3 = (function() return math.floor(1.5) end)()
p(s1, s2, s3)
p(pcall(function() return math.floor({}) end))
return table.concat(out, "\n")
"##;

#[test]
fn tailcall_shapes() {
    let mut lua = Lua::new();
    lua.load_all();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load(CHUNK, Some("=t"))?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    lua.finish(&ex).expect("run");
    let out = lua.enter(|ctx| {
        let s = ctx
            .fetch(&ex)
            .take_result::<LuaString>(ctx)
            .expect("result");
        String::from_utf8_lossy(s.as_bytes()).into_owned()
    });
    insta::assert_snapshot!(out);
}
