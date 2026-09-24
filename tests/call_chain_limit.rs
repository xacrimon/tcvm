//! `__call` chains: at most 15 hops, as the reference's `CIST_CCMT` counter
//! allows, and argument counts that outgrow the CALL instruction's 8 bits.
//! Expected strings come from `lua` 5.5.1 running the same chunks.

use tcvm::env::Value;
use tcvm::{Executor, LoadError, Lua, RuntimeError};

const PRELUDE: &str = "local function chain(n)
  local c = function(...) return select('#', ...) end
  for _ = 1, n do c = setmetatable({}, {__call = c}) end
  return c
end
";

fn run(body: &str) -> Result<String, String> {
    let src = format!("{PRELUDE}{body}");
    let mut lua = Lua::new();
    lua.load_all();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load(&src, Some("=c"))?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    let show = |v: Value<'_>| {
        let s = v.get_string().expect("string value");
        String::from_utf8_lossy(s.as_bytes()).into_owned()
    };
    match lua.finish(&ex) {
        Ok(()) => Ok(lua.enter(|ctx| show(ctx.fetch(&ex).take_result::<Value>(ctx).unwrap()))),
        Err(RuntimeError::Lua(e)) => Err(lua.enter(|ctx| show(ctx.fetch(&e).value()))),
        Err(e) => panic!("unexpected failure for {body:?}: {e:?}"),
    }
}

fn ok(body: &str) -> String {
    run(body).unwrap_or_else(|e| panic!("{body:?} raised {e:?}"))
}

fn err(body: &str) -> String {
    run(body).expect_err(body)
}

#[test]
fn fifteen_hops_resolve() {
    assert_eq!(ok("return tostring(chain(15)(1, 2))"), "17");
    let meta = "local a = setmetatable({}, {__add = chain(15)}) return tostring(a + 1)";
    assert_eq!(ok(meta), "17");
}

#[test]
fn sixteenth_hop_raises() {
    let msg = "c:6: '__call' chain too long";
    assert_eq!(err("local r = chain(16)(1, 2) return r"), msg);
    assert_eq!(err("return chain(16)(1, 2)"), msg);
    assert_eq!(
        err("local a = setmetatable({}, {__add = chain(16)}) return a + 1"),
        msg
    );
}

#[test]
fn sixteenth_hop_from_a_native_has_no_position() {
    assert_eq!(
        ok("local ok, e = pcall(chain(16)) return e"),
        "'__call' chain too long"
    );
    assert_eq!(
        ok("local ok, e = pcall(tostring, setmetatable({}, {__tostring = chain(16)})) return e"),
        "'__call' chain too long"
    );
}

#[test]
fn argument_count_past_255() {
    let args = (1..=252)
        .map(|i| i.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let tail = format!("local function g(c) return c({args}) end return tostring(g(chain(4)))");
    assert_eq!(ok(&tail), "256");
    let call = format!(
        "local function g(c) local r = c({args}) return r end return tostring(g(chain(4)))"
    );
    assert_eq!(ok(&call), "256");
}
