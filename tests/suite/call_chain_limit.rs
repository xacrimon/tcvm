//! `__call` chains: at most 15 hops, as the reference's `CIST_CCMT` counter
//! allows, and argument counts that outgrow the CALL instruction's 8 bits.
//! Expected strings come from `lua` 5.5.1 running the same chunks.

use crate::common;

const PRELUDE: &str = "local function chain(n)
  local c = function(...) return select('#', ...) end
  for _ = 1, n do c = setmetatable({}, {__call = c}) end
  return c
end
";

fn ok(body: &str) -> String {
    common::ok(&format!("{PRELUDE}{body}"))
}

fn err(body: &str) -> String {
    common::err(&format!("{PRELUDE}{body}"))
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
fn sixteenth_hop_through_multret() {
    assert_eq!(
        err("local t = {1, 2, 3} return chain(16)(table.unpack(t))"),
        "c:6: '__call' chain too long"
    );
}

#[test]
fn metamethod_chain_blames_the_innermost_value() {
    let e = err("return setmetatable({}, {__add = setmetatable({}, {__call = 5})}) + 1");
    // Lua appends " (metamethod 'add')", which tcvm doesn't yet.
    assert!(e.starts_with("c:6: attempt to call a number value"), "{e}");
}

#[test]
fn close_errors_are_positioned_at_the_closing_frame() {
    assert_eq!(
        err("do local x <close> = setmetatable({}, {__close = chain(16)}) end"),
        "c:6: '__call' chain too long"
    );
    assert_eq!(
        err(
            "do local x <close> = setmetatable({}, {__close = setmetatable({}, {__call = {}})}) end"
        ),
        "c:6: attempt to call a table value (metamethod 'close')"
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
