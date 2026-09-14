//! `source:line:` prefixes on raised errors follow `luaL_where`: `error`'s
//! level (default 1) and library errors name the Lua frame at that level,
//! nothing is added when that level isn't a Lua function or the value isn't
//! a string. Expected strings were produced by `lua` 5.5.1 on equivalent
//! chunks named `=t`.

use tcvm::{Executor, LoadError, Lua, RuntimeError};

enum Raised {
    Str(String),
    Int(i64),
    Other(&'static str),
}

fn raise(src: &str) -> Raised {
    let mut lua = Lua::new();
    lua.load_all();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load(src, Some("=t"))?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    match lua.execute::<()>(&ex) {
        Err(RuntimeError::Lua(stashed)) => lua.enter(|ctx| {
            let v = ctx.fetch(&stashed).value();
            if let Some(s) = v.get_string() {
                Raised::Str(String::from_utf8_lossy(s.as_bytes()).into_owned())
            } else if let Some(i) = v.get_integer() {
                Raised::Int(i)
            } else {
                Raised::Other(v.type_name())
            }
        }),
        other => panic!("expected a Lua error, got {other:?}"),
    }
}

fn raise_str(src: &str) -> String {
    match raise(src) {
        Raised::Str(s) => s,
        _ => panic!("expected a string error value"),
    }
}

fn run_i64(src: &str) -> i64 {
    let mut lua = Lua::new();
    lua.load_all();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load(src, Some("=t"))?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    lua.execute(&ex).expect("run")
}

#[test]
fn error_default_level_names_caller() {
    assert_eq!(raise_str("error(\"m\")"), "t:1: m");
    assert_eq!(raise_str("\n\nerror(\"m\")"), "t:3: m");
}

#[test]
fn error_level_zero_is_verbatim() {
    assert_eq!(raise_str("error(\"m\", 0)"), "m");
}

#[test]
fn error_level_two_names_the_callers_caller() {
    assert_eq!(
        raise_str("local function f()\n  error(\"m\", 2)\nend\n\nf()"),
        "t:5: m"
    );
    // Level 2 from the main chunk has no Lua frame above it.
    assert_eq!(raise_str("error(\"m\", 2)"), "m");
}

#[test]
fn non_string_values_get_no_prefix() {
    assert!(matches!(raise("error(42)"), Raised::Int(42)));
    assert!(matches!(raise("error({})"), Raised::Other("table")));
}

#[test]
fn assert_raises_like_error() {
    assert_eq!(raise_str("assert(false, \"m\")"), "t:1: m");
    assert_eq!(raise_str("\nassert(false)"), "t:2: assertion failed!");
    assert!(matches!(raise("assert(nil, 42)"), Raised::Int(42)));
}

#[test]
fn library_errors_name_the_calling_lua_frame() {
    let msg = raise_str("local function f()\n  local s = string.rep()\nend\nf()");
    assert!(msg.starts_with("t:2: bad argument #1 to 'rep'"), "{msg}");
    // A tail-called native still reports the tail-calling frame.
    let msg = raise_str("local function f()\n  return string.rep()\nend\nf()");
    assert!(msg.starts_with("t:2: bad argument #1 to 'rep'"), "{msg}");
}

#[test]
fn bad_level_argument() {
    assert_eq!(
        raise_str("error(\"x\", \"y\")"),
        "t:1: bad argument #2 to 'error' (number expected, got string)"
    );
}

#[test]
fn coroutine_errors_are_positioned_in_their_own_thread() {
    assert_eq!(
        run_i64(
            "local co = coroutine.create(function()\n\
               error(\"c\")\n\
             end)\n\
             local ok, e = coroutine.resume(co)\n\
             local co2 = coroutine.create(function() error(\"d\", 2) end)\n\
             local ok2, e2 = coroutine.resume(co2)\n\
             return (not ok and e == \"t:2: c\" and not ok2 and e2 == \"d\") and 1 or 0"
        ),
        1
    );
}
