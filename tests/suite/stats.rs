//! `Lua::set_stats` counts the interpreter's dispatches and the JIT's events
//! without changing what runs.

use tcvm::Lua;
use tcvm::env::Value;

use crate::common::start_on;

const SUM: &str = "local s = 0 for i = 1, 1000 do s = s + i end return tostring(s)";

/// `src`'s string result and the counts of running it.
fn counted(src: &str, jit: bool) -> (String, String) {
    let mut lua = Lua::new();
    lua.set_jit(jit);
    lua.set_stats(true);
    lua.load_all();
    let ex = start_on(&mut lua, src);
    lua.finish(&ex)
        .unwrap_or_else(|e| panic!("{src:?} failed: {e:?}"));
    let got = lua.enter(|ctx| {
        let v: Value = ctx.fetch(&ex).take_result(ctx).expect("one result");
        String::from_utf8_lossy(v.get_string().expect("a string result").as_bytes()).into_owned()
    });
    (got, lua.stats())
}

fn count(stats: &str, key: &str) -> u64 {
    (stats.lines())
        .find_map(|l| l.strip_prefix(key)?.strip_prefix(' ')?.parse().ok())
        .unwrap_or(0)
}

#[test]
fn interpreted_dispatches_are_counted() {
    // Too short to trip a hot counter, which dispatches its word again.
    let src = "local s = 0 for i = 1, 100 do s = s + i end return tostring(s)";
    let (got, stats) = counted(src, false);
    assert_eq!(got, "5050");
    assert_eq!(count(&stats, "op FORLOOP_I"), 100, "{stats}");
    assert_eq!(count(&stats, "op FORPREP"), 1, "{stats}");
    assert!(!stats.contains("jit "), "{stats}");
}

#[test]
fn compiled_code_dispatches_nothing() {
    let (got, stats) = counted(SUM, true);
    assert_eq!(got, "500500");
    // The lib's `jit` cfg (`build.rs`): no JIT on stable or on other targets.
    if cfg!(all(
        feature = "jit",
        feature = "nightly",
        any(target_arch = "aarch64", target_arch = "x86_64")
    )) {
        assert_eq!(count(&stats, "jit compiled FORLOOP_I"), 1, "{stats}");
        assert!(count(&stats, "op FORLOOP_I") < 1000, "{stats}");
    }
}

#[test]
fn nothing_is_counted_until_asked() {
    let mut lua = Lua::new();
    lua.load_all();
    let ex = start_on(&mut lua, SUM);
    lua.finish(&ex).unwrap();
    let stats = lua.stats();
    assert!(stats.contains("op FORLOOP_I 0\n"), "{stats}");
    assert!(stats.lines().all(|l| l.ends_with(" 0")), "{stats}");
}
