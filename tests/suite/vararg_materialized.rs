//! End-to-end coverage for the Lua 5.5 named-vararg table, exercising both
//! the optimized below-base form and the materialized-table form, plus the
//! mutation-visibility semantics the table form must honor (manual §3.4).

use crate::common::{err, eval, ok};

/// Compile and run `src`, returning the first integer result.
fn run(src: &str) -> i64 {
    eval(src)
}

#[test]
fn optimized_index_and_n() {
    // `args` is used only as the base of `args[exp]` / `args.n`, so it stays
    // optimized (below-base reads via VARARGGET, no table built).
    assert_eq!(
        run(
            "local function f(...args) return args[1] + args[2] + args.n end\n\
             return f(5, 6, 7)"
        ),
        5 + 6 + 3
    );
}

#[test]
fn materialized_when_used_as_value() {
    // Binding `args` to a local forces materialization; index sites are
    // rewritten VARARGGET -> GETTABLE and read the same table.
    assert_eq!(
        run("local function unwrap(t) return t[1] end\n\
             local function f(...args) return unwrap(args) + args[1] end\n\
             return f(40, 2, 3)"),
        40 + 40
    );
}

#[test]
fn materialized_when_captured() {
    // Upvalue capture by a nested closure also forces materialization.
    assert_eq!(
        run(
            "local function mk(...args) return function() return args[1] + args[2] end end\n\
             return mk(100, 200, 300)()"
        ),
        300
    );
}

#[test]
fn vararg_expr_reflects_table_mutation() {
    // Once materialized, mutations must be visible through `...` (which reads
    // elements 1..t.n from the table), not just through `args[i]`.
    assert_eq!(
        run("local function first(a) return a end\n\
             local function f(...args)\n\
                 local t = args   -- escape -> materialized\n\
                 t[1] = 99        -- mutate the shared table\n\
                 return first(...)\n\
             end\n\
             return f(10, 20, 30)"),
        99
    );
}

#[test]
fn vararg_expr_reflects_n_mutation() {
    // `...` honors a mutated `n`: shrinking `n` truncates the spread.
    assert_eq!(
        run("local function count(...) local t = {...} return #t end\n\
             local function f(...args)\n\
                 local t = args\n\
                 t.n = 1          -- only the first element should spread\n\
                 return count(...)\n\
             end\n\
             return f(7, 8, 9)"),
        1
    );
}

#[test]
fn anonymous_vararg_still_works() {
    // Anonymous `...` (no named param, never materialized) regression guard.
    assert_eq!(
        run("local function f(...) return ... end\n\
             local function sum(a, b, c) return a + b + c end\n\
             return sum(f(1, 2, 3))"),
        6
    );
}

#[test]
fn materialized_zero_extras() {
    // Materialized with no extra args: empty table, n == 0.
    assert_eq!(
        run("local function f(...args) local t = args return t.n end\n\
             return f()"),
        0
    );
}

/// A named vararg *after* named parameters. `adjust_locals` advances `nactvar`
/// rather than setting it, so passing `num_params + 1` double-counted the
/// parameters and tripped its own assertion — but only when there was at least
/// one named parameter, which no test had combined with a named vararg.
#[test]
fn named_vararg_after_named_params() {
    assert_eq!(
        run(
            "local function f(a, b, ...rest) return a + b + rest.n end\n\
             return f(1, 2, 30, 40, 50)"
        ),
        1 + 2 + 3
    );
    assert_eq!(
        run(
            "local function f(a, ...rest) return a + rest[1] + rest[2] end\n\
             return f(1, 20, 300)"
        ),
        1 + 20 + 300
    );
    // Materialized form, with parameters preceding the vararg.
    assert_eq!(
        run("local function unwrap(t) return t[1] end\n\
             local function f(a, b, ...rest) return a + b + unwrap(rest) end\n\
             return f(1, 2, 300)"),
        1 + 2 + 300
    );
}

#[test]
fn spread_checks_n() {
    // `...` validates the table's `n`, even for a fixed count, and reads
    // nothing past it (#241). Expected messages from lua 5.5.1.
    for n in [
        "-1",
        "1.0",
        "'2'",
        "nil",
        "math.maxinteger",
        "math.mininteger",
        "1 << 30",
    ] {
        assert_eq!(
            err(&format!(
                "local function f(...t) t.n = {n}; return ... end return f(1, 2)"
            )),
            "c:1: vararg table has no proper 'n'",
            "n = {n}"
        );
    }
    assert_eq!(
        err("local function f(...t) t.n = -1; local a, b = ...; return a end return f(1)"),
        "c:1: vararg table has no proper 'n'"
    );
    // In range but past the stack limit.
    assert_eq!(
        err("local function f(...t) t.n = (1 << 30) - 1; return ... end return f()"),
        "c:1: stack overflow"
    );
    assert_eq!(
        ok("local function f(...t) t.n = 1; local a, b = ...; return cat(a, b) end return f(5, 6)"),
        "5 nil"
    );
    assert_eq!(
        ok("local function f(...t) t.n = 4; t[4] = 'd'; return cat(...) end return f('a', 'b')"),
        "a b nil d"
    );
}
