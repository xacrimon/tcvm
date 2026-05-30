//! End-to-end coverage for the Lua 5.5 named-vararg table, exercising both
//! the optimized below-base form and the materialized-table form, plus the
//! mutation-visibility semantics the table form must honor (manual §3.4).

use tcvm::{Executor, LoadError, Lua};

/// Compile and run `src`, returning the first integer result.
fn run(src: &str) -> i64 {
    let mut lua = Lua::new();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load(src, Some("vararg_test"))?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    lua.execute(&ex).expect("run")
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
    // The key semantic the old below-base-only design got wrong: once the
    // table is materialized, mutating it must be visible through a `...`
    // expression (which reads elements 1..t.n from the table).
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
