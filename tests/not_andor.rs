//! Regression for issue #35: `not` of a parenthesized `and`/`or` dropped the
//! operand's short-circuit jump list, miscompiling the value (and emitting an
//! unpatched `TESTSET <NO_REG>` / `JMP +0`). Each case is checked against the
//! result `lua5.5` produces.

use tcvm::{Executor, LoadError, Lua};

fn eval_bool(src: &str) -> bool {
    let mut lua = Lua::new();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load(src, Some("not_andor"))?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    lua.execute::<bool>(&ex).expect("run")
}

fn eval_int(src: &str) -> i64 {
    let mut lua = Lua::new();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load(src, Some("not_andor"))?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    lua.execute::<i64>(&ex).expect("run")
}

#[test]
fn not_of_and_falsy_lhs() {
    // a falsy => (a and b) falsy => not(...) true
    assert!(eval_bool("local a, b = false, 2; return not (a and b)"));
}

#[test]
fn not_of_and_truthy() {
    // both truthy => (a and b) == b (truthy) => not(...) false
    assert!(!eval_bool("local a, b = 1, 2; return not (a and b)"));
}

#[test]
fn not_of_or_all_falsy() {
    assert!(eval_bool("local a, b = false, false; return not (a or b)"));
}

#[test]
fn not_of_or_truthy_lhs() {
    assert!(!eval_bool("local a, b = 1, 2; return not (a or b)"));
}

#[test]
fn not_of_and_with_const_rhs_falsy_lhs() {
    // a falsy => (a and 5) falsy => not(...) true. Pre-fix this folded to a
    // bogus constant `false`.
    assert!(eval_bool("local a = false; return not (a and 5)"));
}

#[test]
fn not_of_and_with_const_rhs_truthy() {
    assert!(!eval_bool("local a = 1; return not (a and 5)"));
}

#[test]
fn not_of_chained_and() {
    assert!(eval_bool(
        "local a, b, c = 1, nil, 3; return not (a and b and c)"
    ));
    assert!(!eval_bool(
        "local a, b, c = 1, 2, 3; return not (a and b and c)"
    ));
}

#[test]
fn not_of_andor_in_branch() {
    // `if not (a and b)` must take the then-arm when a is falsy.
    assert!(eval_bool(
        "local a, b = false, 2; if not (a and b) then return true else return false end"
    ));
    assert!(!eval_bool(
        "local a, b = 1, 2; if not (a and b) then return true else return false end"
    ));
}

#[test]
fn not_of_or_with_comparison_tail() {
    // `not (a or (b<c))`: the `or`'s value-preserving TESTSET must be
    // downgraded to a TEST by the `not`, else the expression yields the
    // operand value instead of a boolean. a truthy => short-circuits to a
    // => not(a) false.
    assert!(!eval_bool(
        "local a, b, c = 2, 1, 9; return not (a or (b < c))"
    ));
    // a falsy => evaluate b<c. 1<9 true => not(true) false.
    assert!(!eval_bool(
        "local a, b, c = false, 1, 9; return not (a or (b < c))"
    ));
    // a falsy, 9<1 false => not(false) true.
    assert!(eval_bool(
        "local a, b, c = false, 9, 1; return not (a or (b < c))"
    ));
    // projected through a further `and`/`or` (the shape the sweep caught):
    assert_eq!(
        eval_int("local a, b, c = 2, 1, 9; return (not (a or (b < c))) and 1 or 0"),
        0
    );
}

#[test]
fn not_of_and_with_comparison_tail() {
    // a truthy, b<c true => `a and (b<c)` true => not(...) false.
    assert!(!eval_bool(
        "local a, b, c = 1, 1, 9; return not (a and (b < c))"
    ));
    // a falsy => `a and (b<c)` false => not(...) true.
    assert!(eval_bool(
        "local a, b, c = false, 1, 9; return not (a and (b < c))"
    ));
}

#[test]
fn neg_of_and_does_not_misfold() {
    // `-(a and 5)` with a truthy => -(5) == -5. (Pre-fix folded regardless
    // of the short-circuit list; the falsy-lhs path would have been dropped.)
    let mut lua = Lua::new();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load("local a = 1; return -(a and 5)", Some("neg_andor"))?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    assert_eq!(lua.execute::<i64>(&ex).expect("run"), -5);
}
