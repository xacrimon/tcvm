//! Regression for the int/float ordering fix: mixed `<`/`<=` (and the stdlib
//! functions that share the path — `math.max`/`math.min` and `table.sort`'s
//! default comparator) compared the `i64` operand via a lossy `as f64` cast.
//! Near `i64::MAX` that cast rounds `maxinteger` up to exactly `2^63`, so
//! `maxinteger < 2.0^63` wrongly became `2^63 < 2^63` (false). Every result
//! below was checked against `lua` 5.5.0.

use tcvm::{Executor, LoadError, Lua};

fn eval_bool(src: &str) -> bool {
    let mut lua = Lua::new();
    lua.load_all(); // math / table globals
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load(src, Some("intfloat_compare"))?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    lua.execute::<bool>(&ex).expect("run")
}

#[test]
fn maxinteger_vs_two_pow_63() {
    // The core repro: 2^63 is one past i64::MAX, so an exact compare must treat
    // maxinteger as strictly below it.
    assert!(eval_bool("return math.maxinteger < (2.0^63)"));
    assert!(eval_bool("return math.maxinteger <= (2.0^63)"));
    assert!(eval_bool("return not ((2.0^63) < math.maxinteger)"));
    assert!(eval_bool("return not ((2.0^63) <= math.maxinteger)"));
}

#[test]
fn mininteger_boundary_is_exact() {
    // mininteger == -2^63 exactly, so the negative boundary must compare equal,
    // not "less than every int" (guards the strict `< -2^63` range check).
    assert!(eval_bool("return not (math.mininteger < -(2.0^63))"));
    assert!(eval_bool("return math.mininteger <= -(2.0^63)"));
    assert!(eval_bool("return not (-(2.0^63) < math.mininteger)"));
    assert!(eval_bool("return -(2.0^63) <= math.mininteger"));
}

#[test]
fn nan_compares_false_both_directions() {
    assert!(eval_bool("local n = 0/0; return not (math.maxinteger < n)"));
    assert!(eval_bool("local n = 0/0; return not (n < math.maxinteger)"));
    assert!(eval_bool(
        "local n = 0/0; return not (math.maxinteger <= n)"
    ));
    assert!(eval_bool(
        "local n = 0/0; return not (n <= math.maxinteger)"
    ));
}

#[test]
fn infinity_ordering() {
    assert!(eval_bool("return math.maxinteger < (1/0)"));
    assert!(eval_bool("return not ((1/0) < math.maxinteger)"));
    assert!(eval_bool("return not (math.mininteger < (-1/0))"));
    assert!(eval_bool("return (-1/0) < math.mininteger"));
}

#[test]
fn small_non_integral_floats_unaffected() {
    assert!(eval_bool("return 3 < 3.5"));
    assert!(eval_bool("return not (4 < 3.5)"));
    assert!(eval_bool("return 3.5 < 4"));
    assert!(eval_bool("return 4 <= 4.0"));
}

#[test]
fn math_max_min_preserve_type() {
    // max(maxinteger, 2^63) must yield the float 2^63 (it is the greater value),
    // not the integer — the lossy cast made them compare equal and kept the int.
    assert!(eval_bool(
        "return math.type(math.max(math.maxinteger, 2.0^63)) == 'float'"
    ));
    assert!(eval_bool(
        "return math.type(math.max(math.maxinteger, math.maxinteger - 1.0)) == 'float'"
    ));
    // Tie keeps the first-seen `best`: min(mininteger, -2^63) stays the integer.
    assert!(eval_bool(
        "return math.type(math.min(math.mininteger, -(2.0^63))) == 'integer'"
    ));
}

#[test]
fn mixed_int_float_equality() {
    // op_eq used a bitwise `Value` compare, so an int never equalled an
    // integer-valued float. raw_eq fixes the value compare across the divide.
    assert!(eval_bool("return 1 == 1.0"));
    assert!(eval_bool("return 0 == -0.0 and -0.0 == 0"));
    assert!(eval_bool("return 2^53 == (1 << 53)"));
    assert!(eval_bool("return not (math.maxinteger == 2.0^63)"));
    // 2^53+1 is not float-representable, so it must not compare equal.
    assert!(eval_bool("return not ((1 << 53) + 1 == 2.0^53)"));
    // Identity/content equality of non-numbers is unchanged.
    assert!(eval_bool("local t = {}; return t == t and not (t == {})"));
    assert!(eval_bool("return 'x' == 'x' and not (1 == 2)"));
}

#[test]
fn table_sort_orders_across_int_float_boundary() {
    // {2^63, maxinteger} sorts to [maxinteger(int), 2^63(float)] since
    // maxinteger < 2^63; pre-fix the equal-compare left the order reversed.
    assert!(eval_bool(
        "local t = {2.0^63, math.maxinteger}; table.sort(t)\n\
         return math.type(t[1]) == 'integer' and math.type(t[2]) == 'float'"
    ));
}
