//! EQ/LT/LE semantics at the int/float boundary. Expected values were checked
//! against reference Lua 5.5 (`lua5.5`).

use tcvm::{Executor, LoadError, Lua};

fn run(src: &str) -> i64 {
    let mut lua = Lua::new();
    lua.load_all();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load(src, Some("compare_ops"))?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    lua.execute(&ex).expect("run")
}

/// Evaluates each expression and packs the booleans into a bit string.
fn truth(exprs: &[&str]) -> String {
    let body: String = exprs
        .iter()
        .map(|e| format!("r = r .. ((({e}) and \"1\") or \"0\")\n"))
        .collect();
    let bits = run(&format!(
        "local nan = 0/0\n\
         local big, bigf = 9007199254740993, 9007199254740992.0\n\
         local r = \"\"\n{body}return tonumber(r)"
    ));
    format!("{bits}")
}

#[test]
fn eq_across_int_float_divide() {
    // Bitwise Value equality would fail all of these.
    assert_eq!(
        truth(&[
            "1 == 1.0",
            "1.0 == 1",
            "0.0 == -0.0",
            "nan ~= nan",
            "big ~= bigf"
        ]),
        "11111"
    );
    assert_eq!(
        truth(&["1 ~= 1.0", "nan == nan", "big == bigf", "1 == '1'"]),
        "0"
    );
}

#[test]
fn ordering_is_exact_above_2_53() {
    // `big as f64` rounds to `bigf`, so a lossy cast would flip both of these.
    assert_eq!(truth(&["bigf < big", "big > bigf", "bigf <= big"]), "111");
    assert_eq!(truth(&["big <= bigf", "big < bigf", "bigf >= big"]), "0");
}

#[test]
fn ordering_against_out_of_range_floats_and_nan() {
    assert_eq!(
        truth(&[
            "math.maxinteger < 2^63",
            "math.maxinteger < math.huge",
            "-math.huge < math.mininteger",
            "math.mininteger <= -2^63",
            "2^63 - 1024 < math.maxinteger",
        ]),
        "11111"
    );
    assert_eq!(
        truth(&[
            "math.mininteger < -math.huge",
            "math.maxinteger <= 2^63 - 1024",
            "1 < nan",
            "nan < 1",
            "1 <= nan",
            "nan <= 1",
        ]),
        "0"
    );
}
