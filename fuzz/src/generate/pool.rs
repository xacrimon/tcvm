//! The value pool: the families of values a driver passes to `f`, `V.<family>` in a program.
//! `reset` in the prelude builds it anew for every unit.

/// Each family and its values, as Lua expressions; one may refer to a family before it.
const FAMILIES: &[(&str, &[&str])] = &[
    (
        "si",
        &["0", "1", "-1", "2", "3", "7", "-7", "100", "5", "16"],
    ),
    (
        "bi",
        &[
            "2147483647",
            "2147483648",
            "-2147483648",
            "-2147483649",
            "1 << 40",
            "math.maxinteger",
            "math.mininteger",
            "4294967296",
            "-(1 << 53)",
            "1 << 62",
        ],
    ),
    (
        "fl",
        &[
            "0.5", "-2.5", "3.75", "1.0", "-1.0", "2.0", "100.25", "1e15", "0.1", "7.0",
        ],
    ),
    (
        "fs",
        &[
            "0.0",
            "-0.0",
            "2.0 ^ 53",
            "2.0 ^ 63",
            "-(2.0 ^ 63)",
            "1 / 0",
            "-1 / 0",
            "0 / 0",
            "1e308",
            "2.0 ^ 31",
        ],
    ),
    // `fs` without the floats that `tostring` writes with more than 14 digits.
    (
        "fc",
        &[
            "0.0",
            "-0.0",
            "1 / 0",
            "-1 / 0",
            "0 / 0",
            "1e308",
            "2.0 ^ 31",
            "4294967296.0",
        ],
    ),
    (
        "ns",
        &[
            "\"10\"",
            "\"0x10\"",
            "\" 3 \"",
            "\"1e2\"",
            "\"-7\"",
            "\"2.5\"",
            "\"9223372036854775808\"",
            "\"0x7fffffffffffffff\"",
        ],
    ),
    (
        "st",
        &[
            "\"\"",
            "\"a\"",
            "\"abc\"",
            "\"hello world\"",
            "\"x\\0y\"",
            "\"10a\"",
            "\"B\"",
        ],
    ),
    ("ot", &["true", "false"]),
    (
        "tb",
        &[
            "mk(\"plain\", {})",
            "mk(\"arr\", { 10, 20, 30, n = 3 })",
            "mk(\"rec\", { x = 1, y = \"two\", z = 3.5 })",
            "mk(\"mt\", { 1, 2, x = 5 }, MT)",
            "mk(\"mti\", {}, MTI)",
            "mk(\"hole\", { 1, nil, 3 })",
            "mk(\"mix\", { 5, 6, x = 7, [2.5] = \"f\", [-1] = \"neg\" })",
        ],
    ),
    (
        "fn",
        &[
            "mk(\"f1\", function(...) return select(\"#\", ...) end)",
            "mk(\"f2\", function(a) return a end)",
            "mk(\"abs\", math.abs)",
        ],
    ),
    ("co", &["coroutine.create(function() end)"]),
    ("nil1", &["nil"]),
    ("sp", &["1", "2", "3", "5", "7"]),
    ("sn", &["-1", "-2", "-3"]),
    (
        "mx",
        &["1", "2.5", "\"10\"", "true", "3", "\"abc\"", "7.0", "false"],
    ),
    ("ms", &["1", "\"2\"", "3", "\"4.5\"", "5", "\"0x10\""]),
    ("fp", &["0.5", "1.5", "2.25", "3.0"]),
    ("mn", &["1", "2.5", "-3", "4.0", "7", "-0.5"]),
    ("ib", &["1", "2147483647", "-2147483648", "3", "1 << 40"]),
    ("tt", &["V.tb[2]", "V.tb[3]", "V.tb[7]"]),
    ("tm", &["V.tb[4]", "V.tb[5]"]),
    // Tables without holes, for code that takes lengths.
    (
        "sq",
        &[
            "mk(\"sq0\", {})",
            "mk(\"sq3\", { 4, 5, 6 })",
            "mk(\"sqs\", { \"p\", \"q\" })",
            "mk(\"sq1\", { 9 })",
        ],
    ),
];

/// How many values `family` has: `V.<family>[1]` to `V.<family>[size]`.
pub(super) fn size(family: &str) -> i64 {
    let f = FAMILIES.iter().find(|f| f.0 == family);
    f.expect("a family").1.len() as i64
}

/// The statements of `reset` that build the pool; a family's `n` is its size.
pub(super) fn lua() -> String {
    let mut out = String::from("V = {}\n");
    for (name, values) in FAMILIES {
        out.push_str(&format!(
            "  V.{name} = {{ n = {}, {} }}\n",
            values.len(),
            values.join(", ")
        ));
    }
    out
}
