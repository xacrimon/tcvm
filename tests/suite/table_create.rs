//! `table.create`'s size limits (#238). Expected messages from lua 5.5.1,
//! apart from the function name (#186).

use crate::common::{err, ok};

#[test]
fn sizes_out_of_range() {
    for (args, msg) in [
        ("1 << 31", "bad argument #1 to 'create' (out of range)"),
        ("-1", "bad argument #1 to 'create' (out of range)"),
        (
            "math.mininteger",
            "bad argument #1 to 'create' (out of range)",
        ),
        ("0, 1 << 31", "bad argument #2 to 'create' (out of range)"),
        ("0, -1", "bad argument #2 to 'create' (out of range)"),
    ] {
        assert_eq!(
            err(&format!("local t = table.create({args})")),
            format!("c:1: {msg}"),
            "{args}"
        );
    }
}

#[test]
fn hash_part_overflow() {
    // Raised by the table itself, so without a position.
    assert_eq!(
        err("local t = table.create(0, (1 << 31) - 1)"),
        "table overflow"
    );
    assert_eq!(
        err("local t = table.create(0, (1 << 30) + 1)"),
        "table overflow"
    );
    assert_eq!(
        ok("local t = table.create(10, 1 << 4) return cat(#t, next(t), #table.create(0, nil))"),
        "0 nil 0"
    );
}

/// A table made with room for both kinds of keys, then filled past it.
#[test]
fn sized_tables_grow_past_their_hints() {
    let src = r#"
local out = {}
local t = table.create(5, 3)
for i = 1, 5 do t[i] = i * 10 end
t.a, t.b, t.c = 1, 2, 3
out[#out + 1] = cat(#t, t[5], t.c)
t[6] = 60 t.d = 4 t[0] = 0 t[1] = nil
out[#out + 1] = cat(t[6], t.d, t[0], t[1], t[2])
local keys = {}
for k, v in pairs(t) do keys[#keys + 1] = tostring(k) .. '=' .. v end
table.sort(keys)
out[#out + 1] = table.concat(keys, ',')
local u = table.create(3, 40)
u[3] = 3
for i = 1, 40 do u['k' .. i] = i end
local n, s = 0, 0
for k, v in pairs(u) do n = n + 1 s = s + v end
out[#out + 1] = cat(n, s, u.k40, next(table.create(100)))
return table.concat(out, ' | ')
"#;
    assert_eq!(
        ok(src),
        "5 50 3 | 60 4 0 nil 20 | 0=0,2=20,3=30,4=40,5=50,6=60,a=1,b=2,c=3,d=4 | 41 823 40 nil"
    );
}
