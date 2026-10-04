//! `<const>` locals whose value folds own no register (#291), so values after
//! one in the same list move down. Expected values come from `lua` 5.5.1.

use crate::common::ok;

#[test]
fn folded_target_mid_list() {
    // Values past the last target still run.
    assert_eq!(
        ok("local n = 0 local function f() n = n + 1 return 'f' end
            local a <const> = 1, f()
            local b, c <const>, d = f(), 'mid', f()
            return cat(a, b, c, d, n)"),
        "1 f mid f 3"
    );
    assert_eq!(
        ok("local e <const>, g, h <const> = 10 return cat(e, g, h)"),
        "10 nil nil"
    );
    assert_eq!(
        ok("local i <const>, j <const> = (function() return 1, 2 end)() return cat(i, j)"),
        "1 2"
    );
    assert_eq!(
        ok("local c1 <const>, c2 <const>, c3 <const> = 1, 'two', true return cat(c1, c2, c3)"),
        "1 two true"
    );
}

#[test]
fn folded_strings() {
    assert_eq!(
        ok("local s <const> = 'str'
            local function outer() return function() return s .. '!', #s end end
            return cat(outer()())"),
        "str! 3"
    );
    assert_eq!(
        ok("local t <const> = 'x' return cat(not t, t and 1, t or 2)"),
        "false 1 x"
    );
    assert_eq!(
        ok("local key <const> = 'kk' local tbl = {[key] = 1, kk2 = key}
            return cat(tbl.kk, tbl.kk2, key == 'kk', key .. key)"),
        "1 kk true kkkk"
    );
}

#[test]
fn folded_const_scopes() {
    assert_eq!(
        ok("local n = 0
            ::top:: local w <const> = 5
            n = n + 1 if n < 3 then goto top end
            return cat(n, w)"),
        "3 5"
    );
    // The closure captures `x`, which sits where `y` would have been.
    assert_eq!(
        ok("local fns = {}
            for z = 1, 2 do local y <const> = 'y' local x = z fns[z] = function() return x, y end end
            return cat(fns[1]()) .. ' ' .. cat(fns[2]())"),
        "1 y 2 y"
    );
}
