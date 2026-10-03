//! `string.format("%p")` (#243). Expected outputs from lua 5.5.1.

use crate::common::{err, ok};

#[test]
fn objects_and_non_objects() {
    // Address length is platform-dependent, so padding is checked against `%p`.
    assert_eq!(
        ok("local t = {} local p = string.format('%p', t) \
            return cat(p:match('^0x%x+$') ~= nil, \
              string.format('%20p', t) == string.rep(' ', 20 - #p) .. p, \
              string.format('%-20p', t) == p .. string.rep(' ', 20 - #p))"),
        "true true true"
    );
    assert_eq!(
        ok("return string.format('[%p] [%10p] [%-10p] [%p] [%p]', 1, nil, true, 2.5, false)"),
        "[(null)] [    (null)] [(null)    ] [(null)] [(null)]"
    );
    // The address is the one `tostring` shows; `__tostring` isn't called.
    assert_eq!(
        ok(
            "local t = setmetatable({}, {__tostring = function() return 'x' end}) \
            local co = coroutine.running() \
            return cat(string.format('%p', print) == tostring(print):match('0x%x+'), \
              string.format('%p', co) == tostring(co):match('0x%x+'), \
              string.format('%p', t) ~= 'x', \
              string.format('%p', 'abc') == string.format('%p', 'ab' .. 'c'))"
        ),
        "true true true true"
    );
}

#[test]
fn modifiers() {
    for spec in ["%0p", "%.3p", "%#p", "%+p"] {
        assert_eq!(
            err(&format!("string.format('{spec}', {{}})")),
            format!("c:1: invalid conversion specification: '{spec}'")
        );
    }
}
