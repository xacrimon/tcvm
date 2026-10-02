//! The basic library's non-function globals, `_G` and `_VERSION`.
//! Expected output comes from `lua` 5.5.1 running the same chunk.

use crate::common::ok;

#[test]
fn g_and_version() {
    // `_G` is a plain field: a custom `_ENV` lacks it, and it can be cleared.
    assert_eq!(
        ok(
            r#"local r = {tostring(_G == _ENV), tostring(_G._G == _G), _VERSION,
                        tostring(load("return _G", "=e", "t", {})())}
              _G = nil
              r[#r + 1] = tostring(rawget(_ENV, "_G"))
              return table.concat(r, " ")"#
        ),
        "true true Lua 5.5 nil nil"
    );
}
