//! String parameters accept numbers, converted as `luaL_checkstring` does.
//! Expected strings come from `lua` 5.5.1 running the same chunk.

use std::path::PathBuf;

use crate::common::{err, ok};

#[test]
fn basic_and_string() {
    assert_eq!(
        ok("return cat(select('#', warn(1, 2.5)), string.rep('ab', 3, 0))"),
        "0 ab0ab0ab"
    );
}

#[test]
fn utf8() {
    assert_eq!(
        ok("return cat(utf8.len(123), utf8.codepoint(7), utf8.offset(42, 1))"),
        "3 55 1 1"
    );
    assert_eq!(
        ok(
            "local t = {} for p, c in utf8.codes(12) do t[#t + 1] = p .. ':' .. c end \
            return table.concat(t, ' ')"
        ),
        "1:49 2:50"
    );
}

#[test]
fn os() {
    assert_eq!(
        ok("return cat(os.getenv(1), os.remove(12345.5))"),
        "nil nil 12345.5: No such file or directory 2"
    );
    assert_eq!(
        ok("return cat(os.rename(98765, 4321))"),
        "nil No such file or directory 2"
    );
}

#[test]
fn io() {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("string_args");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("12345"), "l1\nl2\n").unwrap();
    // `p(12345)` is `dir .. 12345`: a number concatenated onto the path.
    let in_dir = |src: &str| {
        format!(
            "local dir = {:?} local function p(n) return dir .. n end {src}",
            format!("{}/", dir.display())
        )
    };
    assert_eq!(
        ok(&in_dir(
            "local f = io.open(p(12345)) local r = cat(f:read(1), f:seek('cur'), f:setvbuf('no')) \
             f:close() for l in io.lines(p(12345)) do r = r .. ' ' .. l end return r"
        )),
        "l 1 true l1 l2"
    );
    assert_eq!(
        ok(&in_dir(
            "io.input(p(12345)) local l = io.read('l') io.input(io.stdin) return l"
        )),
        "l1"
    );
    assert_eq!(
        err("local f = io.open('x', 1) return f"),
        "c:1: bad argument #2 to 'open' (invalid mode)"
    );
    // Only the reason: method calls' argument numbers and names are #186.
    assert_eq!(
        ok(
            "local function why(f, ...) return select(2, pcall(f, ...)):match('%(.*%)') end \
            return cat(why(io.stdin.seek, io.stdin, 5), why(io.stdout.setvbuf, io.stdout, 5))"
        ),
        "(invalid option '5') (invalid option '5')"
    );
}
