//! Compiler limits surface as `load` errors. Expected strings come from `lua`
//! 5.5.1, minus its `near <token>` suffix (the compiler has no tokens, #221).
//! Lua's line can also be one later, from its lookahead token.

use crate::common::ok;

/// Chunk where the outermost function declares `n1` locals, the middle one
/// `n2`, and the innermost sums them all, so it needs `n1 + n2` upvalues.
/// One statement per read: a long `+` chain overflows a debug test stack.
const NESTED_UPVALUES: &str = r#"
    local function gen(n1, n2)
      local A, B, U = {}, {}, {}
      for i = 1, n1 do A[i] = "local a" .. i .. " = " .. i; U[#U+1] = "s = s + a" .. i end
      for i = 1, n2 do B[i] = "local b" .. i .. " = " .. i; U[#U+1] = "s = s + b" .. i end
      return table.concat(A, " ") .. "\nreturn function()\n" .. table.concat(B, " ") ..
        "\nreturn function()\nlocal s = 0 " .. table.concat(U, " ") .. "\nreturn s end end"
    end
"#;

#[test]
fn upvalues() {
    let run = |tail: &str| ok(&format!("{NESTED_UPVALUES}{tail}"));
    assert_eq!(
        run(r#"return cat(load(gen(128, 127), "=g")()()())"#),
        "16384"
    );
    assert_eq!(
        run(r#"return cat(load(gen(128, 128), "=g"))"#),
        "nil g:5: too many upvalues (limit is 255) in function at line 4"
    );
    // The middle function runs out while capturing for its child: the error
    // names the middle function, at the child's line.
    assert_eq!(
        ok(r#"local A, B, U = {}, {}, {}
              for i = 1, 128 do
                A[i] = "local a" .. i .. " = " .. i; B[i] = "local c" .. i .. " = " .. i
                U[#U+1] = "x = a" .. i; U[#U+1] = "x = c" .. i
              end
              return cat(load(table.concat(A, " ") .. "\nreturn function()\n" .. table.concat(B, " ") ..
                "\nreturn function()\nreturn function()\n" .. table.concat(U, " ") ..
                "\nend end end", "=mid"))"#),
        "nil mid:6: too many upvalues (limit is 255) in function at line 4"
    );
}

/// `names("a", 3)` is `"a1, a2, a3"`; `locals(2)` is `"local x1 local x2"`.
const GEN: &str = r#"
    local function names(p, n) local t = {} for i = 1, n do t[i] = p .. i end return table.concat(t, ", ") end
    local function locals(n) local t = {} for i = 1, n do t[i] = "local x" .. i end return table.concat(t, " ") end
    local function err(src) return select(2, load(src, "=c")) end
"#;

fn limit(tail: &str) -> String {
    ok(&format!("{GEN}{tail}"))
}

#[test]
fn registers() {
    // A local list longer than a `u8` used to wrap and panic.
    assert_eq!(
        limit(
            r#"return err("local " .. names("a", 260) .. "; return #{" .. names("a", 260) .. "}")"#
        ),
        "c:1: too many registers (limit is 255) in main function"
    );
    assert_eq!(
        limit(r#"return err("local a" .. string.rep(",a", 500) .. ";")"#),
        "c:1: too many registers (limit is 255) in main function"
    );
    assert_eq!(
        limit(r#"return err("a = f(x" .. string.rep(",x", 260) .. ")")"#),
        "c:1: too many registers (limit is 255) in main function"
    );
    // lua reports "C stack overflow" here: its parser recurses per name.
    assert_eq!(
        limit(r#"return err("global " .. names("g", 300) .. " = 1")"#),
        "c:1: too many registers (limit is 255) in main function"
    );
}

#[test]
fn local_variables() {
    assert_eq!(
        limit(r#"return cat(load(locals(200) .. " x200 = 200 return x200")())"#),
        "200"
    );
    assert_eq!(
        limit(r#"return err(locals(201))"#),
        "c:1: too many local variables (limit is 200) in main function"
    );
    // One past the register limit used to panic.
    assert_eq!(
        limit(r#"return err(locals(256))"#),
        "c:1: too many local variables (limit is 200) in main function"
    );
    assert_eq!(
        limit(r#"return err(locals(200) .. " local function f() end")"#),
        "c:1: too many local variables (limit is 200) in main function"
    );
    assert_eq!(
        limit(r#"return err("local function f(" .. names("p", 201) .. ") end")"#),
        "c:1: too many local variables (limit is 200) in function at line 1"
    );
    // The generic `for` holds 3 hidden locals; lua counts the same.
    assert_eq!(
        limit(r#"return cat(load(locals(196) .. " for k in pairs({}) do end return 1")())"#),
        "1"
    );
    assert_eq!(
        limit(r#"return err(locals(197) .. " for k in pairs({}) do end")"#),
        "c:1: too many local variables (limit is 200) in main function"
    );
    assert_eq!(
        limit(r#"return err("for " .. names("k", 300) .. " in pairs({}) do end")"#),
        "c:1: too many local variables (limit is 200) in main function"
    );
}

#[test]
fn folded_consts() {
    // A folded `<const>` owns no register, so it counts toward neither limit.
    assert_eq!(
        limit(
            r#"local t = {} for i = 1, 300 do t[i] = "local c" .. i .. " <const> = " .. i end
               return cat(load(table.concat(t, " ") .. " return c300")())"#
        ),
        "300"
    );
    assert_eq!(
        limit(r#"return cat(load(locals(200) .. " local k <const> = 1 return k")())"#),
        "1"
    );
    // lua says "too many registers": it folds only a list's last name.
    assert_eq!(
        limit(
            r#"local v = {} for i = 1, 300 do v[i] = i end
               return cat(load("local " .. names("c", 300):gsub(",", " <const>,") .. " <const> = " ..
                 table.concat(v, ", ") .. " return c300, c1")())"#
        ),
        "300 1"
    );
}

#[test]
fn returns() {
    assert_eq!(
        limit(r##"return cat(select("#", load("return 10" .. string.rep(",10", 253))()))"##),
        "254"
    );
    assert_eq!(
        limit(r#"return err("return 10" .. string.rep(",10", 254))"#),
        "c:1: too many returns (limit is 255) in main function"
    );
}
