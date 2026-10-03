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
