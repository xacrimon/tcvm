//! Numeric `for` follows the reference `forprep`/`OP_FORLOOP` (lvm.c): an
//! integer init and step make an integer loop whose limit is floored/ceiled,
//! clamped, or found unreachable; the iteration count is precomputed so the
//! loop ends cleanly at either i64 boundary; anything else runs on floats.
//! Expected strings are `lua` 5.5.1's output for the same snippets.

use tcvm::{Executor, LoadError, Lua};

/// Run `body` as the body of a function receiving `p`, which records every
/// value it is handed. Returns `"<count> <v1>/<type> <v2>/<type> ..."`.
fn trace(body: &str) -> String {
    let src = format!(
        "local out, n = {{}}, 0\n\
         local function p(v)\n\
           n = n + 1\n\
           if n <= 6 then out[#out + 1] = tostring(v) .. '/' .. math.type(v) end\n\
         end\n\
         {body}\n\
         return n .. (#out > 0 and (' ' .. table.concat(out, ' ')) or '')"
    );
    let mut lua = Lua::new();
    lua.load_all();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load(&src, Some("=c"))?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    lua.finish(&ex).expect("run");
    lua.try_enter(|ctx| {
        let ex = ctx.fetch(&ex);
        let s = ex.take_result::<tcvm::env::LuaString>(ctx)?;
        Ok::<_, tcvm::RuntimeError>(String::from_utf8_lossy(s.as_bytes()).into_owned())
    })
    .expect("result")
}

#[test]
fn integer_loop_with_float_limit() {
    assert_eq!(trace("for i = 1, 2.5 do p(i) end"), "2 1/integer 2/integer");
    assert_eq!(
        trace("for i = 3, 1.5, -1 do p(i) end"),
        "2 3/integer 2/integer"
    );
    // Numeric strings coerce the same way: floor or ceil, then integer.
    assert_eq!(trace("for i = 1, '2' do p(i) end"), "2 1/integer 2/integer");
    assert_eq!(
        trace("for i = 1, '2.5' do p(i) end"),
        "2 1/integer 2/integer"
    );
    assert_eq!(
        trace("for i = 3, '1.5', -1 do p(i) end"),
        "2 3/integer 2/integer"
    );
}

#[test]
fn integer_loop_reaches_the_boundaries() {
    assert_eq!(
        trace("for i = math.maxinteger - 1, math.maxinteger do p(i) end"),
        "2 9223372036854775806/integer 9223372036854775807/integer"
    );
    assert_eq!(
        trace("for i = math.mininteger + 1, math.mininteger, -1 do p(i) end"),
        "2 -9223372036854775807/integer -9223372036854775808/integer"
    );
    assert_eq!(
        trace("for i = 10, 0, math.mininteger do p(i) end"),
        "1 10/integer"
    );
    assert_eq!(
        trace("for i = 0, 10, math.maxinteger do p(i) end"),
        "1 0/integer"
    );
    assert_eq!(
        trace("for i = math.mininteger, math.maxinteger, math.maxinteger do p(i) end"),
        "3 -9223372036854775808/integer -1/integer 9223372036854775806/integer"
    );
    assert_eq!(
        trace("for i = 1, 10, 3 do p(i) end"),
        "4 1/integer 4/integer 7/integer 10/integer"
    );
    assert_eq!(
        trace("for i = 10, 1, -3 do p(i) end"),
        "4 10/integer 7/integer 4/integer 1/integer"
    );
}

#[test]
fn out_of_range_limits_clamp_or_skip() {
    assert_eq!(trace("for i = 1, 1e100 do p(i) break end"), "1 1/integer");
    assert_eq!(trace("for i = 1, 1e100, -1 do p(i) end"), "0");
    assert_eq!(trace("for i = 1, -1e100 do p(i) end"), "0");
    assert_eq!(
        trace("for i = math.mininteger + 1, -1e100, -1 do p(i) end"),
        "2 -9223372036854775807/integer -9223372036854775808/integer"
    );
    // NaN counts as too negative, so it skips ascending and clamps descending.
    assert_eq!(trace("for i = 1, 0/0 do p(i) end"), "0");
    assert_eq!(trace("for i = 1, 0/0, -1 do p(i) break end"), "1 1/integer");
}

#[test]
fn float_loops() {
    assert_eq!(
        trace("for i = 1, 3, 1.0 do p(i) end"),
        "3 1.0/float 2.0/float 3.0/float"
    );
    assert_eq!(
        trace("for i = 1.0, 3 do p(i) end"),
        "3 1.0/float 2.0/float 3.0/float"
    );
    assert_eq!(
        trace("for i = 1, 2, '1' do p(i) end"),
        "2 1.0/float 2.0/float"
    );
    assert_eq!(
        trace("for i = 1, 2, 0.5 do p(i) end"),
        "3 1.0/float 1.5/float 2.0/float"
    );
    assert_eq!(trace("for i = 2.0, 1 do p(i) end"), "0");
    assert_eq!(
        trace("for i = 1.0, math.huge do p(i) if i >= 3 then break end end"),
        "3 1.0/float 2.0/float 3.0/float"
    );
}

#[test]
fn loop_variable_is_a_fresh_upvalue_per_iteration() {
    // The visible variable shares the control register; closures capturing it
    // must still see their own iteration's value after the loop moves on.
    assert_eq!(
        trace(
            "local fs = {}\n\
             for i = 1, 3 do fs[i] = function() return i end end\n\
             for _, f in ipairs(fs) do p(f()) end"
        ),
        "3 1/integer 2/integer 3/integer"
    );
}
