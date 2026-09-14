//! A `goto` that leaves a scope must CLOSE its captured / to-be-closed
//! locals, like `break` and the loop back-edges do. Regression for #135.

use tcvm::{Executor, LoadError, Lua};

fn run(src: &str) -> i64 {
    let mut lua = Lua::new();
    lua.load_all();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load(src, Some("goto_close"))?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    lua.execute(&ex).expect("run")
}

#[test]
fn backward_goto_out_of_block() {
    assert_eq!(
        run("local fns, i = {}, 0\n\
             ::top::\n\
             i = i + 1\n\
             do\n\
               local x = i\n\
               fns[i] = function() return x end\n\
               if i < 3 then goto top end\n\
             end\n\
             return tonumber(fns[1]() .. fns[2]() .. fns[3]())"),
        123
    );
}

#[test]
fn backward_goto_past_local_in_same_scope() {
    assert_eq!(
        run("local g, n = {}, 0\n\
             ::again::\n\
             n = n + 1\n\
             local y = n\n\
             g[n] = function() return y end\n\
             if n < 3 then goto again end\n\
             return tonumber(g[1]() .. g[2]() .. g[3]())"),
        123
    );
}

#[test]
fn forward_goto_out_of_block_then_register_reuse() {
    assert_eq!(
        run("local h\n\
             do\n\
               local z = 42\n\
               h = function() return z end\n\
               goto out\n\
             end\n\
             ::out::\n\
             local w = 7\n\
             return h() * 10 + w"),
        427
    );
}

#[test]
fn continue_idiom_keeps_per_iteration_capture() {
    assert_eq!(
        run("local c = {}\n\
             for k = 1, 4 do\n\
               local v = k\n\
               if k % 2 == 0 then goto continue end\n\
               c[#c + 1] = function() return v end\n\
               ::continue::\n\
             end\n\
             return c[1]() * 10 + c[2]()"),
        13
    );
}
