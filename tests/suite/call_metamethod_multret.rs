//! A MULTRET call (`c(...)`, `c(f())`) through a `__call` chain. The arg
//! count lives in `thread.top`, which the receiver shift must extend.
//! Regression for #46.

use tcvm::{Executor, LoadError, Lua};

fn run(src: &str) -> i64 {
    let mut lua = Lua::new();
    lua.load_all();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load(src, Some("call_mm_multret"))?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    lua.execute(&ex).expect("run")
}

const PRELUDE: &str = "local c = setmetatable({}, {__call = function(self, ...) return select('#', ...), ... end})\n\
                       local function pack(...) return ... end\n";

#[test]
fn spread_call_result() {
    assert_eq!(
        run(&format!(
            "{PRELUDE}local n, a, b, d = c(pack(1, 2, 3)) return n * 1000 + a * 100 + b * 10 + d"
        )),
        3123
    );
}

#[test]
fn spread_empty() {
    assert_eq!(run(&format!("{PRELUDE}return (c(pack()))")), 0);
}

#[test]
fn spread_vararg_and_fixed_prefix() {
    assert_eq!(
        run(&format!(
            "{PRELUDE}local function v(...) return c(...) end\n\
                      local n = v(9, 8)\n\
                      local m = c(1, pack(2, 3))\n\
                      return n * 10 + m"
        )),
        23
    );
}

#[test]
fn two_hop_chain() {
    assert_eq!(
        run(&format!(
            "{PRELUDE}local cc = setmetatable({{}}, {{__call = c}})\n\
                      local n, inner, x = cc(pack(7))\n\
                      return n * 10 + x + (inner == cc and 100 or 0)"
        )),
        127
    );
}
