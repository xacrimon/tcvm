//! `#t` must return a border after array slots are cleared (manual §3.4.7).
//! Regression for #97.

use tcvm::{Executor, LoadError, Lua};

fn run(src: &str) -> i64 {
    let mut lua = Lua::new();
    lua.load_all();
    let ex = lua
        .try_enter(|ctx| -> Result<_, LoadError> {
            let chunk = ctx.load(src, Some("table_border"))?;
            Ok(ctx.stash(Executor::start(ctx, chunk, ())))
        })
        .expect("load");
    lua.execute(&ex).expect("run")
}

#[test]
fn clearing_last_slot_shrinks() {
    assert_eq!(run("local t = {1, 2, 3} t[3] = nil return #t"), 2);
}

#[test]
fn clearing_run_of_trailing_slots_shrinks_past_all() {
    assert_eq!(
        run("local t = {1, 2, 3} t[2] = nil t[3] = nil return #t"),
        1
    );
}

#[test]
fn nil_write_past_end_is_noop() {
    assert_eq!(run("local t = {} t[5] = nil return #t"), 0);
}

#[test]
fn table_remove_shrinks() {
    assert_eq!(
        run("local t = {1, 2, 3, 4, 5} table.remove(t) table.remove(t) return #t"),
        3
    );
}

#[test]
fn regrows_after_shrink() {
    assert_eq!(
        run("local t = {1, 2, 3} t[3] = nil t[3] = 9 return #t + t[3]"),
        12
    );
}

#[test]
fn large_positive_keys_skip_the_array() {
    assert_eq!(
        run("local t = {} t[1<<40] = 1 t[1<<62] = 2 return #t + t[1<<40] + t[2^40]"),
        2
    );
}

#[test]
fn out_of_order_fill_joins_the_array() {
    assert_eq!(run("local r = {} r[3]=3 r[2]=2 r[1]=1 return #r"), 3);
    assert_eq!(run("local h = {1,2,3} h[5]=5 h[4]=4 return #h"), 5);
}

#[test]
fn integral_float_key_is_the_integer_key() {
    assert_eq!(run("local u = {} u[-(1<<40)] = 7 return u[-2^40]"), 7);
}
