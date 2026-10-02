//! What a host sees from `Context::load`.

use tcvm::Lua;

fn load_err(src: &str) -> String {
    let mut lua = Lua::new();
    lua.enter(|ctx| match ctx.load(src, Some("=c")) {
        Ok(_) => panic!("{src:?} loaded"),
        Err(e) => e.to_string(),
    })
}

#[test]
fn parse_error_is_the_rendered_report() {
    assert_eq!(
        load_err("x = = 1"),
        "Error: expected a statement\n   \
         ╭─[ c:1:5 ]\n   \
         │\n \
         1 │ x = = 1\n   \
         │     ┬  \n   \
         │     ╰── expected a statement but got \"=\"\n\
         ───╯"
    );
}
